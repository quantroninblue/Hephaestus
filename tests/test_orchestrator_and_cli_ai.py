from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.ai.orchestrator import HephaestusOrchestrator, OrchestrationError
from hephaestus.ai.types import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderErrorDetails,
    ProviderToolUseError,
    TokenUsage,
    ToolCall,
)
from hephaestus.changes import fingerprint_project
from hephaestus.fixtures import representative_project
from hephaestus.generation import GenerationError
from hephaestus.model import ProjectMetadata
from hephaestus.persistence import load_project, save_project
from hephaestus.tools.project import default_project_tool_registry
from hephaestus.tools.registry import (
    ProjectToolExecutor,
    ToolContext,
    mobile_robot_design_tool_policy,
)

runner = CliRunner()

MARKDOWN_RESPONSE = """### Torque Summary

A **bold** load note.

| Axis | Torque |
| --- | ---: |
| yaw | 4 N*m |
"""


class ScriptedProvider:
    provider_id = "fake"
    model_id = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def capabilities(self):
        return ProviderCapabilities(tool_calls=True, streaming=False)

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            return ModelResponse("done", FinishReason.STOP)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        pass


def _payload_requirement_args() -> dict:
    return {
        "statement": "The robot must carry a payload of at least 20 kg.",
        "quantity_value": 20.0,
        "quantity_unit": "kg",
        "constraint": "at_least",
        "kind": "payload",
        "priority": "medium",
        "verification_method": "unspecified",
    }


def _mobile_robot_design_args() -> dict:
    return {
        "robot_family": "differential_drive",
        "robot_name": "Aisle Scout",
        "mission_summary": "inspect pallet aisles for blocked exits",
        "payload_kg": 14.0,
        "max_speed_mps": 1.3,
        "include_lidar": True,
        "include_imu": True,
        "indoor_warehouse": True,
        "explicitly_supersedes_payload": True,
    }


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("y", True),
        ("Y", True),
        ("yes", True),
        ("YES", True),
        ("n", False),
        ("N", False),
        ("no", False),
        ("NO", False),
        ("", False),
        ("sure", False),
    ],
)
def test_confirmation_parser_accepts_only_explicit_yes(
    monkeypatch, answer: str, expected: bool
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    assert cli._confirm_yes_only("Apply?") is expected


def test_confirmation_parser_treats_eof_as_decline(monkeypatch) -> None:
    def raise_eof(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert cli._confirm_yes_only("Apply?") is False


@pytest.mark.parametrize("answer", ["n", "N", "no", "NO", ""])
def test_cli_chat_declines_mutation_for_supported_decline_inputs(
    monkeypatch, tmp_path: Path, answer: str
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / f"config_{answer or 'blank'}"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / f"project_{answer or 'blank'}"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    before = load_project(project_dir)
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)
    input_text = f"add requirement\n{answer}\n/exit\n"

    result = runner.invoke(cli.app, ["chat", "--project", str(project_dir)], input=input_text)

    assert result.exit_code == 0
    assert "Apply this change?" in result.output
    assert "Change not applied." in result.output
    assert "Applied." not in result.output
    assert len(fake.requests) == 1
    assert project_file.read_bytes() == before_bytes
    assert load_project(project_dir).revision == before.revision
    assert len(load_project(project_dir).requirements) == len(before.requirements)


@pytest.mark.parametrize("answer", ["yes", "YES"])
def test_cli_chat_yes_word_applies_mutation(monkeypatch, tmp_path: Path, answer: str) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / f"config_{answer}"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / f"project_{answer}"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("No change was applied.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(
        cli.app, ["chat", "--project", str(project_dir)], input=f"add\n{answer}\n/exit\n"
    )

    assert result.exit_code == 0
    assert "Change applied." in result.output
    assert "No change was applied." not in result.output
    assert len(fake.requests) == 1
    assert project_file.read_bytes() != before_bytes
    assert load_project(project_dir).revision == 1


def test_orchestrator_decline_no_second_confirmation_or_provider_followup(tmp_path: Path) -> None:
    project_dir = tmp_path / "project_decline_terminal"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_req", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse(
                "I’m ready to add the requirement. Please confirm you’d like to apply this change.",
                FinishReason.STOP,
            ),
        ]
    )
    confirm_calls = 0

    def decline(preview: dict) -> bool:
        nonlocal confirm_calls
        confirm_calls += 1
        return False

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Add requirement", allow_mutation=True, confirm=decline)

    assert result.content == "Change not applied."
    assert "confirm" not in result.content.lower()
    assert len(provider.requests) == 1
    assert confirm_calls == 1
    assert project_file.read_bytes() == before_bytes


def test_orchestrator_project_tool_history_and_no_system_duplication(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("call_1", "project.get_summary", {}),),
            ),
            ModelResponse("The project is revision 0.", FinishReason.STOP, TokenUsage(1, 2, 3)),
        ]
    )
    orchestrator = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    )

    result = orchestrator.ask("What is this project?", allow_mutation=False)

    assert result.content == "The project is revision 0."
    assert result.tool_activity[0].name == "project.get_summary"
    assert provider.requests[0].system_instruction.count("Your name is Hephaestus") == 1
    assert all(
        message.content != provider.requests[0].system_instruction
        for message in orchestrator.history
    )


def test_orchestrator_declined_and_confirmed_mutation(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    before = fingerprint_project(load_project(project_dir))
    call = ToolCall("call_1", "project_add_requirement", _payload_requirement_args())

    declined_provider = ScriptedProvider(
        [
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,)),
            ModelResponse("No change was applied.", FinishReason.STOP),
        ]
    )
    declined = HephaestusOrchestrator(
        provider=declined_provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Add one", allow_mutation=True, confirm=lambda preview: False)
    assert declined.content == "Change not applied."
    assert len(declined_provider.requests) == 1
    assert declined.mutation_outcome is not None
    assert declined.mutation_outcome.status == "declined"
    assert fingerprint_project(load_project(project_dir)) == before

    confirmed_provider = ScriptedProvider(
        [
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,)),
            ModelResponse("No change was applied.", FinishReason.STOP),
        ]
    )
    confirmed = HephaestusOrchestrator(
        provider=confirmed_provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Add one", allow_mutation=True, confirm=lambda preview: True)
    assert confirmed.content == "Change applied."
    assert len(confirmed_provider.requests) == 1
    assert confirmed.mutation_outcome is not None
    assert confirmed.mutation_outcome.status == "applied"
    assert confirmed.mutation_outcome.applied is True
    assert confirmed.mutation_outcome.before_revision == 0
    assert confirmed.mutation_outcome.after_revision == 1
    assert (
        confirmed.mutation_outcome.before_fingerprint
        != confirmed.mutation_outcome.after_fingerprint
    )
    assert load_project(project_dir).revision == 1
    assert len(load_project(project_dir).requirements) == 3
    assert fingerprint_project(load_project(project_dir)) != before


def test_orchestrator_repeated_tool_call_stops(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    call = ToolCall("call_1", "project.get_summary", {})
    provider = ScriptedProvider(
        [
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,)),
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,)),
        ]
    )
    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Loop", allow_mutation=False)
    assert "repeated" in result.content.lower()


def test_cli_ai_status_configure_and_ask_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider(
        [ModelResponse("Hello engineer.", FinishReason.STOP, TokenUsage(1, 1, 2))]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    configure = runner.invoke(cli.app, ["ai", "configure"])
    assert configure.exit_code == 0
    assert "GROQ_API_KEY" in configure.output
    assert "gsk_test_secret" not in configure.output

    status = runner.invoke(cli.app, ["ai", "status", "--json"])
    assert status.exit_code == 0
    payload = json.loads(status.output)
    assert payload["credential_present"] is True
    assert payload["max_completion_tokens"] == "provider default"
    assert "gsk_test_secret" not in status.output
    assert "\x1b[" not in status.output

    asked = runner.invoke(cli.app, ["ask", "Explain torque", "--json"])
    assert asked.exit_code == 0
    answer = json.loads(asked.output)
    assert answer["content"] == "Hello engineer."
    assert answer["provider"] == "fake"
    assert "gsk_test_secret" not in asked.output


def test_cli_ask_human_renders_markdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider([ModelResponse(MARKDOWN_RESPONSE, FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, ["ask", "Explain torque"])

    assert result.exit_code == 0
    assert "Torque Summary" in result.output
    assert "bold load note" in result.output
    assert "yaw" in result.output
    assert "**bold**" not in result.output
    assert "###" not in result.output
    assert "|" not in result.output


def test_cli_chat_human_renders_markdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider([ModelResponse(MARKDOWN_RESPONSE, FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, ["chat"], input="explain\n/exit\n")

    assert result.exit_code == 0
    assert "Torque Summary" in result.output
    assert "bold load note" in result.output
    assert "yaw" in result.output
    assert "**bold**" not in result.output
    assert "###" not in result.output
    assert "| Axis | Torque |" not in result.output
    assert "| --- | ---: |" not in result.output
    assert "| yaw | 4 N*m |" not in result.output


def test_cli_ask_json_preserves_model_markdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider([ModelResponse(MARKDOWN_RESPONSE, FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, ["ask", "Explain torque", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content"] == MARKDOWN_RESPONSE
    assert "\x1b[" not in result.output


def test_cli_ask_noninteractive_mutation_denied(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    before = fingerprint_project(load_project(project_dir))
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, ["ask", "Add entity", "--project", str(project_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content"].startswith("Unexpected tool requested")
    assert payload["tool_activity"][0]["status"] == "unexpected"
    assert "mutation_outcome" not in payload
    assert "\x1b[" not in result.output
    assert fingerprint_project(load_project(project_dir)) == before


def test_cli_chat_confirm_yes_and_no(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    before = fingerprint_project(load_project(project_dir))

    no_provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: no_provider)
    denied = runner.invoke(
        cli.app, ["chat", "--project", str(project_dir)], input="add\nn\n/exit\n"
    )
    assert denied.exit_code == 0
    assert "Proposed Change" in denied.output
    assert "Add requirement" in denied.output
    assert "20 kg" in denied.output
    assert "Current Revision" in denied.output
    assert "Proposed Revision" in denied.output
    assert "Before Fingerprint" not in denied.output
    assert "After Fingerprint" not in denied.output
    assert "Apply this change?" in denied.output
    assert "Change not applied." in denied.output
    assert "Applied." not in denied.output
    assert len(no_provider.requests) == 1
    assert fingerprint_project(load_project(project_dir)) == before

    yes_provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: yes_provider)
    confirmed = runner.invoke(
        cli.app, ["chat", "--project", str(project_dir)], input="add\ny\n/exit\n"
    )
    assert confirmed.exit_code == 0
    assert "Apply this change?" in confirmed.output
    assert "Change applied." in confirmed.output
    assert "No change was applied." not in confirmed.output
    assert len(yes_provider.requests) == 1
    assert load_project(project_dir).revision == 1


def test_cli_chat_provider_rejection_after_confirmed_mutation_is_visible_and_recoverable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Recovered.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(
        cli.app,
        ["chat", "--project", str(project_dir)],
        input="add\ny\nstatus\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Change applied." in result.output
    assert "Recovered." in result.output
    assert len(fake.requests) == 2
    assert load_project(project_dir).revision == 1


def test_orchestrator_accepted_requirement_mutation_preserves_tool_order(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    before_bytes = (project_dir / "hephaestus.project.json").read_bytes()
    before_project = load_project(project_dir)
    before_fingerprint = fingerprint_project(before_project)
    before_requirement_count = len(before_project.requirements)
    call = ToolCall(
        "call_req",
        "project_add_requirement",
        _payload_requirement_args(),
    )
    provider = ScriptedProvider(
        [
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,)),
            ModelResponse("No change was applied.", FinishReason.STOP),
        ]
    )
    previews: list[dict] = []
    orchestrator = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    )

    result = orchestrator.ask(
        "Add a requirement that the robot must carry a payload of at least 20 kg.",
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or True,
    )

    project = load_project(project_dir)
    assert result.content == "Change applied."
    assert len(provider.requests) == 1
    assert previews and previews[0]["status"] == "ok"
    assert project.revision == 1
    assert len(project.requirements) == before_requirement_count + 1
    inserted = next(
        requirement
        for requirement in project.requirements
        if requirement.id == "requirement:the_robot_must_carry_a_payload_of_at_least_20_kg"
    )
    assert inserted.value is not None
    assert inserted.value.to_dict()["minimum"]["value"] == 20.0
    assert fingerprint_project(project) != before_fingerprint
    assert (project_dir / "hephaestus.project.json").read_bytes() != before_bytes
    assert provider.requests[0].messages[-1].content.startswith("Add a requirement")
    event_names = {event.event for event in result.events}
    assert "confirmation_accepted" in event_names
    assert "mutation_applied" in event_names


def test_orchestrator_declined_requirement_mutation_preserves_bytes(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    call = ToolCall(
        "call_req",
        "project_add_requirement",
        _payload_requirement_args(),
    )
    provider = ScriptedProvider(
        [
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,)),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Add requirement", allow_mutation=True, confirm=lambda preview: False)

    assert result.content == "Change not applied."
    assert len(provider.requests) == 1
    assert project_file.read_bytes() == before_bytes
    payload = result.to_dict()["mutation_outcome"]
    assert set(payload) == {
        "attempted",
        "tool",
        "tool_call_id",
        "permission",
        "applied",
        "status",
        "reason",
        "before_revision",
        "before_fingerprint",
        "after_revision",
        "after_fingerprint",
    }
    assert payload["attempted"] is True
    assert payload["tool"] == "project_add_requirement"
    assert payload["tool_call_id"] == "call_req"
    assert payload["permission"] == "project mutation"
    assert payload["status"] == "declined"
    assert payload["applied"] is False
    assert payload["reason"] == "change was previewed but not applied"
    assert payload["before_revision"] == payload["after_revision"] == 0
    assert payload["before_fingerprint"] == payload["after_fingerprint"]


def test_declined_outcome_reports_actual_state_if_confirmation_changes_project(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project_decline_external_change"
    save_project(representative_project(), project_dir)
    call = ToolCall("call_req", "project_add_requirement", _payload_requirement_args())
    provider = ScriptedProvider([ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=(call,))])

    def decline_after_external_change(preview: dict) -> bool:
        project = load_project(project_dir)
        save_project(
            replace(project, metadata=ProjectMetadata(project.metadata.name, "external")),
            project_dir,
            overwrite=True,
        )
        return False

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Add requirement", allow_mutation=True, confirm=decline_after_external_change)

    outcome = result.to_dict()["mutation_outcome"]
    assert outcome["status"] == "declined"
    assert outcome["applied"] is False
    assert outcome["before_fingerprint"] != outcome["after_fingerprint"]
    assert load_project(project_dir).revision == 0


def test_orchestrator_malformed_mutation_returns_tool_error_without_write(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "bad_args",
                        "project_add_requirement",
                        {"statement": "missing quantity"},
                    ),
                ),
            ),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Bad mutation", allow_mutation=True, confirm=lambda preview: True)

    assert result.content.startswith("Change failed:")
    assert len(provider.requests) == 1
    assert project_file.read_bytes() == before_bytes
    payload = result.to_dict()["mutation_outcome"]
    assert set(payload) == {
        "attempted",
        "tool",
        "tool_call_id",
        "permission",
        "applied",
        "status",
        "reason",
        "before_revision",
        "before_fingerprint",
        "after_revision",
        "after_fingerprint",
    }
    assert payload["status"] == "failed"
    assert payload["applied"] is False
    assert payload["reason"] == "invalid arguments"
    assert payload["before_revision"] == payload["after_revision"] == 0
    assert payload["before_fingerprint"] == payload["after_fingerprint"]


def test_provider_failure_after_declined_mutation_reports_no_change(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_req", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Applied.", FinishReason.STOP),
        ]
    )

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Add requirement", allow_mutation=True, confirm=lambda preview: False)

    assert result.content == "Change not applied."
    assert result.finish_reason is FinishReason.ERROR
    assert len(provider.requests) == 1
    assert project_file.read_bytes() == before


def test_orchestrator_empty_provider_response_is_explicit_error(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    provider = ScriptedProvider([ModelResponse("", FinishReason.STOP)])
    orchestrator = HephaestusOrchestrator(provider=provider, project_path=project_dir)

    try:
        orchestrator.ask("hello")
    except OrchestrationError as exc:
        assert "no assistant text or tool calls" in str(exc)
    else:
        raise AssertionError("empty response should raise OrchestrationError")
    assert any(event.event == "invalid_empty_response" for event in orchestrator.events)


def test_cli_ask_empty_provider_response_prints_visible_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider([ModelResponse("", FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, ["ask", "empty"])

    assert result.exit_code == 1
    assert "Provider returned no assistant text or tool calls" in result.output


def test_orchestrator_multiple_read_only_tools_then_mutation_keep_ids(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    calls = (
        ToolCall("call_summary", "project.get_summary", {}),
        ToolCall("call_validate", "project.validate", {}),
        ToolCall("call_apply", "project_add_requirement", _payload_requirement_args()),
    )
    provider = ScriptedProvider(
        [
            ModelResponse("", FinishReason.TOOL_CALLS, tool_calls=calls),
            ModelResponse("Done.", FinishReason.STOP),
        ]
    )

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Inspect then mutate", allow_mutation=True, confirm=lambda preview: True)

    assert result.content == "Change applied."
    assert len(provider.requests) == 1
    assert [item.name for item in result.tool_activity] == [
        "project.get_summary",
        "project.validate",
        "project_add_requirement",
    ]
    assert load_project(project_dir).revision == 1


def test_cli_chat_failed_mutation_is_authoritative_and_recoverable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_failed_mutation"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "project_failed_mutation"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "bad_args",
                        "project_add_requirement",
                        {"statement": "missing quantity"},
                    ),
                ),
            ),
            ModelResponse("Recovered.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(
        cli.app,
        ["chat", "--project", str(project_dir)],
        input="bad mutation\nWhat is 2 + 2?\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Change failed: invalid arguments" in result.output
    assert "Recovered." in result.output
    assert len(fake.requests) == 2
    assert project_file.read_bytes() == before_bytes
    assert load_project(project_dir).revision == 0


def test_cli_chat_declined_mutation_remains_usable_without_misleading_followup(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_recovery"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "project_recovery"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("call_1", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
            ModelResponse("Recovered.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(
        cli.app,
        ["chat", "--project", str(project_dir)],
        input="add requirement\nn\nWhat is 2 + 2?\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Change not applied." in result.output
    assert "Recovered." in result.output
    assert len(fake.requests) == 2
    assert project_file.read_bytes() == before_bytes
    assert load_project(project_dir).revision == 0


def test_cli_chat_tool_use_failed_is_visible_redacted_and_recoverable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    before = (project_dir / "hephaestus.project.json").read_bytes()
    fake = ScriptedProvider(
        [
            ProviderToolUseError(
                "Groq could not produce a valid engineering operation. No changes were made.",
                ProviderErrorDetails(
                    status_code=400,
                    provider_code="tool_use_failed",
                    provider_error_type="invalid_request_error",
                    request_id="req_test_123",
                    retryable=False,
                ),
            ),
            ModelResponse("Recovered.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(
        cli.app,
        ["chat", "--project", str(project_dir)],
        input="add requirement\nWhat is 2 + 2?\n/exit\n",
    )

    assert result.exit_code == 0
    assert "valid engineering operation" in result.output
    assert "tool_use_failed" in result.output
    assert "failed_generation" not in result.output
    assert "gsk_test_secret" not in result.output
    assert "Recovered." in result.output
    assert (project_dir / "hephaestus.project.json").read_bytes() == before


def test_cli_chat_remains_usable_after_empty_response_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    fake = ScriptedProvider(
        [
            ModelResponse("", FinishReason.STOP),
            ModelResponse("Recovered.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, ["chat"], input="bad\ngood\n/exit\n")

    assert result.exit_code == 0
    assert "Provider returned no assistant text or tool calls" in result.output
    assert "Recovered." in result.output


def test_orchestrator_expected_tool_rejects_wrong_design_tool_without_write(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "wrong_tool_project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("wrong_tool", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
        ]
    )
    confirmations: list[dict] = []

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask(
        "Design the warehouse inspection robot",
        allow_mutation=True,
        confirm=lambda preview: confirmations.append(preview) or True,
        expected_tool_names=("project_create_mobile_robot",),
    )

    assert result.finish_reason is FinishReason.ERROR
    assert result.content == (
        "Unexpected tool requested: expected project_create_mobile_robot; "
        "got project_add_requirement."
    )
    assert result.mutation_outcome is None
    assert [item.to_dict() for item in result.tool_activity] == [
        {"name": "project_add_requirement", "status": "unexpected"}
    ]
    assert confirmations == []
    assert project_file.read_bytes() == before
    assert [tool.name for tool in provider.requests[0].tools] == ["project_create_mobile_robot"]


def test_cli_design_rejects_wrong_tool_regression_without_write(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_design_wrong_tool"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "wrong_tool_cli_project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("wrong_tool", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)
    generation_calls: list[Path] = []

    def should_not_generate(path: Path, **kwargs):
        del kwargs
        generation_calls.append(path)
        raise AssertionError("generation must not start after a wrong design operation")

    monkeypatch.setattr(cli, "generate_project_artifacts", should_not_generate)

    result = runner.invoke(
        cli.app,
        [
            "design",
            "Create a differential-drive warehouse inspection robot",
            "--project",
            str(project_dir),
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "wrong_operation_rejected"
    assert payload["generated"] is False
    assert payload["applied"] is False
    assert payload["finish_reason"] == "error"
    assert "project_create_mobile_robot" in payload["content"]
    assert "project_add_requirement" in payload["content"]
    assert payload["tool_activity"] == [
        {"name": "project_add_requirement", "status": "unexpected"}
    ]
    assert generation_calls == []
    assert project_file.read_bytes() == before
    assert load_project(project_dir).revision == 0
    assert [tool.name for tool in fake.requests[0].tools] == [
        "project.get_summary",
        "project.validate",
        "project_create_mobile_robot",
    ]




def test_cli_design_json_without_yes_returns_confirmation_required(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_design_confirm"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "confirm_required_project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("design", "project_create_mobile_robot", _mobile_robot_design_args()),
                ),
            ),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(
        cli.app,
        [
            "design",
            "Create a differential-drive warehouse inspection robot",
            "--project",
            str(project_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "confirmation_required"
    assert payload["operation"] == "project_create_mobile_robot"
    assert payload["applied"] is False
    assert payload["generated"] is False
    assert "Proposed Change" not in result.output
    assert "\x1b[" not in result.output
    assert project_file.read_bytes() == before


def test_cli_design_accept_generates_only_after_verified_mobile_robot_commit(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_design_accept"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "accepted_design_project"
    save_project(representative_project(), project_dir)
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("design", "project_create_mobile_robot", _mobile_robot_design_args()),
                ),
            ),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    class ReadyCapabilities:
        def require(self, capability_ids):
            self.capability_ids = tuple(capability_ids)
            return ()

    ready = ReadyCapabilities()
    monkeypatch.setattr(cli, "collect_capabilities", lambda: ready)
    generated_after_revision: list[int] = []

    class FakeGeneration:
        def to_dict(self):
            return {
                "status": "generated",
                "manifest": {"project_revision": 1, "artifacts": []},
            }

    def generate(path: Path, **kwargs):
        del kwargs
        project = load_project(path)
        generated_after_revision.append(project.revision)
        assert "hephaestus:mobile_robot:v1" in project.extensions
        return FakeGeneration()

    monkeypatch.setattr(cli, "generate_project_artifacts", generate)

    result = runner.invoke(
        cli.app,
        [
            "design",
            "Create a differential-drive warehouse inspection robot",
            "--project",
            str(project_dir),
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "completed"
    assert payload["operation"] == "project_create_mobile_robot"
    assert payload["applied"] is True
    assert payload["generated"] is True
    assert payload["mutation_outcome"]["tool"] == "project_create_mobile_robot"
    assert payload["artifact_manifest_path"].endswith("generated/hephaestus.artifacts.json")
    assert generated_after_revision == [1]
    assert ready.capability_ids == (
        "cad",
        "ros2",
        "gazebo",
        "ros_gz_sim",
        "ros_gz_bridge",
        "gz_ros2_control",
    )
    assert load_project(project_dir).revision == 1


def test_cli_design_generation_failure_preserves_committed_robot(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_design_gen_fail"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "generation_failure_project"
    save_project(representative_project(), project_dir)
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("design", "project_create_mobile_robot", _mobile_robot_design_args()),
                ),
            ),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    class ReadyCapabilities:
        def require(self, capability_ids):
            del capability_ids
            return ()

    monkeypatch.setattr(cli, "collect_capabilities", lambda: ReadyCapabilities())

    def fail_generation(path: Path, **kwargs):
        del path, kwargs
        raise GenerationError("fake CAD backend unavailable")

    monkeypatch.setattr(cli, "generate_project_artifacts", fail_generation)

    result = runner.invoke(
        cli.app,
        [
            "design",
            "Create a differential-drive warehouse inspection robot",
            "--project",
            str(project_dir),
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "generation_failed"
    assert payload["applied"] is True
    assert payload["generated"] is False
    assert payload["safe_failure_reason"] == "fake CAD backend unavailable"
    project = load_project(project_dir)
    assert project.revision == 1
    assert "hephaestus:mobile_robot:v1" in project.extensions
    assert not (project_dir / "generated" / "hephaestus.artifacts.json").exists()



def test_cli_design_capability_gate_blocks_generation_after_commit(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_design_cap_gate"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "capability_gate_project"
    save_project(representative_project(), project_dir)
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("design", "project_create_mobile_robot", _mobile_robot_design_args()),
                ),
            ),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    class Status:
        value = "missing"

    class Compatibility:
        value = "unverified"

    class Failure:
        capability_id = "cad"
        status = Status()
        compatibility = Compatibility()

    class MissingCadCapabilities:
        def require(self, capability_ids):
            self.capability_ids = tuple(capability_ids)
            return (Failure(),)

    capability_report = MissingCadCapabilities()
    monkeypatch.setattr(cli, "collect_capabilities", lambda: capability_report)
    generation_calls: list[Path] = []

    def should_not_generate(path: Path, **kwargs):
        generation_calls.append(path)
        raise AssertionError(kwargs)

    monkeypatch.setattr(cli, "generate_project_artifacts", should_not_generate)

    result = runner.invoke(
        cli.app,
        [
            "design",
            "Create a differential-drive warehouse inspection robot",
            "--project",
            str(project_dir),
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "generation_failed"
    assert payload["applied"] is True
    assert payload["generated"] is False
    assert "cad=missing/unverified" in payload["safe_failure_reason"]
    assert generation_calls == []
    assert capability_report.capability_ids == (
        "cad",
        "ros2",
        "gazebo",
        "ros_gz_sim",
        "ros_gz_bridge",
        "gz_ros2_control",
    )
    assert load_project(project_dir).revision == 1



def test_design_workflow_rejects_mixed_tool_call_batch_before_any_mutation(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "mixed_tool_project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("design", "project_create_mobile_robot", _mobile_robot_design_args()),
                    ToolCall("wrong", "project_add_requirement", _payload_requirement_args()),
                ),
            ),
        ]
    )
    confirmations: list[dict] = []

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(),
            ToolContext(project_dir),
            workflow_policy=mobile_robot_design_tool_policy(),
        ),
        project_path=project_dir,
    ).ask(
        "Design the warehouse inspection robot",
        allow_mutation=True,
        confirm=lambda preview: confirmations.append(preview) or True,
    )

    assert result.finish_reason is FinishReason.ERROR
    assert "project_add_requirement" in result.content
    assert result.mutation_outcome is None
    assert confirmations == []
    assert project_file.read_bytes() == before
    assert load_project(project_dir).revision == 0



def test_cli_design_non_generated_result_reports_generation_failed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config_design_invalid_gen"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    project_dir = tmp_path / "invalid_generation_result_project"
    save_project(representative_project(), project_dir)
    fake = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("design", "project_create_mobile_robot", _mobile_robot_design_args()),
                ),
            ),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    class ReadyCapabilities:
        def require(self, capability_ids):
            del capability_ids
            return ()

    monkeypatch.setattr(cli, "collect_capabilities", lambda: ReadyCapabilities())

    class InvalidGeneration:
        def to_dict(self):
            return {"status": "invalid", "manifest": None}

    def invalid_generation(path: Path, **kwargs):
        del path, kwargs
        return InvalidGeneration()

    monkeypatch.setattr(cli, "generate_project_artifacts", invalid_generation)

    result = runner.invoke(
        cli.app,
        [
            "design",
            "Create a differential-drive warehouse inspection robot",
            "--project",
            str(project_dir),
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "generation_failed"
    assert payload["applied"] is True
    assert payload["generated"] is False
    assert payload["safe_failure_reason"] == "artifact generation did not complete"
    assert "artifact_manifest_path" not in payload

def test_orchestrator_accepted_mobile_robot_design_mutation_is_terminal(tmp_path: Path) -> None:
    project_dir = tmp_path / "design_project"
    save_project(representative_project(), project_dir)
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "call_design",
                        "project_create_mobile_robot",
                        _mobile_robot_design_args(),
                    ),
                ),
            ),
            ModelResponse("I applied it.", FinishReason.STOP),
        ]
    )
    previews: list[dict] = []

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask(
        "Design the warehouse inspection robot",
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or True,
    )

    project = load_project(project_dir)
    assert result.content == "Change applied."
    assert len(provider.requests) == 1
    assert previews and previews[0]["result"]["operation"] == "create_mobile_robot"
    assert result.mutation_outcome is not None
    assert result.mutation_outcome.status == "applied"
    assert project.revision == 1
    assert any(entity.id == "entity:aisle_scout" for entity in project.entities)


def test_orchestrator_declined_mobile_robot_design_preserves_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "design_declined"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "call_design",
                        "project_create_mobile_robot",
                        _mobile_robot_design_args(),
                    ),
                ),
            ),
            ModelResponse("I applied it.", FinishReason.STOP),
        ]
    )

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask(
        "Design the warehouse inspection robot", allow_mutation=True, confirm=lambda preview: False
    )

    assert result.content == "Change not applied."
    assert len(provider.requests) == 1
    assert project_file.read_bytes() == before
    assert result.mutation_outcome is not None
    assert result.mutation_outcome.status == "declined"


def test_orchestrator_malformed_mobile_robot_design_fails_without_write(tmp_path: Path) -> None:
    project_dir = tmp_path / "design_malformed"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before = project_file.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "bad_design",
                        "project_create_mobile_robot",
                        {"robot_name": "Aisle Scout"},
                    ),
                ),
            ),
            ModelResponse("I applied it.", FinishReason.STOP),
        ]
    )

    result = HephaestusOrchestrator(
        provider=provider,
        tool_executor=ProjectToolExecutor(
            default_project_tool_registry(), ToolContext(project_dir)
        ),
        project_path=project_dir,
    ).ask("Bad design", allow_mutation=True, confirm=lambda preview: True)

    assert result.content == "Change failed: invalid arguments"
    assert len(provider.requests) == 1
    assert project_file.read_bytes() == before
    assert result.mutation_outcome is not None
    assert result.mutation_outcome.status == "failed"
