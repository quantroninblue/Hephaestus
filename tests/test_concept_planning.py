from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.ai.types import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderErrorDetails,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderStructuredGenerationError,
    StructuredPlanningRequest,
    StructuredPlanningResponse,
)
from hephaestus.concept_planning import (
    CONCEPT_SCHEMA_NAME,
    MAX_PROVIDER_ATTEMPTS,
    ConceptPlanner,
    ConceptPlanningError,
    audit_structured_json_schema,
)
from hephaestus.concept_robot import (
    MAX_BODIES,
    MAX_JOINTS,
    MAX_SENSORS,
    ConceptBodySpec,
    ConceptColor,
    ConceptJointType,
    ConceptPose,
    ConceptRobotSpec,
    ConceptWorld,
    FunctionalRole,
    LocomotionMode,
    PrimitiveType,
    WheelRole,
    concept_robot_schema,
)
from hephaestus.fixtures import blocky_yellow_warehouse_utility_robot_spec
from hephaestus.model import EngineeringProject
from hephaestus.persistence import load_project, save_project

runner = CliRunner()


class ScriptedStructuredProvider:
    provider_id = "fake"
    model_id = "fake-model"

    def __init__(self, responses: list[StructuredPlanningResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredPlanningRequest] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(tool_calls=False, streaming=False)

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("concept planning must use structured completion")

    def complete_structured(self, request: StructuredPlanningRequest) -> StructuredPlanningResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected provider request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        pass


def _structured_failure() -> ProviderStructuredGenerationError:
    return ProviderStructuredGenerationError(
        "AI provider could not produce schema-valid structured output",
        ProviderErrorDetails(
            status_code=400,
            provider_code="json_validate_failed",
            provider_error_type="invalid_request_error",
            request_id="req_test",
            retryable=False,
        ),
    )


def _account_limit_failure() -> ProviderRateLimitError:
    return ProviderRateLimitError(
        "AI provider account limit reached",
        ProviderErrorDetails(
            status_code=413,
            provider_code="rate_limit_exceeded",
            provider_error_type="tokens",
            request_id="req_limit",
            retryable=False,
        ),
    )


def _response(payload: dict[str, Any] | None = None) -> StructuredPlanningResponse:
    return StructuredPlanningResponse(
        payload=payload or blocky_yellow_warehouse_utility_robot_spec().model_dump(mode="json"),
        raw_provider="fake",
    )


def _minimal_static_spec() -> ConceptRobotSpec:
    return ConceptRobotSpec(
        spec_version="hephaestus.concept_robot.v1",
        robot_name="Static Bot",
        design_summary="Compact static concept robot.",
        locomotion_mode=LocomotionMode.STATIC,
        bodies=(
            ConceptBodySpec(
                semantic_key="base",
                display_name="Base",
                primitive_type=PrimitiveType.BOX,
                size_x_m=0.4,
                size_y_m=0.3,
                size_z_m=0.2,
                radius_m=0.0,
                length_m=0.0,
                pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.1),
                mass_kg=5.0,
                visual_color=ConceptColor(r=0.9, g=0.7, b=0.1, a=1.0),
                functional_role=FunctionalRole.CHASSIS,
            ),
        ),
        joints=(),
        sensors=(),
        world_template=ConceptWorld.WAREHOUSE,
        assumptions=(),
        engineering_warnings=(),
    )


def _invalid_semantic_payload() -> dict[str, Any]:
    payload = blocky_yellow_warehouse_utility_robot_spec().model_dump(mode="json")
    payload["joints"][0]["child_body_key"] = "missing_body"
    return payload


def _maximum_supported_payload() -> dict[str, Any]:
    fixture = blocky_yellow_warehouse_utility_robot_spec().model_dump(mode="json")
    body_template = fixture["bodies"][0]
    joint_template = fixture["joints"][0]
    sensor_template = fixture["sensors"][0]
    bodies = []
    for index in range(MAX_BODIES):
        body = dict(body_template)
        body["semantic_key"] = f"body_{index}"
        body["display_name"] = f"Body {index}"
        bodies.append(body)
    joints = []
    for index in range(MAX_JOINTS):
        joint = dict(joint_template)
        joint["semantic_key"] = f"joint_{index}"
        joint["parent_body_key"] = "body_0"
        joint["child_body_key"] = f"body_{(index % (MAX_BODIES - 1)) + 1}"
        joint["joint_type"] = ConceptJointType.FIXED.value
        joint["wheel_role"] = WheelRole.NONE.value
        joints.append(joint)
    sensors = []
    for index in range(MAX_SENSORS):
        sensor = dict(sensor_template)
        sensor["semantic_key"] = f"sensor_{index}"
        sensor["parent_body_key"] = f"body_{index % MAX_BODIES}"
        sensor["topic"] = f"/sensor_{index}"
        sensors.append(sensor)
    return {
        **fixture,
        "robot_name": "Max Supported Concept",
        "design_summary": "A bounded maximum-size concept payload for output-budget sizing.",
        "locomotion_mode": LocomotionMode.STATIC.value,
        "bodies": bodies,
        "joints": joints,
        "sensors": sensors,
        "assumptions": [f"assumption {index}" for index in range(16)],
        "engineering_warnings": [f"warning {index}" for index in range(16)],
    }


def _rough_tokens(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return (len(encoded) + 3) // 4


def test_real_concept_schema_is_provider_strict_compliant() -> None:
    errors = audit_structured_json_schema(CONCEPT_SCHEMA_NAME, concept_robot_schema())

    assert errors == ()


def test_schema_audit_reports_path_specific_object_and_required_errors() -> None:
    schema = copy.deepcopy(concept_robot_schema())
    pose = schema["properties"]["bodies"]["items"]["properties"]["pose"]
    del pose["additionalProperties"]
    pose["required"].remove("yaw_rad")

    errors = audit_structured_json_schema(CONCEPT_SCHEMA_NAME, schema)

    assert "$.properties.bodies.items.properties.pose: missing additionalProperties=false" in errors
    assert (
        "$.properties.bodies.items.properties.pose.properties.yaw_rad: property is not required"
        in errors
    )


def test_schema_audit_rejects_unresolved_refs_and_unbounded_arrays() -> None:
    schema = copy.deepcopy(concept_robot_schema())
    schema["properties"]["bodies"] = {"type": "array", "items": {"$ref": "#/$defs/missing"}}

    errors = audit_structured_json_schema(CONCEPT_SCHEMA_NAME, schema)

    assert "$.properties.bodies: array schema must define bounded maxItems" in errors
    assert "$.properties.bodies.items: unresolved $ref '#/$defs/missing'" in errors


def test_planner_strict_success_uses_real_schema_and_provider_default_tokens() -> None:
    provider = ScriptedStructuredProvider([_response()])

    result = ConceptPlanner(provider).create("Design a friendly warehouse robot")

    request = provider.requests[0]
    assert result.attempts == 1
    assert result.repaired is False
    assert result.provider_attempt_modes == ("strict",)
    assert request.schema_name == CONCEPT_SCHEMA_NAME
    assert request.strict_schema is True
    assert request.max_output_tokens is None
    assert audit_structured_json_schema(request.schema_name, request.json_schema) == ()


def test_planner_sends_explicit_user_token_limit_exactly() -> None:
    provider = ScriptedStructuredProvider([_response()])

    ConceptPlanner(provider, max_completion_tokens=12345).create(
        "Design a friendly warehouse robot"
    )

    assert provider.requests[0].max_output_tokens == 12345


def test_planner_does_not_retry_provider_account_limits() -> None:
    provider = ScriptedStructuredProvider([_account_limit_failure(), _response()])

    with pytest.raises(ProviderRateLimitError) as raised:
        ConceptPlanner(provider).create("Design a friendly warehouse robot")

    assert raised.value.details is not None
    assert raised.value.details.status_code == 413
    assert raised.value.details.provider_code == "rate_limit_exceeded"
    assert len(provider.requests) == 1


def test_planner_retries_once_after_json_validate_failed_then_succeeds_strictly() -> None:
    provider = ScriptedStructuredProvider([_structured_failure(), _response()])

    result = ConceptPlanner(provider).create("Design a friendly warehouse robot")

    assert result.attempts == 2
    assert result.strict_json_validate_retries == 1
    assert result.best_effort_fallback_used is False
    assert result.provider_attempt_modes == ("strict", "strict")
    assert [request.strict_schema for request in provider.requests] == [True, True]
    assert "previous structured response could not be validated" in provider.requests[1].user_prompt


def test_planner_uses_one_best_effort_structured_fallback_after_repeated_json_validate_failed() -> (
    None
):
    provider = ScriptedStructuredProvider(
        [_structured_failure(), _structured_failure(), _response()]
    )

    result = ConceptPlanner(provider).create("Design a friendly warehouse robot")

    assert result.attempts == 3
    assert result.strict_json_validate_retries == 1
    assert result.best_effort_fallback_used is True
    assert result.provider_attempt_modes == ("strict", "strict", "best_effort")
    assert [request.strict_schema for request in provider.requests] == [True, True, False]


def test_planner_rejects_malformed_best_effort_json_after_bounded_attempts() -> None:
    provider = ScriptedStructuredProvider(
        [
            _structured_failure(),
            _structured_failure(),
            ProviderMalformedResponseError("Provider structured response was malformed JSON"),
        ]
    )

    with pytest.raises(ConceptPlanningError) as raised:
        ConceptPlanner(provider).create("Design a friendly warehouse robot")

    assert "malformed JSON" in str(raised.value)
    assert len(provider.requests) == 3
    assert "failed_generation" not in str(raised.value)


def test_planner_repairs_semantically_invalid_best_effort_output() -> None:
    provider = ScriptedStructuredProvider(
        [
            _structured_failure(),
            _structured_failure(),
            _response(_invalid_semantic_payload()),
            _response(),
        ]
    )

    result = ConceptPlanner(provider).create("Design a friendly warehouse robot")

    assert result.attempts == 4
    assert result.repaired is True
    assert result.best_effort_fallback_used is True
    assert result.provider_attempt_modes == ("strict", "strict", "best_effort", "best_effort")
    assert [request.strict_schema for request in provider.requests] == [True, True, False, False]


def test_planner_exhausts_semantic_repair_budget_without_unbounded_provider_calls() -> None:
    provider = ScriptedStructuredProvider(
        [_response(_invalid_semantic_payload())] * MAX_PROVIDER_ATTEMPTS
    )

    with pytest.raises(ConceptPlanningError, match="invalid concept robot specification"):
        ConceptPlanner(provider).create("Design a friendly warehouse robot")

    assert len(provider.requests) == 3
    assert len(provider.requests) <= MAX_PROVIDER_ATTEMPTS


def test_concept_create_provider_failures_do_not_mutate_project_or_leak_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "concept_project"
    save_project(EngineeringProject.empty(name="Concept"), project_dir)
    before = (project_dir / "hephaestus.project.json").read_bytes()
    provider = ScriptedStructuredProvider(
        [_structured_failure(), _structured_failure(), _structured_failure()]
    )
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr(cli, "_create_provider", lambda config: provider)

    result = runner.invoke(
        cli.app,
        [
            "concept",
            "create",
            "Design a friendly warehouse robot",
            "--project",
            str(project_dir),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert "failed_generation" not in result.output
    assert "gsk_test_secret" not in result.output
    assert (project_dir / "hephaestus.project.json").read_bytes() == before
    assert load_project(project_dir).revision == 0


def test_concept_create_413_does_not_retry_or_mutate_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_dir = tmp_path / "concept_project"
    save_project(EngineeringProject.empty(name="Concept"), project_dir)
    before = (project_dir / "hephaestus.project.json").read_bytes()
    provider = ScriptedStructuredProvider([_account_limit_failure(), _response()])
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr(cli, "_create_provider", lambda config: provider)

    result = runner.invoke(
        cli.app,
        [
            "concept",
            "create",
            "Design a friendly warehouse robot",
            "--project",
            str(project_dir),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["details"]["status_code"] == 413
    assert payload["details"]["provider_code"] == "rate_limit_exceeded"
    assert "failed_generation" not in result.output
    assert "gsk_test_secret" not in result.output
    assert len(provider.requests) == 1
    assert (project_dir / "hephaestus.project.json").read_bytes() == before
    assert load_project(project_dir).revision == 0
