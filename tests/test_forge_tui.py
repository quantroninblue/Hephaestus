from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any

from textual import events
from textual.geometry import Size
from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.ai.agent import AgentTurnResult
from hephaestus.ai.types import FinishReason, ProviderErrorDetails, ProviderRateLimitError, ToolCall
from hephaestus.capabilities import CapabilityReport
from hephaestus.local_workbench import LocalCommandResult
from hephaestus.tools.workspace import WorkspaceToolExecutor
from hephaestus.tui.app import ActivityEntry, ForgeApp, _PendingApproval, layout_mode_for_width
from hephaestus.tui.branding import (
    FULL_LOGO,
    NARROW_LOGO,
    NOTICE_TEXT,
    NOTICE_TITLE,
    NOTICE_URL,
    logo_for_width,
)
from hephaestus.tui.commands import LOCAL_COMMANDS, stage_for_input
from hephaestus.tui.display_paths import DisplayPathFormatter

runner = CliRunner()


class FakeTty:
    def __init__(self, value: bool) -> None:
        self.value = value

    def isatty(self) -> bool:
        return self.value


class FakeLocal:
    def __init__(self, root: Path) -> None:
        self.workspace_root = root
        self.capabilities = CapabilityReport(())
        self.dispatches: list[str] = []
        self.last_result: LocalCommandResult | None = None
        self.closed = False

    def dispatch(self, command: str, *, confirm=None) -> LocalCommandResult:
        self.dispatches.append(command)
        if command == "/exit":
            self.close()
            result = LocalCommandResult(command, "ok", "Exiting.", {}, exit_requested=True)
            self.last_result = result
            return result
        if command == "/clear":
            result = LocalCommandResult(
                command, "ok", "Conversation cleared.", {}, clear_requested=True
            )
            self.last_result = result
            return result
        if command == "/status":
            result = LocalCommandResult(
                command,
                "ok",
                f"Workspace status for {self.workspace_root}",
                {"workspace": str(self.workspace_root)},
            )
            self.last_result = result
            return result
        if command == "/legal":
            result = LocalCommandResult(
                command,
                "ok",
                "\n".join(
                    (
                        "- Repository: https://github.com/quantroninblue/Hephaestus",
                        "- README/legal overview: README.md",
                        "- SAFETY.md: SAFETY.md",
                        "- PRIVACY.md: PRIVACY.md",
                        "- SECURITY.md: SECURITY.md",
                        "- RESPONSIBLE_USE.md: RESPONSIBLE_USE.md",
                        "- THIRD_PARTY_NOTICES.md: THIRD_PARTY_NOTICES.md",
                        "- LICENSE: LICENSE",
                        "- NOTICE: NOTICE",
                    )
                ),
                {"repository": "https://github.com/quantroninblue/Hephaestus"},
            )
            self.last_result = result
            return result
        result = LocalCommandResult(command, "ok", f"{command} ok", {"command": command})
        self.last_result = result
        return result

    def quick_status(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace_root),
            "gazebo": "gazebo=inactive; active=false; world=none",
            "ros": {"runtime_state": "inactive"},
        }

    def close(self) -> None:
        self.closed = True


class FakeAgent:
    provider = "fake"
    model = "fake"

    def __init__(self, response: AgentTurnResult | Exception | None = None) -> None:
        self.response = response or AgentTurnResult(
            "answer", "fake", "fake-model", FinishReason.STOP, {}
        )
        self.calls: list[str] = []
        self.cleared = False

    def ask(self, prompt: str, *, confirm=None) -> AgentTurnResult:
        self.calls.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def clear(self) -> None:
        self.cleared = True


def _run(coro):
    return asyncio.run(coro)


def _app(
    tmp_path: Path,
    *,
    agent=None,
    no_color: bool = False,
    reduced_motion: bool = True,
    display_paths: DisplayPathFormatter | None = None,
):
    return ForgeApp(
        workspace_root=tmp_path,
        local_session=FakeLocal(tmp_path),  # type: ignore[arg-type]
        agent_session=agent,
        provider_label="fake" if agent is not None else "unavailable",
        model_label="fake-model" if agent is not None else "unavailable",
        no_color=no_color,
        reduced_motion=reduced_motion,
        display_paths=display_paths,
    )


def _startup_visible(app: ForgeApp) -> bool:
    return bool(app.query_one("#startup").display)


def test_display_path_formatter_shortens_only_home_ancestry() -> None:
    formatter = DisplayPathFormatter(home=Path("/home/test-user"))

    assert formatter.format_path(Path("/home/test-user")) == "~"
    assert (
        formatter.format_path(Path("/home/test-user/Desktop/Hephaestus"))
        == "~/Desktop/Hephaestus"
    )
    assert (
        formatter.format_path(Path("/home/test-user/Desktop/Hephaestus/models/robot.sdf"))
        == "~/Desktop/Hephaestus/models/robot.sdf"
    )
    assert formatter.format_path(Path("/home/test-user2/Desktop/Hephaestus")) == (
        "/home/test-user2/Desktop/Hephaestus"
    )


def test_forge_title_is_private_and_exact(tmp_path: Path, monkeypatch) -> None:
    workspace = Path("/home/test-user/Desktop/Hephaestus")
    titles: list[str] = []
    app = _app(
        workspace,
        display_paths=DisplayPathFormatter(home=Path("/home/test-user")),
    )
    monkeypatch.setattr(app.console, "set_window_title", titles.append)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            assert app.title == "Hephaestus"
            assert app.sub_title == ""
            assert titles == ["Hephaestus"]
            title_text = " ".join((app.title, app.sub_title, *titles))
            assert "test-user" not in title_text
            assert "robot-host" not in title_text
            assert str(workspace) not in title_text
            assert "@" not in title_text

    _run(run())


def test_home_workspace_display_paths_are_tilde_shortened_in_all_layouts() -> None:
    workspace = Path("/home/test-user/Desktop/Hephaestus")
    formatter = DisplayPathFormatter(home=Path("/home/test-user"))

    async def run_for_width(width: int, expected_mode: str) -> None:
        app = _app(workspace, display_paths=formatter)
        async with app.run_test(size=(width, 24)) as pilot:
            await pilot.pause()
            assert app.workspace_root == workspace
            assert app.local.workspace_root == workspace
            assert app.layout_mode == expected_mode
            header = app.query_one("#header").content
            status = app.query_one("#status-line").content
            assert "~/Desktop/Hephaestus" in header
            assert "~/Desktop/Hephaestus" in status
            assert "/home/test-user/Desktop/Hephaestus" not in header
            assert "/home/test-user/Desktop/Hephaestus" not in status

    _run(run_for_width(120, "full"))
    _run(run_for_width(80, "compact"))
    _run(run_for_width(60, "minimal"))


def test_tui_status_display_shortens_paths_without_mutating_result_payload() -> None:
    workspace = Path("/home/test-user/robot_project")
    app = _app(
        workspace,
        display_paths=DisplayPathFormatter(home=Path("/home/test-user")),
    )

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            app.query_one("#composer").load_text("/status")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.local.last_result is not None
            assert app.local.last_result.payload["workspace"] == str(workspace)
            assert "Workspace status for ~/robot_project" in app.entries[-1].body
            assert '"workspace": "~/robot_project"' in app.entries[-1].body
            assert str(workspace) not in app.entries[-1].body

    _run(run())


def test_approval_preview_and_errors_use_display_paths() -> None:
    workspace = Path("/home/test-user/robot_project")
    formatter = DisplayPathFormatter(home=Path("/home/test-user"))

    async def run_preview() -> None:
        app = _app(workspace, display_paths=formatter)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            pending = _PendingApproval(
                {
                    "tool": "cadquery_build",
                    "result": {
                        "workspace": str(workspace),
                        "outputs": [str(workspace / "generated/cad/robot/manifest.json")],
                        "outside": "/home/test-user2/not-home",
                    },
                },
                threading.Event(),
            )
            app._show_approval(pending)
            await pilot.pause()
            card_text = app.query_one("#approval-card").content
            assert "~/robot_project" in card_text
            assert "~/robot_project/generated/cad/robot/manifest.json" in card_text
            assert "/home/test-user/robot_project" not in card_text
            assert "/home/test-user2/not-home" in card_text

    async def run_error() -> None:
        error = ValueError(f"Cannot read {workspace / 'secret.txt'}")
        app = _app(workspace, agent=FakeAgent(error), display_paths=formatter)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            app.query_one("#composer").load_text("read it")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert "Cannot read ~/robot_project/secret.txt" in app.entries[-1].body
            assert str(workspace) not in app.entries[-1].body

    _run(run_preview())
    _run(run_error())


def test_default_interactive_entrypoint_selects_tui(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys, "stdin", FakeTty(True))
    monkeypatch.setattr(sys, "stdout", FakeTty(True))
    monkeypatch.setattr(cli, "_run_default_forge_tui", lambda **kwargs: calls.append("tui"))
    monkeypatch.setattr(cli, "_run_default_agent_shell", lambda: calls.append("plain"))

    cli._run_default_interface(plain=False, no_color=False, reduced_motion=True)

    assert calls == ["tui"]


def test_plain_and_non_tty_and_dumb_terminal_fallback(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_default_forge_tui", lambda **kwargs: calls.append("tui"))
    monkeypatch.setattr(cli, "_run_default_agent_shell", lambda: calls.append("plain"))
    monkeypatch.setattr(sys, "stdin", FakeTty(True))
    monkeypatch.setattr(sys, "stdout", FakeTty(True))
    monkeypatch.setenv("TERM", "xterm-256color")

    cli._run_default_interface(plain=True, no_color=False, reduced_motion=True)
    monkeypatch.setattr(sys, "stdin", FakeTty(False))
    cli._run_default_interface(plain=False, no_color=False, reduced_motion=True)
    monkeypatch.setattr(sys, "stdin", FakeTty(True))
    monkeypatch.setenv("TERM", "dumb")
    cli._run_default_interface(plain=False, no_color=False, reduced_motion=True)

    assert calls == ["plain", "plain", "plain"]


def test_branding_variants_and_startup_notice_exact_content(tmp_path: Path) -> None:
    assert logo_for_width(100) == FULL_LOGO
    assert logo_for_width(60) == NARROW_LOGO

    async def run() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            assert _startup_visible(app)
            assert app.query_one("#logo").content == FULL_LOGO
            assert app.query_one("#notice-title").content == NOTICE_TITLE
            assert app.query_one("#notice-text").content == NOTICE_TEXT
            url_content = app.query_one("#notice-url").content
            assert getattr(url_content, "plain", str(url_content)) == NOTICE_URL
            notice = "\n".join(entry.body for entry in app.entries)
            assert NOTICE_TITLE not in notice
            assert NOTICE_TEXT not in notice
            assert NOTICE_URL not in notice
            assert all(entry.label != "! NEEDS ATTENTION" for entry in app.entries)

    _run(run())


def test_startup_hero_persists_through_focus_refresh_and_resize(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path, reduced_motion=False)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(1.0)
            assert _startup_visible(app)
            assert app.query_one("#logo").content == FULL_LOGO

            app.query_one("#composer").focus()
            app._render_header()
            app.on_resize(events.Resize(Size(100, 24), Size(100, 24)))
            await pilot.pause()

            assert _startup_visible(app)
            assert app.query_one("#logo").content == FULL_LOGO

    _run(run())


def test_reduced_motion_startup_still_shows_full_logo(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path, reduced_motion=True)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            assert _startup_visible(app)
            assert app.query_one("#logo").content == FULL_LOGO

    _run(run())


def test_narrow_startup_logo_below_seventy_columns(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(69, 20)) as pilot:
            await pilot.pause()
            assert _startup_visible(app)
            assert app.query_one("#logo").content == NARROW_LOGO

    _run(run())


def test_startup_hero_collapses_after_first_message_or_command(tmp_path: Path) -> None:
    async def run_message() -> None:
        agent = FakeAgent()
        app = _app(tmp_path / "message", agent=agent)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert _startup_visible(app)

            app.query_one("#composer").load_text("reason about torque")
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert app.startup_collapsed is True
            assert _startup_visible(app) is False
            assert agent.calls == ["reason about torque"]

    async def run_command() -> None:
        app = _app(tmp_path / "command")
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            assert _startup_visible(app)

            app.query_one("#composer").load_text("/status")
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert app.startup_collapsed is True
            assert _startup_visible(app) is False
            assert app.local.dispatches == ["/status"]

    _run(run_message())
    _run(run_command())


def test_activity_and_clear_do_not_restore_startup_hero(tmp_path: Path) -> None:
    async def run() -> None:
        agent = FakeAgent()
        app = _app(tmp_path, agent=agent)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            app.query_one("#composer").load_text("first message")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert _startup_visible(app) is False

            app.add_activity("✓ VERIFIED", "background status refreshed")
            app.query_one("#composer").load_text("/status")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert _startup_visible(app) is False

            app.query_one("#composer").load_text("/clear")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.entries == []
            assert agent.cleared is True
            assert _startup_visible(app) is False

    _run(run())


def test_legal_and_local_slash_commands_make_zero_provider_calls(tmp_path: Path) -> None:
    agent = FakeAgent()

    async def run() -> None:
        app = _app(tmp_path, agent=agent)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            app.query_one("#composer").load_text("/legal")
            await pilot.press("enter")
            await pilot.pause(0.2)
            text = app.entries[-1].body
            assert "Repository: https://github.com/quantroninblue/Hephaestus" in text
            for name in (
                "README/legal overview",
                "SAFETY.md",
                "PRIVACY.md",
                "SECURITY.md",
                "RESPONSIBLE_USE.md",
                "THIRD_PARTY_NOTICES.md",
                "LICENSE",
                "NOTICE",
            ):
                assert name in text
            app.query_one("#composer").load_text("/status")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert agent.calls == []

    _run(run())


def test_no_key_local_workbench_state(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path, agent=None)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            assert "LOCAL WORKBENCH" in app.query_one("#header").content
            bodies = "\n".join(entry.body for entry in app.entries)
            assert "AI is not configured. CAD, Gazebo and ROS commands are available." in bodies
            assert all(entry.label != "! NEEDS ATTENTION" for entry in app.entries)

    _run(run())


def test_provider_rate_limit_card_keeps_local_tools_usable(tmp_path: Path) -> None:
    rate_limit = ProviderRateLimitError(
        "AI provider rate limit reached",
        ProviderErrorDetails(retry_after_seconds=42.0),
    )
    agent = FakeAgent(rate_limit)

    async def run() -> None:
        app = _app(tmp_path, agent=agent)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            app.query_one("#composer").load_text("reason")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert "Reset expected in approximately 42 seconds" in app.entries[-1].body
            app.query_one("#composer").load_text("/status")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.local.dispatches[-1] == "/status"

    _run(run())


def test_approval_y_n_and_escape_behavior(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            for key, expected in (("y", True), ("n", False), ("escape", False)):
                pending = _PendingApproval(
                    {"tool": "cadquery_build", "result": {}}, threading.Event()
                )
                app._show_approval(pending)
                await pilot.press(key)
                await pilot.pause(0.05)
                assert pending.event.is_set()
                assert pending.approved is expected

    _run(run())


def test_declined_workspace_mutation_preserves_bytes(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("original", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall(
            "write",
            "workspace_replace_text",
            {"relative_path": "note.txt", "old_text": "original", "new_text": "changed"},
        ),
        allow_mutation=True,
        confirm=lambda preview: False,
    )

    assert result["status"] == "ok"
    assert result["result"]["committed"] is False
    assert target.read_text(encoding="utf-8") == "original"


def test_committed_result_followup_failure_card(tmp_path: Path) -> None:
    result = AgentTurnResult(
        "Operation committed; AI explanation unavailable.\n{}",
        "fake",
        "fake-model",
        FinishReason.ERROR,
        {},
        tool_activity=({"name": "cadquery_build", "status": "ok"},),
    )

    async def run() -> None:
        app = _app(tmp_path, agent=FakeAgent(result))
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            app.query_one("#composer").load_text("build cad")
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.entries[-1].label == "✓ CAD BUILD COMMITTED"
            assert "AI explanation unavailable." in app.entries[-1].body

    _run(run())


def test_activity_timeline_expandable_logs() -> None:
    body = "\n".join(f"line {i}" for i in range(30))
    entry = ActivityEntry("logs", body, expandable=True)
    assert "more lines collapsed" in entry.rendered()
    entry.expanded = True
    assert "line 29" in entry.rendered()


def test_ctrl_c_active_motion_requests_zero_velocity(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            app.busy = True
            app.active_input = "/cmdvel 0.1 0 2"
            await pilot.press("ctrl+c")
            await pilot.pause(0.2)
            assert "/stop" in app.local.dispatches

    _run(run())


def test_ctrl_d_empty_composer_cleans_up(tmp_path: Path) -> None:
    async def run() -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause(0.1)
            assert app.local.closed is True

    _run(run())


def test_responsive_width_tiers_and_no_color_readability(tmp_path: Path, monkeypatch) -> None:
    assert layout_mode_for_width(120) == "full"
    assert layout_mode_for_width(80) == "compact"
    assert layout_mode_for_width(60) == "minimal"
    monkeypatch.setenv("NO_COLOR", "1")

    async def run() -> None:
        app = _app(tmp_path, no_color=True)
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            assert app.no_color is True
            assert "LOCAL WORKBENCH" in app.query_one("#header").content
            assert "◆" in NARROW_LOGO

    _run(run())


def test_command_palette_groups_only_implemented_commands() -> None:
    groups = {item.group for item in LOCAL_COMMANDS}
    assert {"Session", "CAD", "Gazebo", "ROS 2", "Motion", "Legal and help"} <= groups
    commands = {item.command for item in LOCAL_COMMANDS}
    assert "/stop" in commands
    assert not any("install" in command for command in commands)
    assert stage_for_input("/cad build robot.py") == "◈ FORGING CAD"
    assert stage_for_input("/gazebo validate world.sdf") == "◇ VALIDATING GEOMETRY"
    assert stage_for_input("/move forward 0.1 for 1") == "➤ COMMANDING MOTION"


def test_existing_explicit_json_command_remains_ansi_free() -> None:
    result = runner.invoke(cli.app, ["ai", "status", "--json"], env={"GROQ_API_KEY": ""})
    assert result.exit_code == 0
    json.loads(result.output)
    assert "\x1b[" not in result.output
