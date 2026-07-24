"""Textual Forge TUI application."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, RichLog, Static, TextArea

from hephaestus.ai.agent import AgentLoopError, AgentTurnResult, HephaestusAgentSession
from hephaestus.ai.config import resolve_config
from hephaestus.ai.provider_registry import default_provider_registry
from hephaestus.ai.types import ProviderError, ProviderRateLimitError
from hephaestus.local_workbench import LocalCommandResult, LocalWorkbenchSession
from hephaestus.tui.branding import NOTICE_TEXT, NOTICE_TITLE, NOTICE_URL, logo_for_width
from hephaestus.tui.commands import LOCAL_COMMANDS, stage_for_input, suggestions_for
from hephaestus.tui.display_paths import DisplayPathFormatter

PALETTE = {
    "background": "#0B0807",
    "surface": "#17110F",
    "ember": "#FF5A1F",
    "gold": "#FFB000",
    "hot": "#FFE2A8",
    "text": "#F4EEE8",
    "muted": "#9B8D84",
    "success": "#4FD18B",
    "warning": "#FFB454",
    "failure": "#FF4D4D",
}


@dataclass(slots=True)
class _PendingApproval:
    preview: dict[str, Any]
    event: threading.Event
    approved: bool | None = None


@dataclass(slots=True)
class ActivityEntry:
    label: str
    body: str
    level: str = "info"
    expandable: bool = False
    expanded: bool = False

    def rendered(self) -> str:
        if not self.expandable or self.expanded:
            return self.body
        lines = self.body.splitlines()
        if len(lines) <= 12:
            return self.body
        return "\n".join((*lines[:12], f"... {len(lines) - 12} more lines collapsed"))


class ApprovalCard(Static):
    """Focused preview card for application-owned approvals."""

    can_focus = True

    def set_preview(self, preview: Mapping[str, Any] | None) -> None:
        if preview is None:
            self.display = False
            self.update("")
            return
        self.display = True
        tool = str(preview.get("tool", "approval"))
        result = preview.get("result")
        rows = [f"■ APPROVAL REQUIRED: {tool}", "Y approve  N decline  Esc decline"]
        formatter = getattr(self.app, "display_paths", DisplayPathFormatter())
        if isinstance(result, Mapping):
            for key, value in result.items():
                if key in {"content", "old_text", "new_text"}:
                    rows.append(f"{key}: [bounded preview omitted]")
                elif isinstance(value, list):
                    display_items = formatter.format_value(value[:12])
                    rows.append(f"{key}: {' '.join(str(item) for item in display_items)}")
                else:
                    rows.append(f"{key}: {formatter.format_value(value)}")
        self.update("\n".join(rows))

    def on_key(self, event: events.Key) -> None:
        if event.key in {"y", "n", "escape"}:
            action = getattr(self.app, "_resolve_approval", None)
            if callable(action):
                action(event.key == "y")
            event.prevent_default()
            event.stop()


class ForgeComposer(TextArea):
    """Multiline composer with Hephaestus submit semantics."""

    BINDINGS = [
        ("enter", "submit", "Submit"),
        ("ctrl+j", "newline", "Newline"),
        ("tab", "complete", "Complete"),
    ]

    def action_submit(self) -> None:
        action = getattr(self.app, "action_submit", None)
        if callable(action):
            action()

    def action_newline(self) -> None:
        self.insert("\n")

    def action_complete(self) -> None:
        action = getattr(self.app, "_complete_suggestion", None)
        if callable(action):
            action()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.action_submit()
            event.prevent_default()
            event.stop()
        elif event.key == "ctrl+j":
            self.action_newline()
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            self.action_complete()
            event.prevent_default()
            event.stop()
        elif event.key == "ctrl+c":
            action = getattr(self.app, "action_interrupt", None)
            if callable(action):
                action()
            event.prevent_default()
            event.stop()
        elif event.key == "ctrl+d":
            action = getattr(self.app, "action_exit_empty", None)
            if callable(action):
                action()
            event.prevent_default()
            event.stop()
        elif event.key == "ctrl+k":
            action = getattr(self.app, "action_command_palette", None)
            if callable(action):
                action()
            event.prevent_default()
            event.stop()


class CommandPalette(ModalScreen[str | None]):
    """Searchable command palette for implemented local commands."""

    CSS = """
    CommandPalette {
        align: center middle;
    }
    #palette-box {
        width: 72;
        max-width: 95%;
        height: 80%;
        background: #17110F;
        color: #F4EEE8;
        border: tall #FF5A1F;
        padding: 1 2;
    }
    #palette-results {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-box"):
            yield Static("◆ Command Palette", id="palette-title")
            yield Input(placeholder="Search local commands", id="palette-search")
            yield Static(id="palette-results")

    def on_mount(self) -> None:
        self.query_one("#palette-search", Input).focus()
        self._refresh("")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh(event.value)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
        if event.key == "enter":
            value = self.query_one("#palette-search", Input).value.strip()
            matches = _filtered_commands(value)
            self.dismiss(matches[0].command if matches else None)
            event.stop()

    def _refresh(self, query: str) -> None:
        grouped: dict[str, list[str]] = {}
        for item in _filtered_commands(query):
            grouped.setdefault(item.group, []).append(item.command)
        lines: list[str] = []
        for group, commands in grouped.items():
            lines.append(f"{group}")
            lines.extend(f"  {command}" for command in commands)
        self.query_one("#palette-results", Static).update("\n".join(lines) or "No commands")


class ForgeApp(App[None]):
    """Full-screen Textual interface for the existing Hephaestus workbench."""

    TITLE = "Hephaestus"
    SUB_TITLE = ""
    BINDINGS = [
        ("ctrl+k", "command_palette", "Commands"),
        ("ctrl+c", "interrupt", "Cancel/Clear"),
        ("ctrl+d", "exit_empty", "Exit"),
        ("pageup", "timeline_page_up", "Timeline up"),
        ("pagedown", "timeline_page_down", "Timeline down"),
    ]
    CSS = """
    Screen {
        background: #0B0807;
        color: #F4EEE8;
    }
    Screen.no-color {
        background: black;
        color: white;
    }
    #root {
        height: 100%;
    }
    #startup {
        background: #0B0807;
        color: #F4EEE8;
        height: auto;
        padding: 1 2;
        border-bottom: solid #FF5A1F;
    }
    #logo {
        color: #FFE2A8;
        text-align: center;
    }
    #notice-title {
        color: #FFB454;
        text-style: bold;
    }
    #notice-text {
        color: #9B8D84;
    }
    #notice-url {
        color: #FF5A1F;
    }
    #header {
        height: auto;
        padding: 0 1;
        background: #17110F;
        border-bottom: solid #FF5A1F;
    }
    #timeline {
        height: 1fr;
        overflow-y: auto;
        background: #0B0807;
        padding: 0 1;
    }
    #approval-card {
        height: auto;
        max-height: 12;
        padding: 1 2;
        background: #17110F;
        color: #FFE2A8;
        border: heavy #FFB000;
        display: none;
    }
    #suggestions {
        height: auto;
        max-height: 5;
        color: #9B8D84;
        padding: 0 1;
    }
    #composer {
        height: 4;
        min-height: 3;
        border: tall #9B8D84;
        background: #17110F;
        color: #F4EEE8;
    }
    #composer:focus {
        border: tall #FF5A1F;
    }
    #status-line {
        height: 1;
        background: #17110F;
        color: #9B8D84;
    }
    .minimal #header {
        padding: 0;
    }
    .minimal #timeline {
        padding: 0;
    }
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        local_session: LocalWorkbenchSession,
        agent_session: HephaestusAgentSession | None,
        provider_label: str,
        model_label: str,
        ai_unavailable: str | None = None,
        no_color: bool = False,
        reduced_motion: bool = False,
        display_paths: DisplayPathFormatter | None = None,
    ) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.display_paths = display_paths or DisplayPathFormatter()
        self.local = local_session
        self.agent = agent_session
        self.provider_label = provider_label
        self.model_label = model_label
        self.ai_unavailable = ai_unavailable
        self.no_color = no_color or bool(os.environ.get("NO_COLOR"))
        self.reduced_motion = reduced_motion
        self.entries: list[ActivityEntry] = []
        self.command_history: list[str] = []
        self.history_index: int | None = None
        self.pending_approval: _PendingApproval | None = None
        self.busy = False
        self.active_input = ""
        self.cancellation_requested = False
        self.layout_mode = "full"
        self.startup_collapsed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="startup"):
                yield Static(id="logo")
                yield Static(NOTICE_TITLE, id="notice-title")
                yield Static(NOTICE_TEXT, id="notice-text")
                yield Static(id="notice-url")
            yield Static(id="header")
            yield RichLog(id="timeline", wrap=True, markup=False, highlight=False)
            yield ApprovalCard(id="approval-card")
            yield Static(id="suggestions")
            yield ForgeComposer(
                "",
                id="composer",
                placeholder="Ask for robot design, CAD, Gazebo, ROS 2, or local /commands",
                soft_wrap=True,
                show_line_numbers=False,
            )
            yield Static(id="status-line")
            yield Footer()

    def on_mount(self) -> None:
        self.title = "Hephaestus"
        self.sub_title = ""
        self._set_terminal_title()
        if self.no_color:
            self.screen.add_class("no-color")
        self._update_layout(self.size.width)
        self._render_startup()
        self.query_one("#notice-url", Static).update(
            Text(NOTICE_URL, style=f"{PALETTE['ember']} link {NOTICE_URL}")
        )
        self._render_header()
        self._append_initial_local_state()
        self.query_one("#composer", TextArea).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._update_layout(event.size.width)
        self._render_startup()
        self._render_header()

    def on_key(self, event: events.Key) -> None:
        if self.pending_approval is not None and event.key in {"y", "n", "escape"}:
            self._resolve_approval(event.key == "y")
            event.stop()
            return
        if event.key == "enter":
            self.action_submit()
            event.stop()
        elif event.key == "ctrl+j":
            self.query_one("#composer", TextArea).insert("\n")
            event.stop()
        elif event.key == "tab":
            self._complete_suggestion()
            event.stop()
        elif event.key == "escape":
            self._hide_suggestions()
            event.stop()
        elif event.key == "/":
            self.set_timer(0.01, self._refresh_suggestions)
        elif event.key in {"up", "down"}:
            self._navigate_history(event.key)
            event.stop()

    def action_submit(self) -> None:
        if self.busy:
            self._write_status("! NEEDS ATTENTION busy; current operation is still running")
            return
        composer = self.query_one("#composer", TextArea)
        text = composer.text.strip()
        if not text:
            return
        composer.load_text("")
        self._collapse_startup()
        self._hide_suggestions()
        self.command_history.append(text)
        self.history_index = None
        self.run_worker(self._handle_submission(text), name="submit", exclusive=True)

    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), self._palette_selected)

    def action_interrupt(self) -> None:
        if not self.busy:
            self.query_one("#composer", TextArea).load_text("")
            self._write_status("✓ VERIFIED composer cleared")
            return
        self.cancellation_requested = True
        self.add_activity("■ STOPPING", "Cancellation requested for the active owned operation.")
        if self.active_input.startswith(("/cmdvel", "/move", "/stop")):
            self.run_worker(self._send_zero_velocity(), name="zero-velocity", thread=True)

    def action_exit_empty(self) -> None:
        if self.query_one("#composer", TextArea).text.strip():
            return
        self.local.close()
        self.exit()

    def action_timeline_page_up(self) -> None:
        self.query_one("#timeline", RichLog).scroll_page_up(animate=not self.reduced_motion)

    def action_timeline_page_down(self) -> None:
        self.query_one("#timeline", RichLog).scroll_page_down(animate=not self.reduced_motion)

    async def _handle_submission(self, text: str) -> None:
        self.busy = True
        self.active_input = text
        started = time.monotonic()
        stage = stage_for_input(text)
        self.add_activity(stage, f"{text}\nelapsed: 0.0s")
        try:
            if text.startswith("/"):
                result = await asyncio.to_thread(
                    self.local.dispatch, text, confirm=self._confirm_from_worker
                )
                self._render_local_result(result, elapsed=time.monotonic() - started)
                if result.clear_requested and self.agent is not None:
                    self.agent.clear()
                    self.clear_timeline()
                if result.exit_requested:
                    self.exit()
            elif self.agent is None:
                self.add_activity(
                    "! NEEDS ATTENTION",
                    (
                        "LOCAL WORKBENCH\n"
                        "AI is not configured. CAD, Gazebo and ROS commands are available."
                    ),
                    level="warning",
                )
            else:
                result = await asyncio.to_thread(
                    self.agent.ask, text, confirm=self._confirm_from_worker
                )
                self._render_agent_result(result, elapsed=time.monotonic() - started)
        except ProviderRateLimitError as exc:
            self._render_rate_limit(exc)
        except (ProviderError, AgentLoopError, ValueError) as exc:
            self.add_activity("× FAILED", self.display_paths.format_text(str(exc)), level="error")
        finally:
            self.busy = False
            self.active_input = ""
            self._render_header()
            self._write_status(self._status_text())

    async def _send_zero_velocity(self) -> None:
        try:
            await asyncio.to_thread(self.local.dispatch, "/stop")
            self.add_activity("✓ VERIFIED", "Zero velocity command requested.")
        except Exception as exc:
            self.add_activity("! NEEDS ATTENTION", f"Zero velocity request failed: {exc}")

    def _render_local_result(self, result: LocalCommandResult, *, elapsed: float) -> None:
        label = "✓ VERIFIED" if result.status == "ok" else "× FAILED"
        body = f"{self.display_paths.format_text(result.message)}\nelapsed: {elapsed:.1f}s"
        if result.payload:
            body += "\n" + _bounded_json(self.display_paths.format_value(result.payload))
        self.add_activity(label, body, level="success" if result.status == "ok" else "error")

    def _render_agent_result(self, result: AgentTurnResult, *, elapsed: float) -> None:
        if result.content.startswith("Operation committed; AI explanation unavailable."):
            title = (
                "✓ CAD BUILD COMMITTED"
                if _has_cad_activity(result)
                else "✓ OPERATION COMMITTED"
            )
            body = "The artifacts were created successfully.\nAI explanation unavailable."
            self.add_activity(title, body, level="success")
            return
        if result.tool_activity:
            display_activity = self.display_paths.format_value(
                {"tool_activity": result.tool_activity}
            )
            self.add_activity("✓ VERIFIED", _bounded_json(display_activity))
        self.add_activity(
            "Assistant",
            f"elapsed: {elapsed:.1f}s",
            markdown=self.display_paths.format_text(result.content),
        )

    def _render_rate_limit(self, exc: ProviderRateLimitError) -> None:
        details = getattr(exc, "details", None)
        reset = getattr(details, "retry_after_seconds", None)
        lines = [
            "╭─ AI PAUSED ─────────────────────────────────────────────╮",
            "│ Groq rate limit reached. Local engineering tools remain │",
        ]
        if isinstance(reset, int | float):
            seconds = int(round(reset))
            lines.append(f"│ available. Reset expected in approximately {seconds} seconds.  │")
        else:
            lines.append("│ available.                                           │")
        lines.extend(
            [
                "│                                                        │",
                "│ /cad  /gazebo  /ros  /cmdvel  /move  /stop             │",
                "╰────────────────────────────────────────────────────────╯",
            ]
        )
        self.add_activity("! NEEDS ATTENTION", "\n".join(lines), level="warning")

    def add_activity(
        self, label: str, body: str, *, level: str = "info", markdown: str | None = None
    ) -> None:
        display_body = self.display_paths.format_text(body)
        expandable = len(display_body.splitlines()) > 14 or len(display_body) > 1800
        entry = ActivityEntry(label=label, body=display_body, level=level, expandable=expandable)
        self.entries.append(entry)
        log = self.query_one("#timeline", RichLog)
        heading = Text(label, style=_level_style(level, self.no_color))
        log.write(heading, scroll_end=True, animate=False)
        if markdown is not None:
            display_markdown = self.display_paths.format_text(markdown)
            log.write(RichMarkdown(display_markdown), scroll_end=True, animate=False)
        else:
            log.write(entry.rendered(), scroll_end=True, animate=False)

    def clear_timeline(self) -> None:
        self.entries.clear()
        self.query_one("#timeline", RichLog).clear()

    def toggle_entry(self, index: int) -> None:
        entry = self.entries[index]
        if not entry.expandable:
            return
        entry.expanded = not entry.expanded
        self.query_one("#timeline", RichLog).clear()
        for item in self.entries:
            self.query_one("#timeline", RichLog).write(Text(item.label), animate=False)
            self.query_one("#timeline", RichLog).write(item.rendered(), animate=False)

    def _confirm_from_worker(self, preview: dict[str, Any]) -> bool:
        pending = _PendingApproval(preview=preview, event=threading.Event())
        self.call_from_thread(self._show_approval, pending)
        pending.event.wait()
        return pending.approved is True

    def _show_approval(self, pending: _PendingApproval) -> None:
        self.pending_approval = pending
        card = self.query_one("#approval-card", ApprovalCard)
        card.set_preview(pending.preview)
        card.focus()
        self._write_status("■ APPROVAL REQUIRED Y approve / N decline / Esc decline")

    def _resolve_approval(self, approved: bool) -> None:
        pending = self.pending_approval
        if pending is None:
            return
        pending.approved = approved
        pending.event.set()
        self.pending_approval = None
        self.query_one("#approval-card", ApprovalCard).set_preview(None)
        self.query_one("#composer", TextArea).focus()
        self.add_activity(
            "✓ VERIFIED" if approved else "! NEEDS ATTENTION",
            "Approval accepted." if approved else "Approval declined safely.",
        )

    def _render_startup(self) -> None:
        self.query_one("#logo", Static).update(logo_for_width(self.size.width))

    def _set_terminal_title(self) -> None:
        setter = getattr(self.console, "set_window_title", None)
        if callable(setter):
            setter("Hephaestus")

    def _collapse_startup(self) -> None:
        if self.startup_collapsed:
            return
        self.startup_collapsed = True
        startup = self.query_one("#startup", Vertical)
        startup.styles.height = 0
        startup.styles.display = "none"

    def _append_initial_local_state(self) -> None:
        if self.agent is None:
            self.add_activity(
                "LOCAL WORKBENCH",
                "AI is not configured. CAD, Gazebo and ROS commands are available.",
                level="warning",
            )

    def _render_header(self) -> None:
        payload = self.local.quick_status()
        gazebo = payload.get("gazebo", "gazebo=unavailable")
        ros = payload.get("ros")
        ros_state = "ros=unavailable"
        if isinstance(ros, Mapping):
            ros_state = f"ros={ros.get('runtime_state', 'unknown')}"
        cad_state = _readiness_label("CAD", _capability_active(self.local, "cad"))
        gazebo_state = (
            _readiness_label("Gazebo", _capability_active(self.local, "gazebo"))
        )
        ros_ready = _readiness_label("ROS", _capability_active(self.local, "ros2"))
        provider = (
            f"✓ AI {self.provider_label}/{self.model_label}"
            if self.agent is not None
            else "◆ LOCAL WORKBENCH"
        )
        display_root = self.display_paths.format_path(self.workspace_root)
        text = (
            f"◆ HEPHAESTUS | {self.layout_mode.upper()} | {provider}\n"
            f"{_short_path(display_root)} | {cad_state} | {gazebo_state} | "
            f"{ros_ready} | {gazebo} | {ros_state}"
        )
        self.query_one("#header", Static).update(text)
        self._write_status(self._status_text())

    def _status_text(self) -> str:
        provider = (
            "LOCAL WORKBENCH"
            if self.agent is None
            else f"{self.provider_label}:{self.model_label}"
        )
        display_root = self.display_paths.format_path(self.workspace_root)
        return (
            f"{provider} | {_short_path(display_root)} | "
            "Enter submit Ctrl+J newline Ctrl+K commands Ctrl+C cancel Ctrl+D exit"
        )

    def _write_status(self, text: str) -> None:
        self.query_one("#status-line", Static).update(text)

    def _update_layout(self, width: int) -> None:
        self.layout_mode = layout_mode_for_width(width)
        root = self.query_one("#root", Vertical) if self.is_mounted else None
        if root is not None:
            root.remove_class("full")
            root.remove_class("compact")
            root.remove_class("minimal")
            root.add_class(self.layout_mode)

    def _refresh_suggestions(self) -> None:
        text = self.query_one("#composer", TextArea).text.strip()
        matches = suggestions_for(text)
        if matches:
            self.query_one("#suggestions", Static).update("\n".join(matches))

    def _hide_suggestions(self) -> None:
        self.query_one("#suggestions", Static).update("")

    def _complete_suggestion(self) -> None:
        composer = self.query_one("#composer", TextArea)
        matches = suggestions_for(composer.text.strip())
        if matches:
            composer.load_text(matches[0])
            self._refresh_suggestions()

    def _navigate_history(self, key: str) -> None:
        if not self.command_history:
            return
        if self.history_index is None:
            self.history_index = len(self.command_history)
        self.history_index += -1 if key == "up" else 1
        self.history_index = max(0, min(self.history_index, len(self.command_history)))
        value = (
            ""
            if self.history_index == len(self.command_history)
            else self.command_history[self.history_index]
        )
        self.query_one("#composer", TextArea).load_text(value)

    def _palette_selected(self, command: str | None) -> None:
        if command:
            self.query_one("#composer", TextArea).load_text(command)
            self.query_one("#composer", TextArea).focus()


def create_forge_app(
    workspace_root: Path,
    *,
    no_color: bool = False,
    reduced_motion: bool = False,
) -> ForgeApp:
    local = LocalWorkbenchSession(workspace_root)
    agent: HephaestusAgentSession | None = None
    provider_label = "unavailable"
    model_label = "unavailable"
    ai_unavailable: str | None = None
    try:
        config = resolve_config()
        provider = default_provider_registry().create(config)
        provider_label = provider.provider_id
        model_label = provider.model_id
        agent = HephaestusAgentSession(
            provider=provider,
            startup_directory=workspace_root,
            tool_executor=local.tool_executor,
            capabilities=local.capabilities,
            tool_calling_temperature=config.default_agent_tool_temperature,
            max_tool_rounds=config.default_agent_max_tool_rounds,
        )
    except (ProviderError, OSError) as exc:
        ai_unavailable = str(exc)
    return ForgeApp(
        workspace_root=workspace_root,
        local_session=local,
        agent_session=agent,
        provider_label=provider_label,
        model_label=model_label,
        ai_unavailable=ai_unavailable,
        no_color=no_color,
        reduced_motion=reduced_motion,
    )


def layout_mode_for_width(width: int) -> str:
    if width < 70:
        return "minimal"
    if width < 100:
        return "compact"
    return "full"


def _filtered_commands(query: str):
    lowered = query.lower().strip()
    for item in LOCAL_COMMANDS:
        if not lowered or lowered in item.command.lower() or lowered in item.group.lower():
            yield item


def _bounded_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if len(text) > 4000:
        return text[:4000] + "\n... bounded output truncated"
    return text


def _short_path(path: Path | str, *, max_chars: int = 54) -> str:
    value = str(path)
    if len(value) <= max_chars:
        return value
    return "..." + value[-(max_chars - 3) :]


def _level_style(level: str, no_color: bool) -> str:
    if no_color:
        return "bold"
    return {
        "success": PALETTE["success"],
        "warning": PALETTE["warning"],
        "error": PALETTE["failure"],
    }.get(level, PALETTE["hot"])


def _capability_active(local: LocalWorkbenchSession, capability_id: str) -> bool:
    if local.capabilities is None:
        return False
    item = local.capabilities.by_id().get(capability_id)
    return item is not None and item.status.value == "active"


def _readiness_label(name: str, active: bool) -> str:
    return f"✓ {name} ready" if active else f"! {name} unavailable"


def _has_cad_activity(result: AgentTurnResult) -> bool:
    return any(item.get("name") == "cadquery_build" for item in result.tool_activity)
