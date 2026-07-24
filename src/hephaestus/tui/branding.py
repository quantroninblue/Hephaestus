"""Branding and legal presentation helpers for the Forge TUI."""

from __future__ import annotations

import os
import sys

FULL_LOGO = (
    "                  ·      ✦      ·\n"
    "             ╲    │    ╱   ╲    │    ╱\n"
    "       █ █ █▀▀ █▀█ █ █ ▄▀█ █▀▀ █▀ ▀█▀ █ █ █▀\n"
    "       █▀█ ██▄ █▀▀ █▀█ █▀█ ██▄ ▄█  █  █▄█ ▄█\n"
    "                    AI ROBOTICS FORGE"
)

NARROW_LOGO = "◆ HEPHAESTUS — AI ROBOTICS FORGE"
ASCII_LOGO = "HEPHAESTUS - AI ROBOTICS FORGE"

NOTICE_TITLE = "Experimental robotics simulation software."
NOTICE_TEXT = (
    "Review the safety, privacy, licensing, and responsible-use notices before use:"
)
NOTICE_URL = "https://github.com/quantroninblue/Hephaestus#readme"


def terminal_supports_unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in encoding


def logo_for_width(width: int, *, ascii_safe: bool | None = None) -> str:
    if ascii_safe is None:
        ascii_safe = False
    if ascii_safe:
        return ASCII_LOGO
    if width < 70:
        return NARROW_LOGO
    return FULL_LOGO


def notice_lines() -> tuple[str, str, str]:
    return NOTICE_TITLE, NOTICE_TEXT, NOTICE_URL


def osc8_link(url: str = NOTICE_URL, label: str | None = None) -> str:
    visible = label or url
    if os.environ.get("NO_COLOR"):
        return visible
    return f"\033]8;;{url}\033\\{visible}\033]8;;\033\\"
