"""Display-only path formatting for the Forge TUI."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ABSOLUTE_PATH_TOKEN = re.compile(r"(?P<prefix>^|[\s=:\"'(\[])(?P<path>/[^\s,\"')\]]+)")


@dataclass(frozen=True, slots=True)
class DisplayPathFormatter:
    """Shorten human-facing paths without changing authoritative values."""

    home: Path = field(default_factory=Path.home)
    _home: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_home", self.home.expanduser().resolve(strict=False))

    def format_path(self, path: Path | str) -> str:
        path_obj = path if isinstance(path, Path) else Path(path)
        if not path_obj.is_absolute():
            return str(path)
        resolved = path_obj.expanduser().resolve(strict=False)
        try:
            relative = resolved.relative_to(self._home)
        except ValueError:
            return str(path)
        if not relative.parts:
            return "~"
        return "~/" + relative.as_posix()

    def format_text(self, text: str) -> str:
        path = Path(text)
        if path.is_absolute() and not any(character.isspace() for character in text):
            return self.format_path(path)

        def replace(match: re.Match[str]) -> str:
            return match.group("prefix") + self.format_path(match.group("path"))

        return _ABSOLUTE_PATH_TOKEN.sub(replace, text)

    def format_value(self, value: Any) -> Any:
        if isinstance(value, Path):
            return self.format_path(value)
        if isinstance(value, str):
            return self.format_text(value)
        if isinstance(value, Mapping):
            return {key: self.format_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.format_value(item) for item in value)
        if isinstance(value, list):
            return [self.format_value(item) for item in value]
        if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            return [self.format_value(item) for item in value]
        return value
