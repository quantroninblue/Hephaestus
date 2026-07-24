"""Local command metadata and real activity labels for the Forge TUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandSpec:
    group: str
    command: str


LOCAL_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("Session", "/help"),
    CommandSpec("Session", "/clear"),
    CommandSpec("Session", "/status"),
    CommandSpec("Session", "/exit"),
    CommandSpec("Session", "/quit"),
    CommandSpec("CAD", "/cad build SOURCE"),
    CommandSpec("Gazebo", "/gazebo validate WORLD"),
    CommandSpec("Gazebo", "/gazebo launch WORLD"),
    CommandSpec("Gazebo", "/gazebo launch WORLD --headless"),
    CommandSpec("Gazebo", "/gazebo status"),
    CommandSpec("Gazebo", "/gazebo logs [MAX_LINES]"),
    CommandSpec("Gazebo", "/gazebo stop"),
    CommandSpec("ROS 2", "/ros status"),
    CommandSpec("ROS 2", "/ros topics"),
    CommandSpec("ROS 2", "/ros controllers"),
    CommandSpec("ROS 2", "/ros build ROS_WORKSPACE"),
    CommandSpec("ROS 2", "/ros launch ROS_WORKSPACE PACKAGE LAUNCH_FILE"),
    CommandSpec("ROS 2", "/ros logs [MAX_LINES]"),
    CommandSpec("ROS 2", "/ros stop"),
    CommandSpec("Motion", "/cmdvel LINEAR_X_MPS ANGULAR_Z_RADPS DURATION_SECONDS"),
    CommandSpec("Motion", "/move forward SPEED_MPS for DURATION_SECONDS"),
    CommandSpec("Motion", "/move backward SPEED_MPS for DURATION_SECONDS"),
    CommandSpec("Motion", "/move turn-left ANGULAR_SPEED_RADPS for DURATION_SECONDS"),
    CommandSpec("Motion", "/move turn-right ANGULAR_SPEED_RADPS for DURATION_SECONDS"),
    CommandSpec("Motion", "/stop"),
    CommandSpec("Legal and help", "/legal"),
    CommandSpec("Legal and help", "/help"),
)

SLASH_COMMANDS = tuple(spec.command for spec in LOCAL_COMMANDS)


def stage_for_input(text: str) -> str:
    command = text.strip()
    if not command:
        return "✓ VERIFIED"
    if not command.startswith("/"):
        return "✦ REASONING"
    if command.startswith("/cad build"):
        return "◈ FORGING CAD"
    if command.startswith("/gazebo validate"):
        return "◇ VALIDATING GEOMETRY"
    if command.startswith("/gazebo launch"):
        return "✦ IGNITING GAZEBO"
    if command.startswith("/ros build") or command.startswith("/ros launch"):
        return "⌁ BRIDGING ROS 2"
    if command.startswith("/cmdvel") or command.startswith("/move"):
        return "➤ COMMANDING MOTION"
    if command in {"/stop", "/gazebo stop", "/ros stop"}:
        return "■ STOPPING"
    return "✓ VERIFIED"


def suggestions_for(prefix: str) -> tuple[str, ...]:
    if not prefix.startswith("/"):
        return ()
    matches = [command for command in SLASH_COMMANDS if command.startswith(prefix)]
    return tuple(dict.fromkeys(matches[:8]))
