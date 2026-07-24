# Hephaestus AI

Hephaestus AI is a robotics engineering CLI. The implemented system provides environment diagnostics, a canonical engineering project model, typed ChangeSets, project-aware AI assistance through Groq, and one integrated Genesis vertical slice for a differential-drive warehouse inspection robot.

> [!WARNING]
> Hephaestus AI is experimental, simulation-first robotics engineering software.
> Generated designs, physics, code, commands, and explanations may be incorrect
> or unsafe. This release permits movement commands only for a session-owned,
> isolated Gazebo simulation and is not approved for physical-robot or
> safety-critical control. Review [SAFETY.md](SAFETY.md),
> [PRIVACY.md](PRIVACY.md), and [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) before
> use.


## Supported Platform

- Ubuntu 24.04
- Python 3.12
- ROS 2 Jazzy for ROS workspace build and simulation commands
- Gazebo Harmonic through the modern gz command
- ros_gz_sim, ros_gz_bridge, and gz_ros2_control for simulation integration
- CadQuery 2.8.x for real CAD export when installing the cad extra

Hephaestus Core remains usable when optional AI, ROS, Gazebo, integration, or CAD capabilities are missing. Capability-dependent commands fail closed with actionable diagnostics.

## Installation

Recommended Ubuntu prerequisites:

```bash
sudo apt install python3.12 python3.12-venv pipx
python3.12 -m pipx ensurepath
```

Install Hephaestus with CadQuery support:

```bash
pipx install "hephaestus-ai[cad]"
```

Install the base package without CadQuery:

```bash
pipx install hephaestus-ai
```

ROS 2 Jazzy, Gazebo Harmonic, ros_gz_sim, ros_gz_bridge, and gz_ros2_control are host dependencies for build and simulation workflows. They are detected by Hephaestus and are not installed by pipx. Hephaestus Core remains usable when optional AI, ROS, Gazebo, integration, or CAD capabilities are missing. Capability-dependent commands fail closed with actionable diagnostics.

Enter any project directory and start Hephaestus:

```bash
cd /path/to/your/project
hephaestus
```

Running `hephaestus` with no subcommand starts the Forge TUI when the terminal supports it, or the safe plain fallback otherwise.

## AI Setup

Hephaestus uses Groq for AI requests. The API key is read only from the runtime environment and is never stored by the CLI. Do not put credentials in project files, generated artifacts, or local logs.

```bash
export GROQ_API_KEY="..."
```

Store non-secret AI settings without storing or printing the key:

```bash
hephaestus ai configure --model openai/gpt-oss-120b
hephaestus ai configure --default-agent-max-tool-rounds 16
hephaestus ai status --json
```

## Development From Source

Use source installation only for developing Hephaestus itself:

```bash
git clone https://github.com/quantroninblue/Hephaestus.git
cd Hephaestus
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,cad]"
```

## Engineering Project File

Hephaestus stores canonical engineering state in hephaestus.project.json. Generated CAD, ROS, and Gazebo files are disposable artifacts under generated/ and are tied back to the committed project revision and fingerprint. The engineering project remains the only source of truth.

Physical values use explicit value, unit, and dimension fields. Internal calculations use SI units with right-handed coordinates: X forward, Y left, Z up. Quaternions are serialized as w, x, y, z.

## Core Commands

```bash
hephaestus --help
hephaestus --version
hephaestus doctor
hephaestus doctor --json
hephaestus doctor --require gazebo --require cad
hephaestus capabilities
hephaestus capabilities --json
hephaestus setup plan --require ros2 --require gazebo --require cad
hephaestus setup plan --json
```

Setup plans are declarative. They do not install packages, run sudo, edit repositories, modify shell startup files, open browsers, create credentials, or execute commands.

## Project Commands

```bash
hephaestus project init robot_project --name "Test Robot"
hephaestus project inspect robot_project
hephaestus project inspect robot_project --json
hephaestus project validate robot_project --strict
hephaestus project validate robot_project --json
hephaestus project apply robot_project changeset.json --dry-run
```

## Genesis Differential-Drive Slice

Create a project, ask Hephaestus to design the supported robot, generate artifacts, build the ROS workspace, and run a bounded simulation launch check:

```bash
hephaestus project init genesis_project --name "Genesis Robot"
hephaestus design "Build a differential-drive warehouse inspection robot for a small sensor payload" --project genesis_project
hephaestus generate genesis_project --target all
hephaestus artifacts inspect genesis_project
hephaestus build genesis_project
hephaestus simulate genesis_project --headless --smoke-test
```

For noninteractive use, add --yes to the design command after reviewing the requested operation. Use --dry-run to preview without committing or generating files. Use --no-generate to commit only the canonical design.

Generation writes a staged artifact tree under generated/ and publishes generated/hephaestus.artifacts.json only after successful validation. The manifest records project revision, project fingerprint, generator versions, backend versions, artifact roles, owner IDs, units, coordinate convention, dependencies, SHA-256 hashes, and fresh/stale status.

Generated artifact layout includes:

```text
generated/
  hephaestus.artifacts.json
  reports/
  cad/source/
  cad/step/
  cad/stl/
  gazebo/models/
  gazebo/worlds/
  ros2_ws/src/
```

The CAD backend generates deterministic CadQuery source and parameter data, STEP files, STL visual meshes, STL collision meshes, and a CAD report. CadQuery geometry uses millimetres internally with explicit metre-to-millimetre conversion; URDF, SDF, ROS, and Gazebo use SI units. Gazebo visual meshes use generated CAD-derived STL files with explicit 0.001 scale.

The ROS/Gazebo backend generates robot description, bringup, and simulation packages, Xacro/URDF, SDF model and local warehouse world, controller YAML, bridge YAML, and launch files. Simulation execution uses fixed application-owned argv and project-scoped generated paths. It is a constrained runner for this workflow, not a complete sandbox.

## AI Commands

```bash
hephaestus
hephaestus --plain
hephaestus ask "Explain the current project constraints" --project robot_project
hephaestus ask "Derive the torque relation for a lever arm"
hephaestus chat --project robot_project
```

Running `hephaestus` with no subcommand starts the full-screen Hephaestus Forge terminal interface when stdin/stdout are interactive, `TERM` is usable, and Textual can initialize safely. `hephaestus --plain` starts the original plain interactive shell. Non-TTY sessions, `TERM=dumb`, missing Textual, or safe terminal-initialization failures fall back to the plain shell.

Forge and the plain shell are rooted at the current working directory. They start even when `GROQ_API_KEY` is not configured. In that local-only state, slash commands remain available and natural-language AI requests report that AI configuration is unavailable. Commands beginning with `/` are parsed locally before any provider request and do not consume AI tokens.

The Forge startup screen always displays the experimental robotics notice and README URL. `--no-color`, the `NO_COLOR` environment variable, and `--reduced-motion` are supported for the default interactive interface.

Local shell command syntax:

```text
/help
/clear
/status
/legal
/exit
/quit
/cad build SOURCE
/gazebo validate WORLD
/gazebo launch WORLD
/gazebo launch WORLD --headless
/gazebo status
/gazebo logs [MAX_LINES]
/gazebo stop
/ros status
/ros topics
/ros controllers
/ros build ROS_WORKSPACE
/ros launch ROS_WORKSPACE PACKAGE LAUNCH_FILE
/ros logs [MAX_LINES]
/ros stop
/cmdvel LINEAR_X_MPS ANGULAR_Z_RADPS DURATION_SECONDS
/move forward SPEED_MPS for DURATION_SECONDS
/move backward SPEED_MPS for DURATION_SECONDS
/move turn-left ANGULAR_SPEED_RADPS for DURATION_SECONDS
/move turn-right ANGULAR_SPEED_RADPS for DURATION_SECONDS
/stop
```

`/legal` is a local provider-free command. It lists the repository URL and workspace-relative legal and safety documents: README/legal overview, SAFETY.md, PRIVACY.md, SECURITY.md, RESPONSIBLE_USE.md, THIRD_PARTY_NOTICES.md, LICENSE, and NOTICE.

When AI is configured, natural-language turns use Groq for general robotics and engineering reasoning and can list workspace files, read bounded UTF-8 text files, create or replace one bounded UTF-8 text file after confirmation, and apply one exact targeted replacement to an existing bounded UTF-8 text file after confirmation.

When the local CadQuery capability is active, the default session also exposes `cadquery_build`. The model can write an editable CadQuery `.py` source file in the workspace and then request a confirmed build of that file. The source must define `parts`, a mapping of 1 to 64 safe unique part names to non-empty CadQuery solid geometry. Dimensions in that source are millimetres. Hephaestus validates the source policy before execution, runs it in a child process with fixed argv, filtered environment, bounded output, timeout, resource limits where supported, closed stdin, and process-group cleanup, and exports only application-owned paths. User confirmation is mandatory because the restricted runner reduces accidental and obvious unsafe behavior but is not a complete OS security sandbox.

The default-agent CadQuery output tree is derived from the source stem:

```text
generated/cad/<source-stem>/
  manifest.json
  step/
    <part>.step
  stl/
    <part>.stl
  assembly.step
```

`assembly.step` is written only when the source defines a valid CadQuery `assembly`. The tool reports actual part paths, bounds in metres, volume in cubic metres, center of mass in metres when available, SHA-256 hashes, and the Gazebo STL mesh scale `0.001 0.001 0.001`. Builds stage into a temporary directory and publish the final directory only after export and validation complete; previous outputs are preserved on failure.

When the local Gazebo Harmonic capability is active, the default session also exposes a bounded Gazebo workbench. Local `/gazebo validate WORLD` validates a workspace `.sdf` or `.world` file with fixed `gz sdf -k` argv. Local `/gazebo launch WORLD` requests a confirmed GUI launch with fixed `gz sim -r WORLD_PATH` argv and `GZ_IP=127.0.0.1`; `/gazebo launch WORLD --headless` uses fixed headless server argv `gz sim -r -s WORLD_PATH`. GUI launch inherits only the required desktop-session variables from the parent Hephaestus process when valid and present: `DISPLAY`, `XAUTHORITY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS`, and `XDG_SESSION_TYPE`. GUI launch fails before spawning if no usable graphical display is available. Launches are non-blocking and Hephaestus allows only one owned Gazebo process per default-agent session. `/gazebo status`, `/gazebo logs`, and `/gazebo stop` operate only on that owned process. `/exit`, EOF, Ctrl-C, and normal shell teardown stop only the owned process group.

The local ROS workbench uses a session-isolated environment with bounded `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=1`, `GZ_IP=127.0.0.1`, and a unique Gazebo partition. `/ros build ROS_WORKSPACE` accepts only a safe relative workspace with `src/` and runs fixed colcon build argv after confirmation. `/ros launch ROS_WORKSPACE PACKAGE LAUNCH_FILE` accepts only a workspace built in the current session, safe package and launch-file identifiers, and starts one owned ROS launch process after confirmation. `/cmdvel`, `/move`, and `/stop` are simulation-only; they are allowed only when the session owns an active isolated ROS simulation, ROS 2 Jazzy is available, an active differential-drive controller is discovered, and the command topic is in the isolated graph. Velocity commands are bounded and publish zero velocity after completion, failure, or interruption.

Default-agent SDF mesh visuals should reference workspace-local meshes, such as `generated/cad/robot/stl/chassis.stl`, with `<scale>0.001 0.001 0.001</scale>` for CadQuery STL files. Gazebo validation and launch add only the bound workspace as the resource search root for the default-agent workbench. Missing meshes, symlink escapes, absolute outside-workspace meshes, and nonlocal mesh URIs prevent validation or launch with bounded evidence.

ask is read-only. If the model requests a mutation, Hephaestus denies it. Chat, design, and default-agent workspace writes/replacements require an application-controlled preview and explicit confirmation; only y or yes approves a change. Declined or failed mutations leave the project or workspace file unchanged.

## Current Limitations

- The deterministic `hephaestus design` workflow currently provides end-to-end generation, simulation, and validated ROS 2 control for differential-drive warehouse inspection robots.
- The default agent can create and visualize broader robot concepts using editable CadQuery geometry and Gazebo SDF, but reliable controllers, actuation, and motion validation are not guaranteed for non-differential-drive designs.
- Simulation smoke testing requires the relevant ROS 2 controller packages to be installed on the host. If they are missing, Hephaestus reports the missing packages and does not launch.
- Hephaestus does not install packages, create credentials, download Gazebo Fuel assets, or execute setup plans.
- The default agent does not expose arbitrary shell access, package installation, arbitrary Python execution, real-hardware control, arbitrary provider-selected ROS topics, arbitrary launch arguments, or external ROS graph control.

## Tests and Linting

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/python -m compileall src tests
.venv/bin/python -m pip check
```
