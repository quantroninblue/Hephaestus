# Repository Instructions

## Engineering Standards

- Use Python 3.12.
- Use the `src/` layout for importable package code.
- Keep type hints on new and changed Python code.
- Prefer simple, concrete modules over speculative abstractions.
- Do not add empty packages or placeholder implementations.
- Keep domain logic, external integrations, CLI presentation, and generated artifacts separated.
- Add or update tests for behavioral changes.
- Do not expose secrets, tokens, credentials, or complete environment-variable values in code, tests, docs, logs, diagnostics, tool results, AI messages, or local ledgers.
- Verify work before reporting completion.

## Engineering Project Model

- The engineering project is the canonical source of truth for engineering state.
- Generated backend files must never become authoritative over the engineering project model.
- Internal model concepts must remain backend-agnostic; backend-specific concepts belong in adapters or generated outputs.
- Preserve `EngineeringProject` as the single canonical aggregate root.
- Physical values require the application-owned `Quantity` abstraction with explicit value, unit, and dimension.
- Pint must remain behind the `Quantity` abstraction; Pint types must not leak through the domain model or serialized project format.
- Project model changes require schema/version consideration and deterministic serialization checks.
- Extension payloads must be namespaced and must not masquerade as validated core engineering data.
- Validation must be deterministic and side-effect free.
- Canonical serialization defines project fingerprints.
- Model changes must include tests for serialization, loading, validation, and relevant CLI behavior.

## AI And Providers

- Provider code is isolated under provider-specific adapters.
- No provider-specific SDK types may escape into orchestration, tools, or the domain model.
- Groq credentials are read only at runtime from the environment and must never be stored or printed.
- AI configuration stores only non-secret settings.
- Tests must not use hosted inference unless an explicit opt-in live marker is added and skipped by default.
- Model output is untrusted, including tool-call arguments.
- Model-facing operations must use simple domain contracts; do not expose raw internal ChangeSet JSON directly to providers.
- Provider-facing tool schemas must stay within the conservative compatibility profile: shallow objects, primitive fields, enums, required fields, descriptions, and `additionalProperties: false`; avoid `$ref`, `$defs`, `oneOf`, `anyOf`, `allOf`, discriminators, recursive schemas, and arbitrary dictionaries.
- Application code supplies revisions, stable IDs, provenance, fingerprints, and transaction metadata for AI-requested mutations.
- The system identity prompt is behavioral guidance, not a security boundary.
- Do not expose hidden system context or model reasoning traces in assistant output, JSON output, logs, or tool results.
- Human model output in the CLI must use the shared terminal Markdown renderer.
- JSON output must remain raw structured JSON and must never contain ANSI formatting.
- Every interactive turn must produce a visible outcome: assistant text, confirmation, denial, or a concise error.
- Nullable provider content is valid only when tool calls are present. Empty content without tool calls is an application error.
- Default terminal responses should be concise, direct, engineering-focused, and proportional to the user request.
- Tool-result claims must remain grounded in actual tool results; do not invent validators, project state, or completed actions.
- Application state outranks model narration; mutation outcomes are application-owned and terminal for the current user turn. After a mutation is applied, declined, or fails, do not ask the provider for another completion in that turn.
- Only explicit `y` or `yes`, case-insensitively, approves a mutation. `n`, `no`, blank input, EOF, Ctrl-C, and any unsupported response decline by default.
- Provider diagnostic data must be redacted: preserve safe status/code/type/request-id metadata, but never print raw failed generations, request bodies, credentials, headers, or hidden instructions.
- Nonexistent interfaces, GUIs, generated artifacts, or integrations must never be mentioned as available.

## Default Agent Shell And Workspace Tools

- Running `hephaestus` with no explicit subcommand starts the Hephaestus Forge Textual TUI when the terminal is interactive and capable; `hephaestus --plain`, non-TTY sessions, `TERM=dumb`, missing Textual, or safe terminal-initialization failures use the plain default agent shell rooted at the startup working directory.
- The Forge TUI is a presentation layer over the existing LocalWorkbenchSession, default agent session, approval callbacks, and owned Gazebo/ROS process lifecycle. It must not duplicate CAD, Gazebo, ROS, movement, provider, permission, validation, or process-management logic in widgets.
- The Forge startup notice must remain visible at every startup and point to the README URL. `/legal` is local/provider-free and lists the repository plus README/legal overview, SAFETY.md, PRIVACY.md, SECURITY.md, RESPONSIBLE_USE.md, THIRD_PARTY_NOTICES.md, LICENSE, and NOTICE using workspace-relative paths where applicable.
- The default agent identity is separate from structured project/design identities and must not use strict `ConceptRobotSpec` planning.
- Inject dynamic context separately from the versioned identity prompt: startup working directory, detected capabilities, available tools, and current session state.
- Do not print or otherwise expose the raw system identity or hidden dynamic context in ordinary CLI output.
- Default-agent provider calls may use conversational tool definitions but must not hardcode a provider output-token budget by default.
- Workspace tools must remain limited to `workspace_list_files`, `workspace_read_text`, `workspace_write_text`, `workspace_replace_text`, the reviewed default-agent `cadquery_build` tool when the CadQuery capability is active, and the reviewed default-agent Gazebo workbench tools until a later reviewed milestone expands them.
- Workspace tools must reject absolute paths, parent traversal, symlink escapes, device files, protected paths, credential-like paths, `.env` files, `.git`, `.venv`, `project_state.txt`, and installed Hephaestus source.
- Workspace reads, writes, and targeted replacements must be size bounded and UTF-8 only.
- Workspace writes and targeted replacements require application-controlled preview and explicit yes-only confirmation, write atomically, and report only the actual committed or declined result.
- `workspace_replace_text` is for small exact repairs of existing bounded UTF-8 text files. It requires `old_text` to occur exactly once, permits empty `new_text` for deletion, binds approval to the pre-preview file fingerprint, rechecks immediately before persistence, and must not print complete old/new content in the confirmation UI.
- A declined workspace write or targeted replacement is terminal for that tool result; the model must not get a chance to claim that the file was written in the same turn.
- Default-agent `cadquery_build` accepts only one existing bounded UTF-8 workspace-relative `.py` file. It must require application-controlled yes-only confirmation before executing generated CAD source, bind approval to the source fingerprint, recheck the fingerprint after approval, and return terminal declined or failed results without allowing provider narration to claim artifacts were created.
- Default-agent CadQuery source uses millimetres and must define `parts`, a mapping of 1 to 64 safe unique part names to non-empty CadQuery solid geometry. Optional `assembly` may be supplied only for combined STEP export. The source validator must reject forbidden imports, filesystem/process/network/environment access, dynamic-code builtins, dunder access, direct CadQuery importer/exporter calls, and output-path selection before execution.
- Default-agent CadQuery execution must use fixed application-owned argv, no `shell=True`, a temporary working directory, filtered environment with no credentials or unrelated values, closed stdin, bounded timeout, bounded stdout/stderr, resource limits where supported, and exact child process-group cleanup on timeout. The restricted runner reduces accidental and obvious unsafe behavior but is not a complete OS security sandbox.
- Default-agent CadQuery outputs are application-owned derived artifacts under `generated/cad/<source-stem>/` with `manifest.json`, per-part `step/<part>.step`, per-part `stl/<part>.stl`, and optional `assembly.step`. Builds must stage in a temporary directory and promote only after complete validation; previous outputs must be preserved on execution, export, validation, timeout, or confirmation failure. Tool results must report bounded structured paths, SI bounds/volume/center metrics, hashes, and the recommended Gazebo mesh scale, never binary contents.
- Default-agent CadQuery STL files are exported in millimetres. SDF visual meshes that reference those STL files must use scale `0.001 0.001 0.001`; SDF collision geometry should remain simpler and physically plausible.
- The default agent shell must not expose shell execution, arbitrary file access outside the startup directory, arbitrary Python execution, package installation, network tools, ROS 2, unreviewed CAD/build tools, movement, controllers, or arbitrary simulation tools.
- The default-agent Gazebo workbench may expose only `gazebo_validate_sdf`, `gazebo_launch`, `gazebo_status`, `gazebo_read_logs`, and `gazebo_stop`, and only when the existing Gazebo capability is active.
- Default-agent Gazebo validation must use fixed application-owned `gz sdf -k WORLD_PATH` argv, bounded timeout, bounded sanitized output, no `shell=True`, paths resolved through the workspace safety boundary, and only the bound workspace added to the Gazebo resource search path for local mesh resolution.
- Default-agent Gazebo launch must use fixed application-owned `gz sim -r WORLD_PATH` argv with `GZ_IP=127.0.0.1`, only the bound workspace in the Gazebo resource search path, no provider-supplied executable, flags, argv, environment, host IP, package installation, ROS interaction, movement command, or controller command.
- Default-agent Gazebo validation and launch must reject missing local mesh files, stale mesh references, symlink escapes, and outside-workspace mesh paths before launch with useful bounded evidence.
- Default-agent Gazebo GUI launch may inherit only valid required desktop-session variables from the parent Hephaestus process: `DISPLAY`, `XAUTHORITY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS`, and `XDG_SESSION_TYPE`. It must not copy the full host environment, must not expose inherited values in provider results or ordinary logs, and must fail before spawning with a graphical-session-unavailable message when neither a usable `DISPLAY` nor `WAYLAND_DISPLAY` is present.
- Default-agent Gazebo launch is consequential and requires application-controlled yes-only confirmation after preview and before start. The exact file must be rechecked after preview; validation failure, launch decline, or file changes after preview must start no process and remain terminal for the user turn.
- A default-agent session may own at most one Gazebo process group. Status, log reads, and stop operations must refer only to that owned process and must not treat unrelated host Gazebo processes as owned.
- Default-agent Gazebo logs must be captured by bounded session-owned mechanisms and sanitized before returning to the model or user.
- Default-agent Gazebo cleanup must stop only the owned process group on explicit `gazebo_stop`, `/exit`, EOF, Ctrl-C, normal shell teardown, or an actual process-manager safety failure. Provider authentication, rate-limit, recoverable completion failure, or tool-round exhaustion after a successful launch must preserve the active owned process so the user can continue the session.

## Engineering Tools And ChangeSets

- Engineering tools are provider-independent.
- Tools operate only against the application-selected active project; model-supplied arbitrary paths are not allowed.
- Do not expose tools for shell execution, arbitrary file access, Python execution, ROS2 command execution, network access, package installation, or credential access.
- Untrusted project and ChangeSet JSON must enter through strict validation boundaries.
- ChangeSet operation variants use stable serialized discriminators.
- Unknown or malformed operations must be rejected cleanly.
- Mutations require an expected project revision and must apply atomically.
- Mutation permission is application-controlled; the model must never grant itself permission.
- Write tools require dry-run preview, fingerprint/revision checks, and explicit confirmation before persistence.
- Mutation previews must fail closed if they do not include the before-fingerprint needed for the pre-write recheck.
- Declined mutation results sent back to the model must clearly state that no write occurred and must not present proposed after-state fingerprints or revisions as applied state.
- Failed or declined changes must leave the original project unchanged.
- Final semantic validation must run before persistent writes.

## Capabilities And Setup Planning

- Use the centralized capability registry for backend and environment detection; do not add duplicate ad hoc doctor checks.
- Stable public capability IDs currently include `core`, `ai`, `ros2`, `gazebo`, `ros_gz_sim`, `ros_gz_bridge`, `gz_ros2_control`, and `cad`.
- Hephaestus Core must remain usable when optional AI, ROS 2, Gazebo, ROS/Gazebo integration, or CAD capabilities are missing.
- Capability detection must use fixed argv subprocess probes only, with no `shell=True`, bounded timeouts, bounded output, sanitized evidence, non-interactive stdin, and no system mutation.
- Gazebo detection must parse semantic versions from fixed `gz sim --version` and `gz sim --versions` probes, model selected/default version separately from installed versions, and map majors 6 Fortress, 7 Garden, 8 Harmonic, 9 Ionic, and 10 Jetty.
- Setup plans are declarative and non-executable. They must not install packages, run sudo or polkit, modify repositories, edit shell startup files, open browsers, download installers, or create credentials.
- Future installation commands must come only from application-owned, versioned, reviewed recipes, never from raw model output, retrieved text, prompt interpolation, arbitrary shell strings, or unverified scripts.
- Future privileged installation requires a one-operation application-controlled grant bound to an exact reviewed plan, executable, argv, package set, repository set, network declaration, filesystem effects, short lifetime, and unique operation ID.
- Administrator authentication must be handled by the operating system; Hephaestus must never read, pipe, log, or store a password.
- Privileged installation is a separate trusted installer-broker boundary, not ordinary sandboxed tool execution and not an LLM sandbox bypass.
- Normal future execution must fail closed unless an application-owned execution broker can enforce deny-by-default policy, project-scoped filesystem access, network denial by default, resource limits, environment filtering, secret redaction, child containment, audit events, and application-owned approvals.
- Do not claim a working sandbox or installer exists until it actually exists and is verified.

## Commands

- Create the local environment: `python3.12 -m venv .venv`
- Upgrade packaging tools: `.venv/bin/python -m pip install --upgrade pip`
- Install for development: `.venv/bin/python -m pip install -e ".[dev]"`
- Install with CadQuery support: `.venv/bin/python -m pip install -e ".[dev,cad]"`
- Run tests: `.venv/bin/python -m pytest`
- Run linting: `.venv/bin/ruff check .`
- Run the CLI: `.venv/bin/hephaestus --help`
- Start the Forge TUI: `.venv/bin/hephaestus`
- Start the plain shell: `.venv/bin/hephaestus --plain`
- Run diagnostics: `.venv/bin/hephaestus doctor`
- Configure AI settings: `.venv/bin/hephaestus ai configure`
- Inspect AI status: `.venv/bin/hephaestus ai status --json`
- Ask a read-only AI question: `.venv/bin/hephaestus ask "Explain torque"`
- Start chat: `.venv/bin/hephaestus chat`
- Initialize a project: `.venv/bin/hephaestus project init robot_project --name "Test Robot"`
- Inspect a project: `.venv/bin/hephaestus project inspect robot_project`
- Validate a project: `.venv/bin/hephaestus project validate robot_project --strict`
- Dry-run a ChangeSet: `.venv/bin/hephaestus project apply robot_project changeset.json --dry-run`
- Design the supported mobile robot slice: `.venv/bin/hephaestus design "Build a differential-drive warehouse inspection robot" --project robot_project --dry-run`
- Generate derived artifacts: `.venv/bin/hephaestus generate robot_project --target all`
- Inspect generated artifacts: `.venv/bin/hephaestus artifacts inspect robot_project --json`
- Build the generated ROS workspace: `.venv/bin/hephaestus build robot_project`
- Run the bounded headless launch check: `.venv/bin/hephaestus simulate robot_project --headless --smoke-test`

## Genesis Vertical Slice Rules

- The implemented design operation supports only differential-drive warehouse inspection robots.
- Unsupported robot families must be rejected clearly rather than approximated.
- The model may provide only the shallow typed design intent; it must not generate CAD, ROS, Gazebo, shell commands, package files, or canonical IDs.
- Generated CAD, URDF/Xacro, SDF, controller, bridge, launch, and report files must derive from the committed `EngineeringProject` revision and artifact manifest.
- `generate`, `build`, and `simulate` commands must fail closed when artifacts are stale or required host capabilities are unavailable.
- The constrained execution helper is specific to this workflow and must not be documented or treated as a complete sandbox.

## Documentation Rules

- Public documentation must describe only functionality that currently works.
- Do not add roadmaps, teaser sections, or descriptions of unimplemented capabilities.
- Do not link to or mention `project_state.txt` from `README.md`.

## Git Rules

- Do not initialize Git unless the user explicitly requests it.
- Do not make commits unless the user explicitly requests them.
- Keep `project_state.txt` ignored and untracked.
- Never force-add `project_state.txt`.

## Local Project-State Ledger

`project_state.txt` is a private, local, persistent engineering ledger for Codex reference across implementation sessions.

Future Codex sessions must:

- Read `project_state.txt` before planning material work.
- Update `CURRENT PROJECT SNAPSHOT` in place before reporting completion of material work.
- Append a dated and timestamped entry to `CHRONOLOGICAL ARCHITECTURE LOG` before reporting completion of material work.
- Use ISO 8601 timestamps with timezone offset, using the machine local timezone when available.
- Keep the file ignored and untracked.
- Never store secrets, API keys, tokens, credentials, personal information, or complete environment-variable values in it.
- Record facts and decisions rather than speculative filler.
- Verify ledger claims against the actual code, tests, and configuration.
- Treat repository code, tests, and configuration as authoritative if they conflict with the ledger.
