"""Command-line interface for Hephaestus AI."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from hephaestus import __version__
from hephaestus.ai.agent import AgentLoopError, HephaestusAgentSession
from hephaestus.ai.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    config_path,
    resolve_config,
    save_user_config,
)
from hephaestus.ai.orchestrator import (
    HephaestusOrchestrator,
    OrchestrationError,
    OrchestrationResult,
)
from hephaestus.ai.provider_registry import default_provider_registry
from hephaestus.ai.types import ProviderError
from hephaestus.backends.cadquery_backend import CadGenerationError
from hephaestus.backends.ros_gazebo import RosGazeboGenerationError
from hephaestus.capabilities import (
    CapabilityInspectionError,
    CapabilityReport,
    collect_capabilities,
    normalize_capability_id,
)
from hephaestus.changes import (
    ChangeError,
    ChangeSetLoadError,
    apply_changeset_file,
    fingerprint_project,
)
from hephaestus.concept_planning import ConceptPlanner, ConceptPlanningError
from hephaestus.concept_robot import (
    CompiledConceptRobot,
    ConceptError,
    compile_concept_robot_design,
    concept_robot_spec,
)
from hephaestus.doctor import DiagnosticReport, DiagnosticStatus, collect_diagnostics
from hephaestus.execution import ConstrainedRunner, ExecutionError, ExecutionResult
from hephaestus.generation import (
    GenerationError,
    GenerationTarget,
    generate_project_artifacts,
    inspect_artifact_state,
)
from hephaestus.local_workbench import (
    LocalCommandResult,
    LocalWorkbenchSession,
    standalone_cad_build,
    standalone_gazebo_validate,
)
from hephaestus.mobile_robot import (
    CompiledMobileRobot,
    MobileRobotIntent,
    compile_mobile_robot_design,
    mobile_robot_spec,
    validate_mobile_robot_project,
)
from hephaestus.model import EngineeringProject, UnsupportedSchemaVersionError
from hephaestus.persistence import (
    ProjectPersistenceError,
    load_project,
    resolve_project_file,
    save_project,
)
from hephaestus.setup_plan import SetupPlan, build_setup_plan
from hephaestus.tools.project import default_project_tool_registry
from hephaestus.tools.registry import (
    ProjectToolExecutor,
    ToolContext,
    ToolWorkflowPolicy,
    mobile_robot_design_tool_policy,
)
from hephaestus.validation import (
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    validate_project,
)

app = typer.Typer(
    add_completion=False,
    help="Hephaestus AI robotics engineering CLI.",
    no_args_is_help=False,
)
project_app = typer.Typer(help="Create, inspect, and validate engineering project files.")
design_app = typer.Typer(help="Compile typed design intents into engineering projects.")
ai_app = typer.Typer(help="Configure and inspect AI provider settings.")
setup_app = typer.Typer(help="Inspect read-only setup plans.")
artifacts_app = typer.Typer(help="Inspect generated artifact manifests.")
concept_app = typer.Typer(help="Create and edit bounded experimental concept robots.")
cad_app = typer.Typer(help="Build local provider-free CadQuery source.")
gazebo_app = typer.Typer(help="Validate local provider-free Gazebo worlds.")
app.add_typer(project_app, name="project")
project_app.add_typer(design_app, name="design")
app.add_typer(ai_app, name="ai")
app.add_typer(setup_app, name="setup")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(concept_app, name="concept")
app.add_typer(cad_app, name="cad")
app.add_typer(gazebo_app, name="gazebo")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hephaestus {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed Hephaestus version.",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Use the original plain interactive shell instead of the Forge TUI.",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable color in the default interactive interface.",
    ),
    reduced_motion: bool = typer.Option(
        False,
        "--reduced-motion",
        help="Disable startup animation and nonessential animated indicators.",
    ),
) -> None:
    """Hephaestus AI command line."""
    if ctx.invoked_subcommand is None:
        _run_default_interface(
            plain=plain,
            no_color=no_color,
            reduced_motion=reduced_motion,
        )


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit diagnostics as machine-readable JSON."
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero when implemented core runtime checks are not ready.",
    ),
    require: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require an optional capability ID to be active."),
    ] = None,
) -> None:
    """Inspect the local development environment."""

    required = tuple(normalize_capability_id(item) for item in (require or ()))
    try:
        report = collect_diagnostics(required_capabilities=required)
    except CapabilityInspectionError as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _render_doctor_table(report)
    if (strict and not report.is_strict_ready) or not report.is_required_ready:
        raise typer.Exit(1)


@app.command("capabilities")
def capabilities(
    json_output: bool = typer.Option(
        False, "--json", help="Emit capabilities as machine-readable JSON."
    ),
) -> None:
    """Inspect backend capability state without modifying the system."""

    report = collect_capabilities()
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _render_capability_report(report)


@cad_app.command("build")
def cad_build(
    source: Annotated[str, typer.Argument(help="Safe relative CadQuery .py source file.")],
    yes: bool = typer.Option(False, "--yes", help="Execute without interactive approval."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit build result as machine-readable JSON."
    ),
) -> None:
    """Build provider-free CadQuery source into local STEP/STL artifacts."""

    if json_output and not yes:
        result = standalone_cad_build(Path.cwd(), source, confirm=lambda preview: False)
        payload = _standalone_tool_payload(result)
        payload["status"] = "confirmation_required"
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        raise typer.Exit(1)
    confirm = (lambda preview: True) if yes else _confirm_cadquery_build
    result = standalone_cad_build(Path.cwd(), source, confirm=confirm)
    payload = _standalone_tool_payload(result)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_local_payload("CadQuery Build", payload)
    if payload["status"] != "succeeded":
        raise typer.Exit(1)


@gazebo_app.command("validate")
def gazebo_validate(
    world: Annotated[str, typer.Argument(help="Safe relative .sdf or .world file.")],
    json_output: bool = typer.Option(
        False, "--json", help="Emit validation result as machine-readable JSON."
    ),
) -> None:
    """Validate a provider-free local Gazebo SDF/world file."""

    result = standalone_gazebo_validate(Path.cwd(), world)
    payload = _standalone_tool_payload(result)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_local_payload("Gazebo Validation", payload)
    if payload["status"] == "failed":
        raise typer.Exit(1)


@setup_app.command("plan")
def setup_plan(
    require: Annotated[
        list[str] | None,
        typer.Option("--require", help="Limit the plan to a capability ID."),
    ] = None,
    json_output: bool = typer.Option(
        False, "--json", help="Emit setup plan as machine-readable JSON."
    ),
) -> None:
    """Create a read-only setup plan. No installation is performed."""

    required = tuple(normalize_capability_id(item) for item in (require or ()))
    try:
        plan = build_setup_plan(collect_capabilities(), required_capabilities=required)
    except ValueError as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        _render_setup_plan(plan)


@app.command("generate")
def generate(
    project_path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    target: Annotated[
        GenerationTarget, typer.Option("--target", help="Artifact target to generate.")
    ] = GenerationTarget.ALL,
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the generation plan only."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit generation result as machine-readable JSON."
    ),
) -> None:
    """Generate derived CAD, ROS 2, and Gazebo artifacts."""

    try:
        result = generate_project_artifacts(project_path, target=target, dry_run=dry_run)
    except (
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        CadGenerationError,
        RosGazeboGenerationError,
        GenerationError,
        ValueError,
        OSError,
    ) as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    payload = result.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_generation_plan(payload)
    if payload["status"] in {"failed", "invalid"}:
        raise typer.Exit(1)


@artifacts_app.command("inspect")
def artifacts_inspect(
    project_path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit artifact state as machine-readable JSON."
    ),
) -> None:
    """Inspect the current generated artifact manifest."""

    try:
        payload = inspect_artifact_state(project_path)
    except (
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        GenerationError,
        ValueError,
    ) as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_artifact_state(payload)


@app.command("build")
def build(
    project_path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the fixed build command only."),
    timeout_seconds: float = typer.Option(120.0, "--timeout", help="Build timeout in seconds."),
    json_output: bool = typer.Option(
        False, "--json", help="Emit execution result as machine-readable JSON."
    ),
) -> None:
    """Build the generated ROS 2 workspace with fixed colcon argv."""

    _load_project_or_exit(project_path)
    project_dir = resolve_project_file(project_path).parent
    if dry_run:
        payload = {
            "status": "planned",
            "command": "build_ros_gazebo",
            "argv": ["colcon", "build", "--merge-install"],
            "cwd": str(project_dir / "generated" / "ros2_ws"),
        }
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _render_execution_payload(payload)
        return
    try:
        result = ConstrainedRunner(
            project_dir=project_dir, timeout_seconds=timeout_seconds
        ).build_ros_gazebo()
    except (ExecutionError, GenerationError) as exc:
        _render_execution_error(exc, json_output)
        raise typer.Exit(1) from exc
    _finish_execution_result(result, json_output)


@app.command("simulate")
def simulate(
    project_path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    headless: bool = typer.Option(False, "--headless", help="Use headless Gazebo server mode."),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Run bounded readiness checks."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the fixed launch plan only."),
    yes: bool = typer.Option(False, "--yes", help="Authorize the predefined GUI launch."),
    timeout_seconds: float = typer.Option(
        120.0, "--timeout", help="Simulation timeout in seconds."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit execution result as machine-readable JSON."
    ),
) -> None:
    """Launch or smoke-test the generated Gazebo/ROS workflow."""

    _load_project_or_exit(project_path)
    project_dir = resolve_project_file(project_path).parent
    payload = {
        "command": "simulate_ros_gazebo",
        "project": str(project_dir),
        "headless": headless,
        "smoke_test": smoke_test,
        "cwd": str(project_dir / "generated" / "ros2_ws"),
    }
    if dry_run:
        payload["status"] = "planned"
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _render_execution_payload(payload)
        return
    if not headless and not yes:
        if json_output:
            typer.echo(
                json.dumps(
                    {**payload, "status": "confirmation_required"},
                    indent=2,
                    sort_keys=True,
                )
            )
            raise typer.Exit(1)
        _render_execution_payload({**payload, "status": "awaiting_confirmation"})
        if not _confirm_yes_only("Launch the generated Gazebo GUI workflow?"):
            typer.echo("Simulation launch not started.")
            return
    try:
        result = ConstrainedRunner(
            project_dir=project_dir, timeout_seconds=timeout_seconds
        ).simulate_ros_gazebo(headless=headless, smoke_test=smoke_test)
    except (ExecutionError, GenerationError) as exc:
        _render_execution_error(exc, json_output)
        raise typer.Exit(1) from exc
    _finish_execution_result(result, json_output)


@ai_app.command("configure")
def ai_configure(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", help="AI provider identifier."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Model identifier."),
    timeout_seconds: float = typer.Option(30.0, "--timeout", help="Request timeout in seconds."),
    max_completion_tokens: int | None = typer.Option(
        None,
        "--max-completion-tokens",
        help="Optional structured-output completion token budget.",
    ),
    default_agent_tool_temperature: float | None = typer.Option(
        None,
        "--default-agent-tool-temperature",
        help="Optional default-agent tool-calling temperature.",
    ),
    default_agent_max_tool_rounds: int | None = typer.Option(
        None,
        "--default-agent-max-tool-rounds",
        help="Optional default-agent sequential provider tool-round limit.",
    ),
) -> None:
    """Store non-secret AI configuration."""

    try:
        path = save_user_config(
            provider=provider,
            model=model,
            timeout_seconds=timeout_seconds,
            max_completion_tokens=max_completion_tokens,
            default_agent_tool_temperature=default_agent_tool_temperature,
            default_agent_max_tool_rounds=default_agent_max_tool_rounds,
            path=config_path(),
        )
    except ProviderError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"AI configuration written to {path}")
    typer.echo("Set GROQ_API_KEY in your environment before making AI requests.")


@ai_app.command("status")
def ai_status(
    json_output: bool = typer.Option(False, "--json", help="Emit status as machine-readable JSON."),
) -> None:
    """Show safe AI configuration status."""

    try:
        config = resolve_config()
    except ProviderError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    payload = {
        **config.safe_dict(),
        "capabilities": {"tool_calls": True, "streaming": False},
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        table = Table(title="Hephaestus AI Status")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for key, value in payload.items():
            table.add_row(
                key.replace("_", " ").title(),
                json.dumps(value) if isinstance(value, dict) else str(value),
            )
        Console(width=140).print(table)


@app.command("design")
def design(
    prompt: Annotated[str, typer.Argument(help="Natural-language mobile robot design request.")],
    project_path: Annotated[Path, typer.Option("--project", help="Project directory or file.")],
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without committing."),
    yes: bool = typer.Option(False, "--yes", help="Apply without interactive prompt."),
    no_generate: bool = typer.Option(
        False, "--no-generate", help="Commit the canonical design without generating artifacts."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit response as machine-readable JSON."
    ),
) -> None:
    """Create the supported differential-drive robot from natural language."""

    confirmation_required = json_output and not yes and not dry_run

    def confirm(preview: dict[str, object]) -> bool:
        if dry_run or confirmation_required:
            if dry_run and not json_output:
                _render_mutation_preview(preview)
            return False
        if yes:
            return True
        return _confirm_mutation(preview)

    try:
        result = _run_ai_turn(
            prompt,
            project_path=project_path,
            allow_mutation=True,
            confirm=confirm,
            workflow_policy=mobile_robot_design_tool_policy(),
        )
    except (
        ProviderError,
        OrchestrationError,
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        ValueError,
    ) as exc:
        _render_ai_error(exc, json_output)
        raise typer.Exit(1) from exc

    payload = _design_ai_payload(
        result,
        dry_run=dry_run,
        confirmation_required=confirmation_required,
    )
    exit_code = 0
    generation_payload: dict[str, object] | None = None

    if payload["status"] == "design_committed" and not no_generate:
        try:
            _verify_design_generation_gate(project_path, result)
            generation = generate_project_artifacts(project_path, target=GenerationTarget.ALL)
        except (
            CadGenerationError,
            RosGazeboGenerationError,
            GenerationError,
            OSError,
            ValueError,
        ) as exc:
            generation_payload = {"status": "failed", "error": str(exc)}
            payload["status"] = "generation_failed"
            payload["generated"] = False
            payload["generation"] = generation_payload
            payload["safe_failure_reason"] = str(exc)
            exit_code = 1
        else:
            generation_payload = generation.to_dict()
            payload["generation"] = generation_payload
            if generation_payload.get("status") != "generated":
                payload["status"] = "generation_failed"
                payload["generated"] = False
                payload["safe_failure_reason"] = "artifact generation did not complete"
                exit_code = 1
            else:
                payload["status"] = "completed"
                payload["generated"] = True
                manifest_path = (
                    resolve_project_file(project_path).parent
                    / "generated"
                    / "hephaestus.artifacts.json"
                )
                payload["artifact_manifest_path"] = str(manifest_path)

    if payload["status"] in {"interpretation_failed", "wrong_operation_rejected"}:
        exit_code = 1

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_design_ai_outcome(result, payload, generation_payload)

    if exit_code:
        raise typer.Exit(exit_code)


@concept_app.command("create")
def concept_create(
    description: Annotated[
        str, typer.Argument(help="Natural-language bounded robot concept request.")
    ],
    project_path: Annotated[Path, typer.Option("--project", help="Project directory or file.")],
    yes: bool = typer.Option(False, "--yes", help="Commit without interactive confirmation."),
    no_generate: bool = typer.Option(
        False, "--no-generate", help="Commit the concept without generating artifacts."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit response as machine-readable JSON."
    ),
) -> None:
    """Create a bounded experimental concept robot from natural language."""

    try:
        project = load_project(project_path)
        config = resolve_config()
        planner = ConceptPlanner(
            _create_provider(config), max_completion_tokens=config.max_completion_tokens
        )
        planned = planner.create(description)
        compilation = compile_concept_robot_design(project, planned.spec)
    except (
        ProviderError,
        ConceptPlanningError,
        ConceptError,
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        ValueError,
    ) as exc:
        _render_ai_error(exc, json_output)
        raise typer.Exit(1) from exc
    _finish_concept_command(
        project_path,
        operation="create",
        compilation=compilation,
        planning=planned.to_dict(),
        yes=yes,
        no_generate=no_generate,
        json_output=json_output,
    )


@concept_app.command("edit")
def concept_edit(
    change_request: Annotated[str, typer.Argument(help="Natural-language concept edit request.")],
    project_path: Annotated[Path, typer.Option("--project", help="Project directory or file.")],
    yes: bool = typer.Option(False, "--yes", help="Commit without interactive confirmation."),
    no_generate: bool = typer.Option(
        False, "--no-generate", help="Commit the concept edit without generating artifacts."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit response as machine-readable JSON."
    ),
) -> None:
    """Edit the current bounded concept robot from natural language."""

    try:
        project = load_project(project_path)
        current_spec = concept_robot_spec(project)
        config = resolve_config()
        planner = ConceptPlanner(
            _create_provider(config), max_completion_tokens=config.max_completion_tokens
        )
        planned = planner.edit(current_spec, change_request)
        compilation = compile_concept_robot_design(
            project, planned.spec, previous_spec=current_spec
        )
    except (
        ProviderError,
        ConceptPlanningError,
        ConceptError,
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        ValueError,
    ) as exc:
        _render_ai_error(exc, json_output)
        raise typer.Exit(1) from exc
    _finish_concept_command(
        project_path,
        operation="edit",
        compilation=compilation,
        planning=planned.to_dict(),
        yes=yes,
        no_generate=no_generate,
        json_output=json_output,
    )


@app.command("ask")
def ask(
    prompt: Annotated[str, typer.Argument(help="Prompt for one non-interactive AI request.")],
    project_path: Annotated[
        Path | None, typer.Option("--project", help="Project directory or file.")
    ] = None,
    json_output: bool = typer.Option(
        False, "--json", help="Emit response as machine-readable JSON."
    ),
) -> None:
    """Ask Hephaestus one project-aware, read-only question."""

    try:
        result = _run_ai_turn(prompt, project_path=project_path, allow_mutation=False)
    except (
        ProviderError,
        OrchestrationError,
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        ValueError,
    ) as exc:
        _render_ai_error(exc, json_output)
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _render_model_response(result.content)


@app.command("chat")
def chat(
    project_path: Annotated[
        Path | None, typer.Option("--project", help="Project directory or file.")
    ] = None,
) -> None:
    """Start an interactive in-memory Hephaestus session."""

    try:
        config = resolve_config()
        provider = _create_provider(config)
    except ProviderError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    executor = _project_executor(project_path)
    orchestrator = HephaestusOrchestrator(
        provider=provider,
        tool_executor=executor,
        project_path=project_path,
    )
    project_label = str(project_path) if project_path else "none"
    typer.echo(
        "Hephaestus chat | "
        f"provider={provider.provider_id} "
        f"model={provider.model_id} "
        f"project={project_label}"
    )
    typer.echo("Type /help for commands, /exit to quit.")
    while True:
        try:
            user_input = input("hephaestus> ")
        except EOFError:
            typer.echo("")
            break
        except KeyboardInterrupt:
            typer.echo("")
            break
        command = user_input.strip()
        if not command:
            continue
        if command in {"/exit", "/quit"}:
            break
        if command == "/help":
            typer.echo("Commands: /help, /clear, /status, /exit, /quit")
            continue
        if command == "/clear":
            orchestrator.clear()
            typer.echo("Conversation cleared.")
            continue
        if command == "/status":
            typer.echo(
                f"provider={provider.provider_id} model={provider.model_id} project={project_label}"
            )
            continue
        try:
            result = orchestrator.ask(command, allow_mutation=True, confirm=_confirm_mutation)
        except (ProviderError, OrchestrationError) as exc:
            _render_ai_error(exc, json_output=False)
            continue
        except (ProjectPersistenceError, UnsupportedSchemaVersionError, ValueError):
            typer.echo("Error: AI turn failed before completion", err=True)
            continue
        _render_model_response(result.content)


def _run_default_interface(*, plain: bool, no_color: bool, reduced_motion: bool) -> None:
    if no_color:
        os.environ.setdefault("NO_COLOR", "1")
    if plain or not _default_tui_capable():
        _run_default_agent_shell()
        return
    try:
        _run_default_forge_tui(no_color=no_color, reduced_motion=reduced_motion)
    except Exception as exc:
        typer.echo(f"Forge TUI unavailable; using plain shell. ({exc})", err=True)
        _run_default_agent_shell()


def _default_tui_capable() -> bool:
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    if stdin is None or stdout is None:
        return False
    if not stdin.isatty() or not stdout.isatty():
        return False
    try:
        import textual  # noqa: F401
    except Exception:
        return False
    return True


def _run_default_forge_tui(*, no_color: bool, reduced_motion: bool) -> None:
    from hephaestus.tui import create_forge_app

    app = create_forge_app(
        Path.cwd(),
        no_color=no_color,
        reduced_motion=reduced_motion,
    )
    app.run()


def _run_default_agent_shell() -> None:
    startup_directory = Path.cwd()
    local = LocalWorkbenchSession(startup_directory)
    session: HephaestusAgentSession | None = None
    provider = None
    ai_unavailable: str | None = None
    try:
        config = resolve_config()
        provider = _create_provider(config)
        session = HephaestusAgentSession(
            provider=provider,
            startup_directory=startup_directory,
            tool_executor=local.tool_executor,
            capabilities=local.capabilities,
            tool_calling_temperature=config.default_agent_tool_temperature,
            max_tool_rounds=config.default_agent_max_tool_rounds,
        )
    except (ProviderError, OSError) as exc:
        ai_unavailable = str(exc)
    provider_label = provider.provider_id if provider is not None else "unavailable"
    model_label = provider.model_id if provider is not None else "unavailable"
    typer.echo(
        "Hephaestus | "
        f"provider={provider_label} "
        f"model={model_label} "
        f"workspace={startup_directory}"
    )
    if ai_unavailable is not None:
        typer.echo("AI unavailable; local engineering commands remain available.")
    typer.echo("Type /help for commands, /exit to quit.")
    try:
        while True:
            try:
                user_input = input("hephaestus> ")
            except EOFError:
                typer.echo("")
                break
            except KeyboardInterrupt:
                typer.echo("")
                break
            command = user_input.strip()
            if not command:
                continue
            if command.startswith("/"):
                local_result = local.dispatch(command, confirm=_confirm_local_workbench_action)
                if local_result.clear_requested and session is not None:
                    session.clear()
                _render_local_command_result(local_result)
                if local_result.exit_requested:
                    break
                continue
            if session is None:
                typer.echo("AI unavailable; local engineering commands remain available.")
                continue
            try:
                result = session.ask(command, confirm=_confirm_default_agent_tool)
            except (ProviderError, AgentLoopError, ValueError) as exc:
                _render_ai_error(exc, json_output=False)
                continue
            _render_model_response(result.content)
    finally:
        local.close()


def _confirm_workspace_write(preview: dict[str, object]) -> bool:
    result = preview.get("result") if isinstance(preview, dict) else None
    if isinstance(result, dict):
        tool = preview.get("tool")
        title = (
            "Proposed Workspace Replacement"
            if tool == "workspace_replace_text"
            else "Proposed Workspace Write"
        )
        table = Table(title=title)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Path", str(result.get("relative_path", "")))
        table.add_row("Operation", str(result.get("operation", "")))
        if tool == "workspace_replace_text":
            table.add_row("Old bytes", str(result.get("old_byte_count", "")))
            table.add_row("New bytes", str(result.get("new_byte_count", "")))
            table.add_row("Resulting file bytes", str(result.get("resulting_file_byte_count", "")))
        else:
            table.add_row("Bytes", str(result.get("bytes", "")))
        Console(width=140).print(table)
    question = (
        "Apply this replacement?"
        if preview.get("tool") == "workspace_replace_text"
        else "Write this file?"
    )
    return _confirm_yes_only(question)


def _confirm_default_agent_tool(preview: dict[str, object]) -> bool:
    tool = preview.get("tool") if isinstance(preview, dict) else None
    if tool == "gazebo_launch":
        return _confirm_gazebo_launch(preview)
    if tool == "cadquery_build":
        return _confirm_cadquery_build(preview)
    return _confirm_workspace_write(preview)


def _confirm_local_workbench_action(preview: dict[str, object]) -> bool:
    tool = preview.get("tool") if isinstance(preview, dict) else None
    if tool == "gazebo_launch":
        return _confirm_gazebo_launch(preview)
    if tool == "cadquery_build":
        return _confirm_cadquery_build(preview)
    if tool == "ros_build":
        return _confirm_ros_build(preview)
    if tool == "ros_launch":
        return _confirm_ros_launch(preview)
    return _confirm_workspace_write(preview)


def _confirm_cadquery_build(preview: dict[str, object]) -> bool:
    result = preview.get("result") if isinstance(preview, dict) else None
    if isinstance(result, dict):
        table = Table(title="Proposed CadQuery Build")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Source", str(result.get("relative_source_path", "")))
        table.add_row("Fingerprint", str(result.get("source_fingerprint", "")))
        table.add_row("Output", str(result.get("generated_output_directory", "")))
        table.add_row("Warning", str(result.get("warning", "")))
        Console(width=140).print(table)
    return _confirm_yes_only("Execute generated CadQuery source?")


def _confirm_gazebo_launch(preview: dict[str, object]) -> bool:
    result = preview.get("result") if isinstance(preview, dict) else None
    if isinstance(result, dict):
        table = Table(title="Proposed Gazebo Launch")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("World", str(result.get("relative_path", "")))
        table.add_row("Mode", "headless" if result.get("headless") else "GUI")
        table.add_row("Executable", str(result.get("executable", "")))
        argv = result.get("argv_summary", [])
        argv_summary = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
        table.add_row("Arguments", argv_summary)
        table.add_row("SDF validation", "passed" if result.get("validation_passed") else "failed")
        Console(width=140).print(table)
    return _confirm_yes_only("Launch Gazebo?")


def _confirm_ros_build(preview: dict[str, object]) -> bool:
    result = preview.get("result") if isinstance(preview, dict) else None
    if isinstance(result, dict):
        table = Table(title="Proposed ROS Build")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Workspace", str(result.get("relative_workspace", "")))
        argv = result.get("argv", [])
        arguments = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
        table.add_row("Arguments", arguments)
        table.add_row("Domain", str(result.get("domain_id", "")))
        table.add_row("Warning", str(result.get("warning", "")))
        Console(width=140).print(table)
    return _confirm_yes_only("Build this ROS workspace?")


def _confirm_ros_launch(preview: dict[str, object]) -> bool:
    result = preview.get("result") if isinstance(preview, dict) else None
    if isinstance(result, dict):
        table = Table(title="Proposed ROS Launch")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Workspace", str(result.get("relative_workspace", "")))
        table.add_row("Package", str(result.get("package", "")))
        table.add_row("Launch File", str(result.get("launch_file", "")))
        argv = result.get("argv", [])
        arguments = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
        table.add_row("Arguments", arguments)
        table.add_row("Domain", str(result.get("domain_id", "")))
        Console(width=140).print(table)
    return _confirm_yes_only("Launch this ROS simulation?")


def _finish_concept_command(
    project_path: Path,
    *,
    operation: str,
    compilation: CompiledConceptRobot,
    planning: dict[str, object],
    yes: bool,
    no_generate: bool,
    json_output: bool,
) -> None:
    preview = compilation.preview(current_revision=compilation.project.revision - 1)
    payload: dict[str, object] = {
        "schema": "hephaestus.concept.command_result.v1",
        "operation": operation,
        "status": "previewed",
        "applied": False,
        "generated": False,
        "planning": planning,
        "preview": preview,
    }
    if json_output and not yes:
        payload["status"] = "confirmation_required"
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not yes:
        _render_concept_preview(preview)
        if not _confirm_yes_only("Commit this concept design?"):
            payload["status"] = "denied"
            if json_output:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo("Concept design not committed.")
            return
    try:
        _commit_concept_compilation(project_path, compilation)
    except (ProjectPersistenceError, UnsupportedSchemaVersionError, ValueError) as exc:
        payload.update({"status": "failed", "error": str(exc)})
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    payload.update(
        {
            "status": "design_committed",
            "applied": True,
            "revision": compilation.project.revision,
            "after_fingerprint": compilation.after_fingerprint,
        }
    )
    if no_generate:
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Concept design committed at revision {compilation.project.revision}.")
        return
    try:
        generation = generate_project_artifacts(project_path, target=GenerationTarget.ALL)
    except (
        CadGenerationError,
        RosGazeboGenerationError,
        GenerationError,
        OSError,
        ValueError,
    ) as exc:
        payload.update(
            {
                "status": "generation_failed",
                "generated": False,
                "generation": {"status": "failed", "error": str(exc)},
                "safe_failure_reason": str(exc),
            }
        )
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo("Design committed; artifact generation failed.", err=True)
            typer.echo(f"Committed revision: {compilation.project.revision}")
            typer.echo(f"Committed fingerprint: {compilation.after_fingerprint}")
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    payload["generation"] = generation.to_dict()
    if generation.status != "generated":
        payload.update(
            {
                "status": "generation_failed",
                "generated": False,
                "safe_failure_reason": "artifact generation did not complete",
            }
        )
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo("Design committed; artifact generation failed.", err=True)
            _render_generation_plan(generation.to_dict())
        raise typer.Exit(1)
    payload.update(
        {
            "status": "completed",
            "generated": True,
            "artifact_manifest_path": str(
                resolve_project_file(project_path).parent
                / "generated"
                / "hephaestus.artifacts.json"
            ),
        }
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Concept design committed at revision {compilation.project.revision}.")
        _render_generation_plan(generation.to_dict())


def _commit_concept_compilation(project_path: Path, compilation: CompiledConceptRobot) -> None:
    current = load_project(project_path)
    if fingerprint_project(current) != compilation.before_fingerprint:
        raise ValueError("active project changed after concept evaluation")
    if not compilation.validation.is_valid:
        codes = ", ".join(finding.code for finding in compilation.validation.findings)
        raise ValueError(f"compiled concept is invalid: {codes}")
    save_project(compilation.project, project_path, overwrite=True)
    committed = load_project(project_path)
    if committed.revision != compilation.project.revision:
        raise ValueError("committed concept revision could not be verified")
    if fingerprint_project(committed) != compilation.after_fingerprint:
        raise ValueError("committed concept fingerprint could not be verified")


def _render_concept_preview(preview: dict[str, object]) -> None:
    table = Table(title="Hephaestus Concept Preview")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key in (
        "robot_name",
        "locomotion_mode",
        "world_template",
        "current_revision",
        "proposed_revision",
        "before_fingerprint",
        "after_fingerprint",
    ):
        table.add_row(key.replace("_", " ").title(), str(preview.get(key, "")))
    counts = preview.get("entity_counts")
    if isinstance(counts, dict):
        table.add_row("Entity Counts", json.dumps(counts, sort_keys=True))
    assumptions = preview.get("assumptions")
    if isinstance(assumptions, list):
        table.add_row("Assumptions", "\n".join(str(item) for item in assumptions) or "none")
    warnings = preview.get("warnings")
    if isinstance(warnings, list):
        table.add_row("Warnings", "\n".join(str(item) for item in warnings) or "none")
    diff = preview.get("semantic_diff")
    if isinstance(diff, list) and diff:
        table.add_row("Semantic Diff", "\n".join(str(item) for item in diff))
    Console(width=140).print(table)
    validation = preview.get("validation")
    if isinstance(validation, dict):
        findings = validation.get("findings", [])
        if findings:
            _render_validation_report(
                ValidationReport(
                    tuple(
                        ValidationFinding(
                            code=str(item["code"]),
                            severity=ValidationSeverity(str(item["severity"])),
                            message=str(item["message"]),
                            path=str(item["path"]) if "path" in item else None,
                            evidence=item.get("evidence"),
                            remediation=item.get("remediation"),
                        )
                        for item in findings
                        if isinstance(item, dict)
                    )
                )
            )


def _run_ai_turn(
    prompt: str,
    *,
    project_path: Path | None,
    allow_mutation: bool,
    confirm=None,
    expected_tool_names: tuple[str, ...] | None = None,
    workflow_policy: ToolWorkflowPolicy | None = None,
) -> OrchestrationResult:
    config = resolve_config()
    provider = _create_provider(config)
    orchestrator = HephaestusOrchestrator(
        provider=provider,
        tool_executor=_project_executor(project_path, workflow_policy=workflow_policy),
        project_path=project_path,
    )
    return orchestrator.ask(
        prompt,
        allow_mutation=allow_mutation,
        confirm=confirm,
        expected_tool_names=expected_tool_names,
    )


def _create_provider(config):
    return default_provider_registry().create(config)


def _project_executor(
    project_path: Path | None, *, workflow_policy: ToolWorkflowPolicy | None = None
) -> ProjectToolExecutor | None:
    if project_path is None:
        return None
    return ProjectToolExecutor(
        default_project_tool_registry(),
        ToolContext(project_path),
        workflow_policy=workflow_policy,
    )


DESIGN_OPERATION_TOOL = "project_create_mobile_robot"
GENERATION_CAPABILITIES = (
    "cad",
    "ros2",
    "gazebo",
    "ros_gz_sim",
    "ros_gz_bridge",
    "gz_ros2_control",
)


def _design_ai_payload(
    result: OrchestrationResult, *, dry_run: bool, confirmation_required: bool
) -> dict[str, object]:
    payload: dict[str, object] = result.to_dict()
    outcome = result.mutation_outcome
    payload["operation"] = DESIGN_OPERATION_TOOL
    payload["applied"] = False
    payload["generated"] = False
    if _has_tool_status(result, "wrong_operation_rejected") or _has_tool_status(
        result, "unexpected"
    ):
        payload["status"] = "wrong_operation_rejected"
        payload["safe_failure_reason"] = result.content
        return payload
    if outcome is None:
        payload["status"] = "interpretation_failed"
        payload["safe_failure_reason"] = result.content
        return payload
    payload["applied"] = outcome.applied
    if outcome.tool != DESIGN_OPERATION_TOOL:
        payload["status"] = "wrong_operation_rejected"
        payload["safe_failure_reason"] = (
            f"design workflow expected {DESIGN_OPERATION_TOOL}; got {outcome.tool}"
        )
        return payload
    if (dry_run or confirmation_required) and outcome.status == "declined":
        payload["status"] = "confirmation_required"
        return payload
    if outcome.status == "declined":
        payload["status"] = "declined"
        return payload
    if outcome.status != "applied":
        payload["status"] = "interpretation_failed"
        payload["safe_failure_reason"] = outcome.reason
        return payload
    payload["status"] = "design_committed"
    return payload


def _has_tool_status(result: OrchestrationResult, status: str) -> bool:
    return any(activity.status == status for activity in result.tool_activity)


def _verify_design_generation_gate(project_path: Path, result: OrchestrationResult) -> None:
    outcome = result.mutation_outcome
    if outcome is None or not outcome.applied or outcome.tool != DESIGN_OPERATION_TOOL:
        raise GenerationError("mobile robot design operation was not committed")
    project = load_project(project_path)
    if project.revision != outcome.after_revision:
        raise GenerationError("committed design revision could not be verified")
    if fingerprint_project(project) != outcome.after_fingerprint:
        raise GenerationError("committed design fingerprint could not be verified")
    mobile_robot_spec(project)
    validation = validate_project(project)
    mobile_findings = validate_mobile_robot_project(project)
    if not validation.is_valid or mobile_findings:
        codes = [finding.code for finding in validation.findings]
        codes.extend(finding.code for finding in mobile_findings)
        raise GenerationError("committed mobile robot design is invalid: " + ", ".join(codes))
    failures = collect_capabilities().require(GENERATION_CAPABILITIES)
    if failures:
        details = ", ".join(
            f"{item.capability_id}={item.status.value}/{item.compatibility.value}"
            for item in failures
        )
        raise GenerationError("required backend capabilities are not ready: " + details)


def _render_design_ai_outcome(
    result: OrchestrationResult,
    payload: dict[str, object],
    generation_payload: dict[str, object] | None,
) -> None:
    status = str(payload["status"])
    if status == "generation_failed":
        typer.echo("Design committed; artifact generation failed.", err=True)
        outcome = result.mutation_outcome
        if outcome is not None:
            typer.echo(f"Committed revision: {outcome.after_revision}")
            typer.echo(f"Committed fingerprint: {outcome.after_fingerprint}")
        reason = payload.get("safe_failure_reason")
        if reason:
            typer.echo(f"Error: {reason}", err=True)
        typer.echo("Retry generation with: hephaestus generate PROJECT_PATH")
        return
    if status in {"wrong_operation_rejected", "interpretation_failed"}:
        typer.echo(f"Error: {payload.get('safe_failure_reason', result.content)}", err=True)
        return
    _render_model_response(result.content)
    if generation_payload is not None:
        _render_generation_plan(generation_payload)


def _render_mutation_preview(preview: dict[str, object]) -> None:
    result = preview.get("result") if isinstance(preview, dict) else None
    if not isinstance(result, dict):
        return
    human_preview = result.get("human_preview")
    if isinstance(human_preview, dict):
        table = Table(title="Proposed Change")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Change", str(human_preview.get("title", "Project change")))
        table.add_row("Statement", str(human_preview.get("statement", "")))
        table.add_row("Current Revision", str(human_preview.get("current_revision", "")))
        table.add_row("Proposed Revision", str(human_preview.get("proposed_revision", "")))
        table.add_row("Validation", str(human_preview.get("validation", "unknown")))
        Console(width=140).print(table)
    else:
        table = Table(title="Proposed Project Mutation")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for key in ("summary", "revision", "before_fingerprint", "after_fingerprint"):
            if key in result:
                table.add_row(key.replace("_", " ").title(), str(result[key]))
        Console(width=140).print(table)
    validation = result.get("validation")
    if isinstance(validation, dict):
        findings = validation.get("findings", [])
        if findings:
            _render_validation_report(
                ValidationReport(
                    tuple(
                        ValidationFinding(
                            code=str(item["code"]),
                            severity=ValidationSeverity(str(item["severity"])),
                            message=str(item["message"]),
                            path=str(item["path"]) if "path" in item else None,
                        )
                        for item in findings
                        if isinstance(item, dict)
                    )
                )
            )


def _confirm_mutation(preview: dict[str, object]) -> bool:
    _render_mutation_preview(preview)
    result = preview.get("result") if isinstance(preview, dict) else None
    prompt = "Apply this change?"
    if isinstance(result, dict) and not isinstance(result.get("human_preview"), dict):
        prompt = "Apply this ChangeSet?"
    return _confirm_yes_only(prompt)


def _confirm_yes_only(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        typer.echo("")
        return False
    return answer.strip().lower() in {"y", "yes"}


def _render_generation_plan(payload: dict[str, object]) -> None:
    table = Table(title="Hephaestus Artifact Generation")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key in ("status", "target", "dry_run", "project_path"):
        if key in payload:
            table.add_row(key.replace("_", " ").title(), str(payload[key]))
    plan = payload.get("plan")
    if isinstance(plan, list):
        table.add_row("Plan", "\n".join(str(item) for item in plan))
    manifest = payload.get("manifest")
    if isinstance(manifest, dict):
        artifacts = manifest.get("artifacts", [])
        table.add_row("Artifacts", str(len(artifacts) if isinstance(artifacts, list) else 0))
        table.add_row("Revision", str(manifest.get("project_revision", "")))
        table.add_row("Fingerprint", str(manifest.get("project_fingerprint", "")))
    Console(width=140).print(table)


def _render_artifact_state(payload: dict[str, object]) -> None:
    table = Table(title="Hephaestus Artifacts")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key in (
        "status",
        "project_revision",
        "project_fingerprint",
        "generation_id",
        "artifact_count",
    ):
        if key in payload:
            table.add_row(key.replace("_", " ").title(), str(payload[key]))
    roles = payload.get("roles")
    if isinstance(roles, dict):
        table.add_row("Roles", json.dumps(roles, sort_keys=True))
    paths = payload.get("important_paths")
    if isinstance(paths, list):
        table.add_row("Important Paths", "\n".join(str(item) for item in paths[:12]))
    Console(width=140).print(table)


def _render_execution_payload(payload: dict[str, object]) -> None:
    table = Table(title="Hephaestus Execution Plan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(
            key.replace("_", " ").title(),
            json.dumps(value) if isinstance(value, list | dict) else str(value),
        )
    Console(width=140).print(table)


def _render_generation_result(payload: dict[str, object]) -> None:
    table = Table(title="Generated ROS 2/Gazebo Workspace")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Workspace", str(payload["workspace"]))
    table.add_row("Model", str(payload["model_name"]))
    files = payload.get("files", [])
    warnings = payload.get("warnings", [])
    table.add_row("Files", str(len(files) if isinstance(files, list) else 0))
    table.add_row("Warnings", str(len(warnings) if isinstance(warnings, list) else 0))
    Console(width=140).print(table)


def _render_execution_error(exc: Exception, json_output: bool) -> None:
    payload = {"status": "failed", "error": str(exc)}
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Error: {exc}", err=True)


def _finish_execution_result(result: ExecutionResult, json_output: bool) -> None:
    payload = result.to_dict()
    payload["status"] = "succeeded" if result.succeeded else "failed"
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_execution_result(result)
    if not result.succeeded:
        raise typer.Exit(1)


def _render_execution_result(result: ExecutionResult) -> None:
    table = Table(title="Hephaestus Constrained Execution")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Command", result.command.value)
    table.add_row("Working Directory", str(result.cwd))
    table.add_row("Return Code", str(result.returncode))
    table.add_row("Timed Out", str(result.timed_out).lower())
    table.add_row("Argv", " ".join(result.argv))
    if result.stdout:
        table.add_row("Stdout", result.stdout)
    if result.stderr:
        table.add_row("Stderr", result.stderr)
    Console(width=140).print(table)


def _standalone_tool_payload(result: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "succeeded" if result.get("status") == "ok" else "failed",
        "tool": result.get("tool"),
    }
    if result.get("status") != "ok":
        payload["error"] = result.get("error", "tool failed")
        if "evidence" in result:
            payload["evidence"] = result["evidence"]
        return payload
    inner = result.get("result")
    if isinstance(inner, dict):
        payload.update(inner)
        if "generated_part_names" in inner and isinstance(inner["generated_part_names"], list):
            payload["part_count"] = len(inner["generated_part_names"])
            payload["part_names"] = sorted(str(item) for item in inner["generated_part_names"])
        if inner.get("committed") is False:
            payload["status"] = "declined"
    return payload


def _render_local_command_result(result: LocalCommandResult) -> None:
    if result.status == "error":
        typer.echo(f"Error: {result.message}", err=True)
        if result.payload:
            _render_local_payload(result.command, result.payload)
        return
    if result.command == "/help":
        typer.echo(result.message)
        return
    if result.message:
        typer.echo(result.message)
    if result.payload:
        _render_local_payload(result.command, dict(result.payload))


def _render_local_payload(title: str, payload: Mapping[str, object]) -> None:
    table = Table(title=title)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in payload.items():
        if key in {"content"}:
            continue
        if isinstance(value, list):
            rendered = "\n".join(
                json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
                for item in value[:20]
            )
        elif isinstance(value, dict):
            rendered = json.dumps(value, sort_keys=True)
        else:
            rendered = str(value)
        table.add_row(key.replace("_", " ").title(), rendered)
    Console(width=140).print(table)


def _render_ai_error(exc: Exception, json_output: bool) -> None:
    payload = {"status": "failed", "error": str(exc)}
    details = getattr(exc, "details", None)
    if details is not None and hasattr(details, "to_dict"):
        safe_details = details.to_dict()
        if safe_details:
            payload["details"] = safe_details
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        suffix = _safe_error_suffix(payload.get("details"))
        typer.echo(f"Error: {exc}{suffix}", err=True)


def _safe_error_suffix(details: object) -> str:
    if not isinstance(details, dict):
        return ""
    parts: list[str] = []
    if "status_code" in details:
        parts.append(f"status={details['status_code']}")
    if "provider_code" in details:
        parts.append(f"code={details['provider_code']}")
    return f" ({', '.join(parts)})" if parts else ""


def _render_model_response(content: str, console: Console | None = None) -> None:
    if not content.strip():
        (console or Console()).print("No assistant response was returned.")
        return
    (console or Console()).print(Markdown(content))


@project_app.command("init")
def project_init(
    path: Annotated[
        Path, typer.Argument(help="Directory where hephaestus.project.json will be created.")
    ],
    name: str = typer.Option(..., "--name", help="Human-readable project name."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing project file."),
) -> None:
    """Create an empty engineering project file."""

    project = EngineeringProject.empty(name=name)
    try:
        project_file = save_project(project, path, overwrite=force)
    except ProjectPersistenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(str(project_file))


@project_app.command("inspect")
def project_inspect(
    path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit summary as machine-readable JSON."
    ),
) -> None:
    """Show a summary of an engineering project."""

    project = _load_project_or_exit(path)
    summary = _project_summary(project)
    if json_output:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _render_project_summary(summary)


@project_app.command("validate")
def project_validate(
    path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    json_output: bool = typer.Option(
        False, "--json", help="Emit findings as machine-readable JSON."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit nonzero when errors or fatal findings exist."
    ),
) -> None:
    """Validate an engineering project."""

    try:
        project = load_project(path)
        report = validate_project(project)
    except UnsupportedSchemaVersionError as exc:
        report = ValidationReport(
            (
                _fatal_finding(
                    "project.schema.unsupported",
                    str(exc),
                    "schema_version",
                    exc.schema_version,
                ),
            )
        )
    except ProjectPersistenceError as exc:
        report = ValidationReport((_fatal_finding("project.load.failed", str(exc)),))

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _render_validation_report(report)
    if strict and not report.is_valid:
        raise typer.Exit(1)


@design_app.command("warehouse-inspection-robot")
def design_warehouse_inspection_robot(
    project_path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    robot_name: str = typer.Option(..., "--name", help="Robot name."),
    mission: str = typer.Option(..., "--mission", help="Warehouse inspection mission."),
    warehouse_environment: str = typer.Option(
        ..., "--environment", help="Warehouse operating environment."
    ),
    robot_family: str = typer.Option(
        "differential_drive", "--family", help="Robot family; only differential_drive is supported."
    ),
    payload_kg: float | None = typer.Option(None, "--payload-kg", min=0.0),
    max_speed_mps: float | None = typer.Option(None, "--max-speed-mps", min=0.0),
    chassis_length_m: float | None = typer.Option(None, "--chassis-length-m", min=0.0),
    chassis_width_m: float | None = typer.Option(None, "--chassis-width-m", min=0.0),
    chassis_height_m: float | None = typer.Option(None, "--chassis-height-m", min=0.0),
    wheel_diameter_m: float | None = typer.Option(None, "--wheel-diameter-m", min=0.0),
    wheel_width_m: float | None = typer.Option(None, "--wheel-width-m", min=0.0),
    has_lidar: Annotated[
        bool | None, typer.Option("--lidar/--no-lidar", help="Include a 2D lidar.")
    ] = None,
    has_imu: Annotated[bool | None, typer.Option("--imu/--no-imu", help="Include an IMU.")] = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing."),
    json_output: bool = typer.Option(False, "--json", help="Emit result as machine-readable JSON."),
    yes: bool = typer.Option(False, "--yes", help="Apply without interactive prompt."),
) -> None:
    """Compile a narrow differential-drive warehouse inspection robot design."""

    try:
        base_project = load_project(project_path)
        compilation = compile_mobile_robot_design(
            base_project,
            MobileRobotIntent(
                robot_family=robot_family,
                robot_name=robot_name,
                mission_summary=mission,
                payload_kg=payload_kg,
                max_speed_mps=max_speed_mps,
                chassis_length_m=chassis_length_m,
                chassis_width_m=chassis_width_m,
                chassis_height_m=chassis_height_m,
                wheel_radius_m=(wheel_diameter_m / 2.0 if wheel_diameter_m is not None else None),
                wheel_width_m=wheel_width_m,
                include_lidar=True if has_lidar is None else has_lidar,
                include_imu=True if has_imu is None else has_imu,
                indoor_warehouse=_is_indoor_warehouse(warehouse_environment),
            ),
            dry_run=dry_run,
        )
    except (
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        ValueError,
    ) as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if dry_run:
        payload = _compiled_mobile_robot_payload(compilation, dry_run=True)
        payload["status"] = "planned" if compilation.validation.is_valid else "invalid"
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _render_design_result(payload)
        if not compilation.validation.is_valid:
            raise typer.Exit(1)
        return

    if not yes and not _confirm_mutation(
        {"result": _compiled_mobile_robot_payload(compilation, dry_run=True)}
    ):
        payload = {
            "status": "denied",
            "not_applied": True,
            "summary": str(compilation.preview["summary"]),
            "before_fingerprint": compilation.before_fingerprint,
        }
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo("Change not applied.")
        return

    try:
        current = load_project(project_path)
        if fingerprint_project(current) != compilation.before_fingerprint:
            raise ValueError("active project changed after design evaluation")
        if not compilation.validation.is_valid:
            codes = ", ".join(finding.code for finding in compilation.validation.findings)
            raise ValueError(f"compiled design is invalid: {codes}")
        save_project(compilation.project, project_path, overwrite=True)
    except (ProjectPersistenceError, UnsupportedSchemaVersionError, ValueError) as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = _compiled_mobile_robot_payload(compilation, dry_run=False)
    payload["status"] = "applied"
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_design_result(payload)


@project_app.command("apply")
def project_apply(
    project_path: Annotated[
        Path, typer.Argument(help="Project directory or hephaestus.project.json path.")
    ],
    changeset_path: Annotated[Path, typer.Argument(help="Serialized ChangeSet JSON file.")],
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without writing changes."),
    json_output: bool = typer.Option(False, "--json", help="Emit result as machine-readable JSON."),
) -> None:
    """Apply a typed engineering ChangeSet to a project."""

    try:
        result = apply_changeset_file(project_path, changeset_path, dry_run=dry_run)
    except (
        ProjectPersistenceError,
        UnsupportedSchemaVersionError,
        ChangeSetLoadError,
        ChangeError,
    ) as exc:
        payload = {"status": "failed", "error": str(exc)}
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = result.to_dict()
    payload["status"] = "valid" if result.validation.is_valid else "invalid"
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_change_result(payload)
    if not result.validation.is_valid:
        raise typer.Exit(1)


def _load_project_or_exit(path: Path) -> EngineeringProject:
    try:
        return load_project(path)
    except (ProjectPersistenceError, UnsupportedSchemaVersionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def _fatal_finding(
    code: str,
    message: str,
    path: str | None = None,
    evidence: object | None = None,
) -> ValidationFinding:
    return ValidationFinding(code, ValidationSeverity.FATAL, message, path, evidence)


def _project_summary(project: EngineeringProject) -> dict[str, object]:
    counts = project.collection_counts()
    return {
        "name": project.metadata.name,
        "schema_version": project.schema_version,
        "revision": project.revision,
        "counts": counts,
        "assumptions": len(project.assumptions),
        "requirements": len(project.requirements),
        "latest_validation_summary": project.validation_state.latest_summary,
    }


def _is_indoor_warehouse(value: str) -> bool:
    text = value.casefold()
    return "warehouse" in text and "indoor" in text and "outdoor" not in text


def _compiled_mobile_robot_payload(
    compilation: CompiledMobileRobot, *, dry_run: bool
) -> dict[str, object]:
    payload = compilation.to_result(dry_run=dry_run).to_dict()
    payload["operation"] = "create_mobile_robot"
    payload["design_preview"] = compilation.preview
    payload["human_preview"] = {
        "title": compilation.preview["summary"],
        "statement": compilation.preview["robot"]["mission"],
        "current_revision": compilation.preview["current_revision"],
        "proposed_revision": compilation.preview["proposed_revision"],
        "validation": "passed" if compilation.validation.is_valid else "has findings",
    }
    return payload


def _render_project_summary(summary: dict[str, object]) -> None:
    console = Console(width=140)
    table = Table(title="Hephaestus Engineering Project")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key in ("name", "schema_version", "revision", "assumptions", "requirements"):
        table.add_row(key.replace("_", " ").title(), str(summary[key]))
    table.add_row("Latest Validation", str(summary["latest_validation_summary"]))
    console.print(table)

    counts = summary["counts"]
    if isinstance(counts, dict):
        counts_table = Table(title="Engineering Categories")
        counts_table.add_column("Category", style="bold", no_wrap=True)
        counts_table.add_column("Count", justify="right")
        for key, value in counts.items():
            counts_table.add_row(key.replace("_", " "), str(value))
        console.print(counts_table)


def _render_validation_report(report: ValidationReport) -> None:
    table = Table(title="Hephaestus Project Validation")
    table.add_column("Severity")
    table.add_column("Code", no_wrap=True)
    table.add_column("Message")
    table.add_column("Path")
    for finding in report.findings:
        table.add_row(
            _format_validation_severity(finding.severity),
            finding.code,
            finding.message,
            finding.path or "",
        )
    Console(width=140).print(table)


def _render_design_result(result: dict[str, object]) -> None:
    table = Table(title="Hephaestus Design Result")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key in (
        "status",
        "summary",
        "dry_run",
        "revision",
        "before_fingerprint",
        "after_fingerprint",
    ):
        if key in result:
            table.add_row(key.replace("_", " ").title(), str(result[key]))
    Console(width=140).print(table)

    validation = result.get("validation")
    if isinstance(validation, dict):
        findings = validation.get("findings", [])
        if findings:
            report = ValidationReport(
                tuple(
                    ValidationFinding(
                        code=str(item["code"]),
                        severity=ValidationSeverity(str(item["severity"])),
                        message=str(item["message"]),
                        path=str(item["path"]) if "path" in item else None,
                        evidence=item.get("evidence"),
                        remediation=item.get("remediation"),
                    )
                    for item in findings
                    if isinstance(item, dict)
                )
            )
            _render_validation_report(report)


def _render_change_result(result: dict[str, object]) -> None:
    table = Table(title="Hephaestus ChangeSet Result")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key in (
        "status",
        "summary",
        "dry_run",
        "revision",
        "before_fingerprint",
        "after_fingerprint",
    ):
        table.add_row(key.replace("_", " ").title(), str(result[key]))
    Console(width=140).print(table)

    validation = result.get("validation")
    if isinstance(validation, dict):
        findings = validation.get("findings", [])
        if findings:
            report = ValidationReport(
                tuple(
                    ValidationFinding(
                        code=str(item["code"]),
                        severity=ValidationSeverity(str(item["severity"])),
                        message=str(item["message"]),
                        path=str(item["path"]) if "path" in item else None,
                        evidence=item.get("evidence"),
                        remediation=item.get("remediation"),
                    )
                    for item in findings
                    if isinstance(item, dict)
                )
            )
            _render_validation_report(report)


def _render_capability_report(report: CapabilityReport) -> None:
    table = Table(title="Hephaestus Backend Capabilities", show_lines=False)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Capability")
    table.add_column("Status")
    table.add_column("Compatibility")
    table.add_column("Evidence")
    for capability in report.capabilities:
        table.add_row(
            capability.capability_id,
            capability.display_name,
            _format_status(str(capability.status)),
            str(capability.compatibility),
            "; ".join(capability.evidence[:2]),
        )
    Console(width=140).print(table)


def _render_setup_plan(plan: SetupPlan) -> None:
    console = Console(width=140)
    console.print(
        "Read-only setup plan. No installation, sudo, downloads, or repository changes will run."
    )
    table = Table(title="Hephaestus Setup Plan", show_lines=False)
    table.add_column("Capability", style="bold", no_wrap=True)
    table.add_column("State")
    table.add_column("What It Enables")
    table.add_column("Requires")
    for item in plan.items:
        remediation = item.remediation
        table.add_row(
            item.capability.capability_id,
            str(item.capability.status),
            remediation.enables if remediation else "Already ready or no remediation metadata.",
            _setup_requirements(remediation),
        )
    console.print(table)


def _setup_requirements(remediation: object | None) -> str:
    if remediation is None:
        return "none"
    network = "network" if remediation.network_required else "no network"
    privilege = remediation.privilege_required.value
    repos = (
        "repository changes"
        if remediation.changes_package_repositories
        else "no repository changes"
    )
    return f"{network}; privilege={privilege}; {repos}"


def _render_doctor_table(report: DiagnosticReport) -> None:
    table = Table(title="Hephaestus Environment Diagnostics", show_lines=False)
    table.add_column("Category", style="bold")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")
    for check in report.checks:
        label = f"{check.label} ({check.capability_id})" if check.required_by_user else check.label
        table.add_row(str(check.category), label, _format_status(str(check.status)), check.detail)
    Console(width=140).print(table)


def _format_status(status: str) -> str:
    styles = {
        DiagnosticStatus.READY: "green",
        DiagnosticStatus.MISSING: "red",
        DiagnosticStatus.NOT_CONFIGURED: "yellow",
        DiagnosticStatus.MISMATCH: "yellow",
        "active": "green",
        "installed_inactive": "yellow",
        "incompatible": "yellow",
        "degraded": "yellow",
        "unknown": "yellow",
        "failed": "red",
        "timeout": "red",
    }
    return f"[{styles.get(status, 'yellow')}]{status}[/]"


def _format_validation_severity(severity: ValidationSeverity) -> str:
    styles = {
        ValidationSeverity.INFO: "cyan",
        ValidationSeverity.WARNING: "yellow",
        ValidationSeverity.ERROR: "red",
        ValidationSeverity.FATAL: "bold red",
    }
    return f"[{styles[severity]}]{severity.value}[/]"
