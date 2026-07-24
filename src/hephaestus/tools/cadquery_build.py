"""Restricted CadQuery build tool for the default agent workspace."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hephaestus.execution import _bounded_text, _sanitized_env, _terminate_process_group

CADQUERY_PERMISSION_BUILD = "cadquery build"
CADQUERY_BUILD_TIMEOUT_SECONDS = 60.0
MAX_CADQUERY_SOURCE_BYTES = 64_000
MAX_AST_NODES = 5_000
MAX_AST_DEPTH = 80
MAX_PARTS = 64
MAX_PART_NAME_LENGTH = 64
MAX_ARTIFACT_BYTES = 50_000_000
MAX_TOTAL_ARTIFACT_BYTES = 250_000_000
MM_TO_M = 0.001
MESH_SCALE = "0.001 0.001 0.001"

_PART_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_ALLOWED_MODULES = frozenset({"cadquery", "math"})
_ALLOWED_MATH_NAMES = frozenset(
    {
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "degrees",
        "e",
        "exp",
        "fabs",
        "floor",
        "hypot",
        "isfinite",
        "log",
        "pi",
        "pow",
        "radians",
        "sin",
        "sqrt",
        "tan",
        "tau",
    }
)
_FORBIDDEN_MODULES = frozenset(
    {
        "ctypes",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "threading",
        "urllib",
    }
)
_FORBIDDEN_BUILTINS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)
_FORBIDDEN_ATTR_NAMES = frozenset(
    {
        "export",
        "exporters",
        "getenv",
        "home",
        "importers",
        "read",
        "read_text",
        "save",
        "write",
        "write_text",
    }
)
_FORBIDDEN_ASSIGNMENT_NAMES = frozenset(
    {
        "filename",
        "filepath",
        "out",
        "out_dir",
        "output",
        "output_dir",
        "output_path",
        "path",
        "target",
    }
)


class CadQueryBuildError(RuntimeError):
    """Raised when generated CadQuery source is rejected or cannot be exported."""


class CadQueryBuildArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=300)


class _RunProcess(Protocol):
    def __call__(
        self, argv: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class CadQueryBuildContext:
    workspace_root: Path
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    run_process: _RunProcess = field(default_factory=lambda: _run_cadquery_process)
    timeout_seconds: float = CADQUERY_BUILD_TIMEOUT_SECONDS
    python_executable: str = sys.executable

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", self.workspace_root.resolve(strict=True))

    def build(
        self,
        source_path: Path,
        relative_path: str,
        *,
        confirm: Callable[[dict[str, Any]], bool] | None,
    ) -> dict[str, Any]:
        _check_source_path(source_path)
        source_bytes = source_path.read_bytes()
        _check_source_bytes(source_bytes)
        source_fingerprint = _sha256_bytes(source_bytes)
        source_text = _decode_source(source_bytes)
        validate_cadquery_source(source_text)
        final_dir = _output_dir(self.workspace_root, source_path)
        preview = {
            "status": "preview",
            "tool": "cadquery_build",
            "permission": CADQUERY_PERMISSION_BUILD,
            "result": {
                "relative_source_path": relative_path,
                "source_fingerprint": source_fingerprint,
                "generated_output_directory": _relative_to_workspace(
                    self.workspace_root, final_dir
                ),
                "warning": (
                    "model-generated CAD source will be executed through the restricted "
                    "CadQuery runner; this is not a complete OS security sandbox"
                ),
            },
        }
        if confirm is None or not confirm(preview):
            return {
                "success": False,
                "committed": False,
                "relative_source_path": relative_path,
                "source_fingerprint": source_fingerprint,
                "message": "cadquery build was previewed but not approved",
            }
        if _sha256_bytes(source_path.read_bytes()) != source_fingerprint:
            raise CadQueryBuildError("source file changed after CadQuery build preview")

        parent = final_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", suffix=".tmp", dir=parent))
        backup_dir: Path | None = None
        try:
            worker_result = self._run_worker(source_path, temp_dir)
            manifest = _validate_build_directory(temp_dir, source_path, relative_path)
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _check_artifact_size_bounds(temp_dir)
            manifest = _validate_build_directory(temp_dir, source_path, relative_path)
            manifest_path.write_text(
                json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            backup_dir = _promote_directory(temp_dir, final_dir)
            temp_dir = final_dir
            if backup_dir is not None:
                shutil.rmtree(backup_dir)
            manifest = _validate_build_directory(final_dir, source_path, relative_path)
            (final_dir / "manifest.json").write_text(
                json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (final_dir / ".worker_manifest.json").unlink(missing_ok=True)
            return _result_from_manifest(manifest, worker_result)
        except CadQueryBuildError:
            raise
        except Exception as exc:
            raise CadQueryBuildError("CadQuery build failed") from exc
        finally:
            if temp_dir != final_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if backup_dir is not None and backup_dir.exists():
                if final_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                else:
                    with suppress(OSError):
                        os.replace(backup_dir, final_dir)

    def _run_worker(self, source_path: Path, output_dir: Path) -> dict[str, Any]:
        argv = (
            self.python_executable,
            "-m",
            "hephaestus.tools.cadquery_worker",
            str(source_path),
            str(output_dir),
        )
        env = _cadquery_runner_env(self.env)
        try:
            completed = self.run_process(
                argv,
                cwd=str(output_dir),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=env,
                start_new_session=True,
                preexec_fn=_resource_limiter if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stderr or exc.stdout or "") if isinstance(exc.stderr, str) else ""
            if output:
                raise CadQueryBuildError(
                    f"CadQuery build timed out: {_bounded_text(output)}"
                ) from exc
            raise CadQueryBuildError("CadQuery build timed out") from exc
        except OSError as exc:
            raise CadQueryBuildError(
                f"CadQuery runner failed to start: {_bounded_text(str(exc))}"
            ) from exc
        if completed.returncode != 0:
            raise CadQueryBuildError(
                _cadquery_runner_failure_message(
                    completed.returncode,
                    completed.stderr,
                    completed.stdout,
                    protected_paths=(source_path, output_dir),
                )
            )
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise CadQueryBuildError("CadQuery runner returned malformed JSON") from exc
        if not isinstance(payload, dict) or payload.get("success") is not True:
            message = str(payload.get("error") if isinstance(payload, dict) else "unknown error")
            raise CadQueryBuildError(f"CadQuery runner failed: {_bounded_text(message)}")
        return payload


def cadquery_tool_definition() -> dict[str, Any]:
    return {
        "name": "cadquery_build",
        "description": (
            "Build one existing workspace .py CadQuery source file after confirmation. "
            "`parts` must map 1-64 safe unique names to non-empty CadQuery solids in "
            "millimetres. Hephaestus executes with a restricted CadQuery runner, exports "
            "per-part STEP/STL artifacts under generated/cad/<source-stem>/, reports SI "
            "bounds/volumes/centers and hashes, and recommends SDF mesh scale "
            "0.001 0.001 0.001. Use this before writing SDF mesh visuals."
        ),
        "permission": CADQUERY_PERMISSION_BUILD,
        "args_model": CadQueryBuildArgs,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": (
                        "Relative existing UTF-8 .py CadQuery source file beneath the "
                        "startup directory. Dimensions are millimetres."
                    ),
                }
            },
            "required": ["relative_path"],
        },
    }


def validate_cadquery_source(source: str) -> None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise CadQueryBuildError(f"invalid CadQuery syntax: {exc.msg}") from exc
    node_count = 0
    max_depth = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        max_depth = max(max_depth, depth)
        if node_count > MAX_AST_NODES:
            raise CadQueryBuildError("CadQuery source exceeds AST complexity limit")
        if depth > MAX_AST_DEPTH:
            raise CadQueryBuildError("CadQuery source exceeds AST depth limit")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    _CadQuerySourceValidator().visit(tree)


class _CadQuerySourceValidator(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in _FORBIDDEN_MODULES or root not in _ALLOWED_MODULES:
                raise CadQueryBuildError(f"import is not allowed: {alias.name}")
            if alias.asname and _has_dunder(alias.asname):
                raise CadQueryBuildError("dunder names are not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if node.level != 0 or root in _FORBIDDEN_MODULES or root not in _ALLOWED_MODULES:
            raise CadQueryBuildError(f"import is not allowed: {module}")
        if root == "math":
            for alias in node.names:
                if alias.name not in _ALLOWED_MATH_NAMES:
                    raise CadQueryBuildError(f"math import is not allowed: {alias.name}")
        if root == "cadquery":
            for alias in node.names:
                if alias.name in {"exporters", "importers"}:
                    raise CadQueryBuildError(
                        "direct CadQuery importer/exporter access is forbidden"
                    )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if _has_dunder(node.id):
            raise CadQueryBuildError("dunder names are not allowed")
        if node.id in _FORBIDDEN_BUILTINS:
            raise CadQueryBuildError(f"builtin is not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _has_dunder(node.attr):
            raise CadQueryBuildError("dunder attributes are not allowed")
        if node.attr in _FORBIDDEN_ATTR_NAMES:
            raise CadQueryBuildError(f"attribute is not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTINS:
            raise CadQueryBuildError(f"builtin is not allowed: {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"export", "save"}:
            raise CadQueryBuildError("direct CadQuery export calls are forbidden")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            _check_assignment_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        _check_assignment_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        _check_assignment_target(node.target)
        self.generic_visit(node)


def _check_assignment_target(target: ast.AST) -> None:
    if isinstance(target, ast.Name):
        name = target.id.lower()
        if _has_dunder(target.id):
            raise CadQueryBuildError("dunder names are not allowed")
        if name in _FORBIDDEN_ASSIGNMENT_NAMES or "path" in name or "file" in name:
            raise CadQueryBuildError("generated source may not choose output paths")
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _check_assignment_target(item)


def _has_dunder(value: str) -> bool:
    return "__" in value


def _check_source_path(path: Path) -> None:
    if path.suffix.lower() != ".py":
        raise CadQueryBuildError("cadquery_build accepts only .py source files")
    if not path.is_file():
        raise CadQueryBuildError("CadQuery source path is not a regular file")


def _check_source_bytes(source: bytes) -> None:
    if len(source) > MAX_CADQUERY_SOURCE_BYTES:
        raise CadQueryBuildError("CadQuery source exceeds size limit")


def _decode_source(source: bytes) -> str:
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CadQueryBuildError("CadQuery source is not valid UTF-8") from exc


def _output_dir(workspace_root: Path, source_path: Path) -> Path:
    return workspace_root / "generated" / "cad" / source_path.stem


def _promote_directory(temp_dir: Path, final_dir: Path) -> Path | None:
    backup_dir: Path | None = None
    if final_dir.exists():
        backup_dir = final_dir.with_name(f".{final_dir.name}.previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        os.replace(final_dir, backup_dir)
    try:
        os.replace(temp_dir, final_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    return backup_dir


def _validate_build_directory(
    output_dir: Path, source_path: Path, relative_source_path: str
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    worker_manifest_path = output_dir / ".worker_manifest.json"
    try:
        worker_manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CadQueryBuildError("CadQuery worker manifest is missing or invalid") from exc
    parts = worker_manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise CadQueryBuildError("CadQuery worker returned no parts")
    source_fingerprint = _sha256_file(source_path)
    manifest_parts: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            raise CadQueryBuildError("CadQuery worker returned malformed part metadata")
        name = str(part.get("name", ""))
        _validate_part_name(name)
        step_path = output_dir / "step" / f"{name}.step"
        stl_path = output_dir / "stl" / f"{name}.stl"
        for artifact_path in (step_path, stl_path):
            _check_generated_artifact(output_dir, artifact_path)
        step_rel = _relative_to_workspace(output_dir.parents[2], step_path)
        stl_rel = _relative_to_workspace(output_dir.parents[2], stl_path)
        manifest_parts.append(
            {
                "name": name,
                "step_path": step_rel,
                "stl_path": stl_rel,
                "bounds_m": _finite_positive_list(part.get("bounds_m"), "bounds_m"),
                "volume_m3": _finite_positive_float(part.get("volume_m3"), "volume_m3"),
                "center_of_mass_m": _finite_list_or_none(
                    part.get("center_of_mass_m"), "center_of_mass_m"
                ),
                "hashes": {
                    "step_sha256": _sha256_file(step_path),
                    "stl_sha256": _sha256_file(stl_path),
                },
                "bytes": {
                    "step": step_path.stat().st_size,
                    "stl": stl_path.stat().st_size,
                },
            }
        )
    assembly = None
    assembly_path = output_dir / "assembly.step"
    if assembly_path.exists():
        _check_generated_artifact(output_dir, assembly_path)
        assembly = {
            "path": _relative_to_workspace(output_dir.parents[2], assembly_path),
            "sha256": _sha256_file(assembly_path),
            "bytes": assembly_path.stat().st_size,
        }
    return {
        "schema": "hephaestus.agent_cadquery_manifest.v1",
        "success": True,
        "source_path": relative_source_path,
        "source_fingerprint": source_fingerprint,
        "output_directory": _relative_to_workspace(output_dir.parents[2], output_dir),
        "manifest_path": _relative_to_workspace(output_dir.parents[2], manifest_path),
        "unit_convention": {
            "cadquery_source_dimensions": "millimetres",
            "reported_bounds": "metres",
            "reported_volume": "cubic metres",
            "reported_center_of_mass": "metres",
            "gazebo_stl_mesh_scale": MESH_SCALE,
        },
        "parts": sorted(manifest_parts, key=lambda item: str(item["name"])),
        "assembly_step": assembly,
        "runner_limitations": (
            "AST validation and filtered child execution reduce accidental and obvious unsafe "
            "behavior but are not a complete OS security sandbox"
        ),
    }


def _result_from_manifest(
    manifest: dict[str, Any], worker_result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "success": True,
        "committed": True,
        "source_path": manifest["source_path"],
        "source_fingerprint": manifest["source_fingerprint"],
        "manifest_path": manifest["manifest_path"],
        "output_directory": manifest["output_directory"],
        "generated_part_names": [part["name"] for part in manifest["parts"]],
        "parts": manifest["parts"],
        "assembly_step": manifest["assembly_step"],
        "recommended_sdf_mesh_scale": MESH_SCALE,
        "unit_convention": manifest["unit_convention"],
        "backend_version": worker_result.get("backend_version"),
        "runner_limitations": manifest["runner_limitations"],
    }


def _validate_part_name(name: str) -> None:
    if not _PART_NAME_PATTERN.fullmatch(name):
        raise CadQueryBuildError(f"invalid CadQuery part name: {name}")


def _finite_positive_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CadQueryBuildError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise CadQueryBuildError(f"{label} must be finite and positive")
    return number


def _finite_positive_list(value: object, label: str) -> list[float]:
    values = _finite_list(value, label)
    if any(item <= 0.0 for item in values):
        raise CadQueryBuildError(f"{label} must contain positive values")
    return values


def _finite_list_or_none(value: object, label: str) -> list[float] | None:
    if value is None:
        return None
    return _finite_list(value, label)


def _finite_list(value: object, label: str) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise CadQueryBuildError(f"{label} must contain three values")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise CadQueryBuildError(f"{label} contains non-finite values")
    return result


def _check_generated_artifact(output_dir: Path, path: Path) -> None:
    resolved = path.resolve(strict=True)
    resolved.relative_to(output_dir.resolve(strict=True))
    if not resolved.is_file():
        raise CadQueryBuildError("generated artifact is not a file")
    size = resolved.stat().st_size
    if size <= 0:
        raise CadQueryBuildError("generated artifact is empty")
    if size > MAX_ARTIFACT_BYTES:
        raise CadQueryBuildError("generated artifact exceeds size limit")


def _check_artifact_size_bounds(output_dir: Path) -> None:
    total = 0
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        total += path.stat().st_size
        if total > MAX_TOTAL_ARTIFACT_BYTES:
            raise CadQueryBuildError("generated artifacts exceed total size limit")


def _relative_to_workspace(workspace_root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(workspace_root.resolve(strict=True)).as_posix()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cadquery_runner_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    env = _sanitized_env(parent_env)
    for key in tuple(env):
        if key not in {"HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "TERM"}:
            env.pop(key, None)
    return env


def _run_cadquery_process(
    argv: Sequence[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    timeout = kwargs.pop("timeout", None)
    check = kwargs.pop("check", None)
    capture_output = kwargs.pop("capture_output", None)
    if check not in {None, False}:
        raise ValueError("cadquery runner does not support check=True")
    if capture_output not in {None, True}:
        raise ValueError("cadquery runner requires captured output")
    process = subprocess.Popen(
        tuple(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    try:
        stdout, stderr = process.communicate(
            timeout=timeout if isinstance(timeout, int | float) else None
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = "", "process did not exit after SIGKILL"
        raise subprocess.TimeoutExpired(tuple(argv), timeout, output=stdout, stderr=stderr) from exc
    return subprocess.CompletedProcess(tuple(argv), int(process.returncode or 0), stdout, stderr)


def _cadquery_runner_failure_message(
    returncode: int,
    stderr: str | None,
    stdout: str | None,
    *,
    protected_paths: Sequence[Path] = (),
) -> str:
    if returncode < 0:
        reason = _signal_exit_description(-returncode)
    else:
        reason = f"process exited with code {returncode}"
    output = _sanitize_cadquery_runner_output(stderr or stdout or "", protected_paths)
    if output:
        return f"CadQuery runner failed: {reason}; output: {output}"
    return f"CadQuery runner failed: {reason}; no diagnostic output was captured"


def _signal_exit_description(signum: int) -> str:
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = f"signal {signum}"
    return f"process terminated by signal {signal_name} ({signum})"


def _sanitize_cadquery_runner_output(text: str, protected_paths: Sequence[Path]) -> str:
    bounded = _bounded_text(text)
    for path in protected_paths:
        with suppress(OSError):
            bounded = bounded.replace(str(path.resolve(strict=False)), "[path]")
        bounded = bounded.replace(str(path), "[path]")
    return bounded


def _resource_limiter() -> None:
    try:
        import resource

        cpu_seconds = max(1, int(CADQUERY_BUILD_TIMEOUT_SECONDS))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
        files = 256
        resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))
    except Exception:
        return
