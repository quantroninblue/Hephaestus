"""Child-process worker for app-owned CadQuery exports."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import io
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hephaestus.backends.cadquery_backend import _normalize_step_metadata
from hephaestus.execution import _bounded_text
from hephaestus.tools.cadquery_build import (
    MAX_PARTS,
    MM_TO_M,
    CadQueryBuildError,
    _validate_part_name,
    validate_cadquery_source,
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        return _emit(False, "expected source path and output directory")
    source_path = Path(args[0])
    output_dir = Path(args[1])
    try:
        payload = build_source(source_path, output_dir)
    except Exception as exc:  # noqa: BLE001 - child result must be contained and sanitized.
        return _emit(False, _bounded_text(str(exc) or exc.__class__.__name__))
    return _emit(True, None, payload)


def build_source(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    validate_cadquery_source(source)
    try:
        cadquery_version = importlib.metadata.version("cadquery")
    except Exception:
        cadquery_version = None
    import cadquery as cq
    from cadquery import exporters

    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(cq),
        "cq": cq,
        "math": math,
    }
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    code = compile(source, str(source_path.name), "exec")
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        exec(code, namespace, namespace)
    parts = namespace.get("parts")
    if not isinstance(parts, Mapping):
        raise CadQueryBuildError("CadQuery source must define a parts mapping")
    if not (1 <= len(parts) <= MAX_PARTS):
        raise CadQueryBuildError("parts mapping must contain between 1 and 64 parts")

    step_dir = output_dir / "step"
    stl_dir = output_dir / "stl"
    step_dir.mkdir(parents=True, exist_ok=True)
    stl_dir.mkdir(parents=True, exist_ok=True)

    part_payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_name, solid_like in parts.items():
        if not isinstance(raw_name, str):
            raise CadQueryBuildError("part names must be strings")
        _validate_part_name(raw_name)
        if raw_name in seen:
            raise CadQueryBuildError(f"duplicate part name: {raw_name}")
        seen.add(raw_name)
        shape = _shape_from_cadquery_object(solid_like)
        metrics = _metrics(shape)
        step_path = step_dir / f"{raw_name}.step"
        stl_path = stl_dir / f"{raw_name}.stl"
        exporters.export(solid_like, str(step_path), exportType="STEP")
        _normalize_step_metadata(step_path)
        exporters.export(
            solid_like,
            str(stl_path),
            exportType="STL",
            tolerance=0.001,
            angularTolerance=0.05,
        )
        part_payloads.append({"name": raw_name, **metrics})

    assembly = namespace.get("assembly")
    assembly_exported = False
    if assembly is not None:
        export = getattr(assembly, "export", None)
        if export is None or not callable(export):
            raise CadQueryBuildError("optional assembly must be a CadQuery Assembly")
        assembly_path = output_dir / "assembly.step"
        export(str(assembly_path))
        _normalize_step_metadata(assembly_path)
        assembly_exported = True

    manifest = {
        "success": True,
        "backend": "cadquery",
        "backend_version": cadquery_version,
        "parts": sorted(part_payloads, key=lambda item: item["name"]),
        "assembly_step_exported": assembly_exported,
        "discarded_source_stdout_bytes": len(captured_stdout.getvalue().encode("utf-8")),
        "discarded_source_stderr_bytes": len(captured_stderr.getvalue().encode("utf-8")),
    }
    (output_dir / ".worker_manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"backend_version": cadquery_version, "part_count": len(part_payloads)}


def _shape_from_cadquery_object(value: Any) -> Any:
    candidates = []
    val = getattr(value, "val", None)
    if callable(val):
        with contextlib.suppress(Exception):
            candidates.append(val())
    candidates.append(value)
    for candidate in candidates:
        volume = getattr(candidate, "Volume", None)
        bbox = getattr(candidate, "BoundingBox", None)
        if callable(volume) and callable(bbox):
            return candidate
    raise CadQueryBuildError("each part must be a CadQuery solid shape")


def _metrics(shape: Any) -> dict[str, Any]:
    try:
        volume_mm3 = float(shape.Volume())
        bbox = shape.BoundingBox()
        bounds_mm = (float(bbox.xlen), float(bbox.ylen), float(bbox.zlen))
    except Exception as exc:
        raise CadQueryBuildError("part geometry metrics could not be computed") from exc
    if not math.isfinite(volume_mm3) or volume_mm3 <= 0.0:
        raise CadQueryBuildError("part volume must be finite and positive")
    if any(not math.isfinite(item) or item <= 0.0 for item in bounds_mm):
        raise CadQueryBuildError("part bounds must be finite and positive")
    center_m = None
    center = getattr(shape, "Center", None)
    if callable(center):
        with contextlib.suppress(Exception):
            point = center()
            center_m = [
                _finite_coord(float(point.x) * MM_TO_M),
                _finite_coord(float(point.y) * MM_TO_M),
                _finite_coord(float(point.z) * MM_TO_M),
            ]
    return {
        "bounds_m": [round(item * MM_TO_M, 9) for item in bounds_mm],
        "volume_m3": volume_mm3 * (MM_TO_M**3),
        "center_of_mass_m": center_m,
    }


def _finite_coord(value: float) -> float:
    if not math.isfinite(value):
        raise CadQueryBuildError("center of mass contains non-finite values")
    return value


def _safe_builtins(cq: Any) -> dict[str, Any]:
    allowed = {
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    def restricted_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | None = (),
        level: int = 0,
    ) -> Any:
        del globals_, locals_
        if level != 0:
            raise ImportError("relative imports are not allowed")
        root = name.split(".", 1)[0]
        if root == "cadquery":
            requested = fromlist or ()
            if any(item in {"exporters", "importers"} for item in requested):
                raise ImportError("CadQuery importer/exporter access is not allowed")
            return cq
        if root == "math":
            return math
        raise ImportError(f"import is not allowed: {name}")

    allowed["__import__"] = restricted_import
    return allowed


def _emit(success: bool, error: str | None, payload: dict[str, Any] | None = None) -> int:
    result: dict[str, Any] = {"success": success}
    if error:
        result["error"] = error
    if payload:
        result.update(payload)
    sys.stdout.write(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
