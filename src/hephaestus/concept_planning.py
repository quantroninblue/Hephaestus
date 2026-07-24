"""Provider-independent structured planning for concept robots."""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hephaestus.ai.types import (
    ModelProvider,
    ProviderMalformedResponseError,
    ProviderStructuredGenerationError,
    StructuredPlanningRequest,
)
from hephaestus.concept_robot import (
    ConceptError,
    ConceptRobotSpec,
    concept_robot_schema,
    parse_concept_robot_spec,
)

MAX_REPAIR_ATTEMPTS = 2
MAX_STRICT_JSON_VALIDATE_RETRIES = 1
MAX_BEST_EFFORT_STRUCTURED_ATTEMPTS = 1
MAX_PROVIDER_ATTEMPTS = (
    1 + MAX_REPAIR_ATTEMPTS + MAX_STRICT_JSON_VALIDATE_RETRIES + MAX_BEST_EFFORT_STRUCTURED_ATTEMPTS
)
CONCEPT_SCHEMA_NAME = "hephaestus_concept_robot_v1"
_MAX_PROVIDER_SCHEMA_BYTES = 65536
_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JSON_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "allOf",
    "anyOf",
    "dependentSchemas",
    "discriminator",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "then",
    "unevaluatedProperties",
}


class ConceptPlanningError(RuntimeError):
    """Raised when structured planning cannot produce a valid concept spec."""


@dataclass(frozen=True, slots=True)
class ConceptPlanningResult:
    spec: ConceptRobotSpec
    attempts: int
    repaired: bool
    strict_json_validate_retries: int = 0
    best_effort_fallback_used: bool = False
    provider_attempt_modes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hephaestus.concept.planning_result.v1",
            "attempts": self.attempts,
            "repaired": self.repaired,
            "strict_json_validate_retries": self.strict_json_validate_retries,
            "best_effort_fallback_used": self.best_effort_fallback_used,
            "provider_attempt_modes": list(self.provider_attempt_modes),
            "spec": self.spec.to_payload(),
        }


class ConceptPlanner:
    """Ask a provider for a full strict ConceptRobotSpec and validate it locally."""

    def __init__(self, provider: ModelProvider, *, max_completion_tokens: int | None = None) -> None:
        self._provider = provider
        self._max_completion_tokens = max_completion_tokens

    def create(self, description: str) -> ConceptPlanningResult:
        return self._plan(_create_prompt(description), repair_context=None)

    def edit(self, current: ConceptRobotSpec, change_request: str) -> ConceptPlanningResult:
        return self._plan(_edit_prompt(current, change_request), repair_context=current)

    def _plan(
        self, prompt: str, *, repair_context: ConceptRobotSpec | None
    ) -> ConceptPlanningResult:
        schema = concept_robot_schema()
        schema_errors = audit_structured_json_schema(CONCEPT_SCHEMA_NAME, schema)
        if schema_errors:
            joined = "; ".join(schema_errors)
            raise ConceptPlanningError(
                f"concept planning schema is not provider-compatible: {joined}"
            )

        active_prompt = prompt
        last_error = "provider did not return a valid concept robot spec"
        last_payload: dict[str, Any] | None = None
        provider_attempts = 0
        semantic_repairs = 0
        strict_json_validate_retries = 0
        best_effort_fallback_used = False
        attempt_modes: list[str] = []

        while provider_attempts < MAX_PROVIDER_ATTEMPTS:
            strict_schema = not best_effort_fallback_used
            attempt_modes.append("strict" if strict_schema else "best_effort")
            provider_attempts += 1
            try:
                response = self._provider.complete_structured(
                    StructuredPlanningRequest(
                        system_instruction=CONCEPT_SYSTEM_INSTRUCTION,
                        user_prompt=active_prompt,
                        schema_name=CONCEPT_SCHEMA_NAME,
                        json_schema=schema,
                        strict_schema=strict_schema,
                        max_output_tokens=self._max_completion_tokens,
                    )
                )
            except ProviderStructuredGenerationError as exc:
                if _is_json_validate_failed(exc) and strict_schema:
                    last_error = "provider could not produce schema-valid structured output"
                    if strict_json_validate_retries < MAX_STRICT_JSON_VALIDATE_RETRIES:
                        strict_json_validate_retries += 1
                        active_prompt = _json_validate_retry_prompt(active_prompt)
                        continue
                    if not best_effort_fallback_used and MAX_BEST_EFFORT_STRUCTURED_ATTEMPTS:
                        best_effort_fallback_used = True
                        active_prompt = _best_effort_prompt(active_prompt)
                        continue
                raise ConceptPlanningError(
                    f"structured concept planning failed: {last_error}"
                ) from exc
            except ProviderMalformedResponseError as exc:
                last_error = str(exc)
                break

            try:
                payload = dict(response.payload)
                spec = parse_concept_robot_spec(payload)
            except ConceptError as exc:
                last_payload = dict(response.payload)
                last_error = str(exc)
                if semantic_repairs >= MAX_REPAIR_ATTEMPTS:
                    break
                if provider_attempts >= MAX_PROVIDER_ATTEMPTS:
                    break
                semantic_repairs += 1
                active_prompt = _repair_prompt(
                    original_prompt=active_prompt,
                    invalid_payload=last_payload,
                    validation_error=last_error,
                    current=repair_context,
                )
                continue
            return ConceptPlanningResult(
                spec=spec,
                attempts=provider_attempts,
                repaired=semantic_repairs > 0,
                strict_json_validate_retries=strict_json_validate_retries,
                best_effort_fallback_used=best_effort_fallback_used,
                provider_attempt_modes=tuple(attempt_modes),
            )
        raise ConceptPlanningError(f"structured concept planning failed: {last_error}")


CONCEPT_SYSTEM_INSTRUCTION = """You convert bounded wheeled robot descriptions into a complete JSON ConceptRobotSpec.
Return JSON only. Do not include code, shell commands, paths, URDF, SDF, launch files, controller YAML, ROS commands, canonical IDs, explanations outside JSON, or extra keys.
Emit every required key exactly as declared by the schema. Use SI units. Use empty arrays for no joints, sensors, assumptions, or engineering_warnings. Use numeric 0.0 for primitive dimensions that do not apply to a shape, and include full pose, color, axis, and joint-limit objects every time.
Use concise names, assumptions, and warnings. Preserve awkward requested geometry when representable and report engineering concerns as warnings.
Supported locomotion: differential_drive, skid_steer, static. Supported primitive geometry: box, cylinder, sphere. Supported joints: fixed, continuous, revolute. Supported sensors: lidar_2d, imu. Supported worlds: engineering_lab, warehouse, ramp_course, rough_terrain.
Only left_drive and right_drive wheel joints are actively controller-supported in this release.
"""


def audit_structured_json_schema(schema_name: str, schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Return provider strict-schema compatibility errors for a JSON Schema."""

    errors: list[str] = []
    if not _SCHEMA_NAME_PATTERN.fullmatch(schema_name):
        errors.append("schema name is not provider-safe")
    try:
        encoded = json.dumps(schema, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        errors.append(f"$: schema is not JSON-compatible: {exc}")
        encoded = ""
    if len(encoded.encode("utf-8")) > _MAX_PROVIDER_SCHEMA_BYTES:
        errors.append(f"$: schema exceeds {_MAX_PROVIDER_SCHEMA_BYTES} bytes")
    if not isinstance(schema, Mapping):
        return tuple((*errors, "$: schema must be an object"))
    if not _schema_includes_type(schema, "object"):
        errors.append("$: root schema must be type object")
    _audit_schema_node(schema, "$", schema, errors, ())
    return tuple(errors)


def _audit_schema_node(
    node: Any,
    path: str,
    root: Mapping[str, Any],
    errors: list[str],
    resolving: tuple[str, ...],
) -> None:
    if not isinstance(node, Mapping):
        errors.append(f"{path}: schema node must be an object")
        return
    for keyword in sorted(_UNSUPPORTED_SCHEMA_KEYWORDS.intersection(node)):
        errors.append(f"{path}: unsupported keyword {keyword}")
    raw_type = node.get("type")
    if raw_type is not None:
        for item in _schema_types(raw_type, path, errors):
            if item not in _JSON_SCHEMA_TYPES:
                errors.append(f"{path}: unsupported type {item}")
    if "$ref" in node:
        ref = node.get("$ref")
        target_path, target = _resolve_ref(ref, root)
        if target is None or target_path is None:
            errors.append(f"{path}: unresolved $ref {ref!r}")
        elif target_path in resolving:
            errors.append(f"{path}: recursive $ref {ref!r}")
        else:
            _audit_schema_node(target, target_path, root, errors, (*resolving, target_path))
        return
    defs = node.get("$defs")
    if defs is not None:
        if not isinstance(defs, Mapping):
            errors.append(f"{path}.$defs: must be an object")
        else:
            for name, child in defs.items():
                _audit_schema_node(child, f"$defs.{name}", root, errors, resolving)
    if _schema_includes_type(node, "object"):
        _audit_object_schema(node, path, root, errors, resolving)
    if _schema_includes_type(node, "array"):
        _audit_array_schema(node, path, root, errors, resolving)


def _audit_object_schema(
    node: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    errors: list[str],
    resolving: tuple[str, ...],
) -> None:
    if node.get("additionalProperties") is not False:
        errors.append(f"{path}: missing additionalProperties=false")
    properties = node.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        errors.append(f"{path}.properties: must be an object")
        return
    required = node.get("required")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        errors.append(f"{path}.required: must be a list of property names")
        required_names: set[str] = set()
    else:
        required_names = set(required)
    property_names = set(properties)
    for name in sorted(property_names - required_names):
        errors.append(f"{path}.properties.{name}: property is not required")
    for name in sorted(required_names - property_names):
        errors.append(f"{path}.required.{name}: required name is absent from properties")
    for name, child in properties.items():
        _audit_schema_node(child, f"{path}.properties.{name}", root, errors, resolving)


def _audit_array_schema(
    node: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    errors: list[str],
    resolving: tuple[str, ...],
) -> None:
    items = node.get("items")
    if not isinstance(items, Mapping):
        errors.append(f"{path}: array schema must define object items")
        return
    max_items = node.get("maxItems")
    if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 0:
        errors.append(f"{path}: array schema must define bounded maxItems")
    _audit_schema_node(items, f"{path}.items", root, errors, resolving)


def _schema_includes_type(node: Mapping[str, Any], expected: str) -> bool:
    raw = node.get("type")
    if isinstance(raw, str):
        return raw == expected
    if isinstance(raw, list):
        return expected in raw
    return False


def _schema_types(raw_type: Any, path: str, errors: list[str]) -> tuple[str, ...]:
    if isinstance(raw_type, str):
        return (raw_type,)
    if isinstance(raw_type, list) and all(isinstance(item, str) for item in raw_type):
        if len(raw_type) != len(set(raw_type)):
            errors.append(f"{path}.type: contains duplicate types")
        return tuple(raw_type)
    errors.append(f"{path}.type: must be a string or list of strings")
    return ()


def _resolve_ref(ref: Any, root: Mapping[str, Any]) -> tuple[str | None, Any | None]:
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return None, None
    current: Any = root.get("$defs")
    path_parts = ["$defs"]
    for raw_part in ref.removeprefix("#/$defs/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            return None, None
        current = current[part]
        path_parts.append(part)
    return ".".join(path_parts), current


def _is_json_validate_failed(exc: ProviderStructuredGenerationError) -> bool:
    details = exc.details
    return bool(
        details and details.status_code == 400 and details.provider_code == "json_validate_failed"
    )


def _create_prompt(description: str) -> str:
    return (
        "Create one full ConceptRobotSpec for this bounded concept robot request. "
        "Use semantic keys, not canonical IDs. Include assumptions and engineering_warnings. "
        "Every schema property is required; emit empty arrays and neutral numeric fields instead of omitting keys.\n\n"
        f"REQUEST:\n{description.strip()}"
    )


def _edit_prompt(current: ConceptRobotSpec, change_request: str) -> str:
    current_json = json.dumps(current.to_payload(), allow_nan=False, indent=2, sort_keys=True)
    return (
        "Revise the current ConceptRobotSpec according to the change request. "
        "Return one complete revised ConceptRobotSpec, not a patch. Preserve unchanged semantic keys. "
        "Every schema property is required; emit empty arrays and neutral numeric fields instead of omitting keys.\n\n"
        f"CURRENT_SPEC:\n{current_json}\n\nCHANGE_REQUEST:\n{change_request.strip()}"
    )


def _json_validate_retry_prompt(original_prompt: str) -> str:
    return (
        "The previous structured response could not be validated by the provider against the strict JSON Schema. "
        "Retry once with the same schema. Emit every required key, including empty arrays, 0.0 non-applicable dimensions, full pose angles, color alpha, joint axes, and joint limits. Return JSON only.\n\n"
        f"ORIGINAL_REQUEST:\n{original_prompt}"
    )


def _best_effort_prompt(original_prompt: str) -> str:
    return (
        "Return one complete ConceptRobotSpec JSON object. The provider schema is best-effort, but Hephaestus will reject malformed JSON, extra keys, missing required keys, invalid references, unsafe physics, or unsupported values. Return JSON only.\n\n"
        f"ORIGINAL_REQUEST:\n{original_prompt}"
    )


def _repair_prompt(
    *,
    original_prompt: str,
    invalid_payload: dict[str, Any] | None,
    validation_error: str,
    current: ConceptRobotSpec | None,
) -> str:
    payload_json = json.dumps(invalid_payload or {}, allow_nan=False, indent=2, sort_keys=True)
    current_json = ""
    if current is not None:
        current_json = "\nCURRENT_SPEC_TO_PRESERVE_WHERE_POSSIBLE:\n" + json.dumps(
            current.to_payload(), allow_nan=False, indent=2, sort_keys=True
        )
    return (
        "Repair the previous full ConceptRobotSpec so it satisfies the schema and local semantic validation. "
        "Return one complete corrected ConceptRobotSpec JSON object only. Emit every required key.\n\n"
        f"VALIDATION_ERROR:\n{validation_error}\n\nPREVIOUS_INVALID_PAYLOAD:\n{payload_json}{current_json}\n\nORIGINAL_REQUEST:\n{original_prompt}"
    )
