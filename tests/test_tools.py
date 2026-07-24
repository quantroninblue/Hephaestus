from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hephaestus.ai.types import ToolCall
from hephaestus.changes import (
    AddEntityOperation,
    ChangeSet,
    OperationKind,
    canonical_changeset_json,
    fingerprint_project,
)
from hephaestus.fixtures import representative_project
from hephaestus.model import Entity, ProjectMetadata, Provenance, ProvenanceSource
from hephaestus.persistence import load_project, save_project
from hephaestus.tools.project import default_project_tool_registry
from hephaestus.tools.registry import (
    EmptyArgs,
    PermissionClass,
    ProjectToolExecutor,
    RegisteredTool,
    ToolContext,
    ToolRegistry,
    mobile_robot_design_tool_policy,
)


def _changeset() -> dict:
    cs = ChangeSet(
        "change:tool_add",
        0,
        "Add tool entity.",
        Provenance(ProvenanceSource.USER, "test"),
        (AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:tool_added", "Tool added")),),
    )
    import json

    return json.loads(canonical_changeset_json(cs))


def _add_requirement_args() -> dict:
    return {
        "statement": "The robot must carry a payload of at least 25 kg.",
        "quantity_value": 25.0,
        "quantity_unit": "kg",
        "constraint": "at_least",
        "kind": "payload",
        "priority": "high",
        "verification_method": "test",
    }


def _mobile_robot_args() -> dict:
    return {
        "robot_family": "differential_drive",
        "robot_name": "Aisle Scout",
        "mission_summary": "inspect pallet aisles for blocked exits",
        "payload_kg": 14.0,
        "max_speed_mps": 1.3,
        "include_lidar": True,
        "include_imu": True,
        "indoor_warehouse": True,
        "explicitly_supersedes_payload": True,
    }


def test_registry_rejects_duplicates_and_strict_args(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = RegisteredTool("x.test", "test", PermissionClass.READ_ONLY, EmptyArgs, lambda a, c: {})
    registry.register(tool)
    try:
        registry.register(tool)
    except ValueError as exc:
        assert "already registered" in str(exc)
    result = registry.execute(ToolCall("1", "x.test", {"extra": True}), ToolContext(tmp_path))
    assert result["status"] == "error"


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _schema_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_schema_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_schema_depth(item) for item in value), default=0)
    return 0


def test_model_facing_tools_hide_raw_changesets_and_use_simple_mutation_schema() -> None:
    executor = ProjectToolExecutor(default_project_tool_registry(), ToolContext())
    definitions = executor.definitions(allow_mutation=True)
    names = {definition.name for definition in definitions}

    assert "project_add_requirement" in names
    assert "project_create_mobile_robot" in names
    assert "project.preview_changeset" not in names
    assert "project.apply_changeset" not in names

    banned = {"$ref", "$defs", "oneOf", "anyOf", "allOf", "discriminator"}
    for definition in definitions:
        for key in banned:
            assert not _contains_key(definition.input_schema, key), definition.name

    schema = next(
        definition.input_schema
        for definition in definitions
        if definition.name == "project_add_requirement"
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "statement",
        "quantity_value",
        "quantity_unit",
        "constraint",
    ]
    assert set(schema["properties"]["constraint"]["enum"]) == {
        "at_least",
        "at_most",
        "exactly",
    }
    assert _schema_depth(schema) <= 5

    assert "project_design_warehouse_inspection_robot" not in names
    design_schema = next(
        definition.input_schema
        for definition in definitions
        if definition.name == "project_create_mobile_robot"
    )
    assert design_schema["type"] == "object"
    assert design_schema["additionalProperties"] is False
    assert design_schema["required"] == ["robot_name", "mission_summary"]
    assert set(design_schema["properties"]["robot_family"]["enum"]) == {
        "differential_drive",
        "tracked",
        "legged",
        "manipulator",
    }
    assert _schema_depth(design_schema) <= 5


def test_model_facing_definitions_can_be_narrowed_to_expected_tool() -> None:
    executor = ProjectToolExecutor(default_project_tool_registry(), ToolContext())

    definitions = executor.definitions(
        allow_mutation=True, tool_names={"project_create_mobile_robot"}
    )

    assert [definition.name for definition in definitions] == ["project_create_mobile_robot"]




def test_design_workflow_provider_tool_set_is_narrow_and_safe() -> None:
    executor = ProjectToolExecutor(
        default_project_tool_registry(),
        ToolContext(),
        workflow_policy=mobile_robot_design_tool_policy(),
    )

    definitions = executor.definitions(allow_mutation=True)
    names = [definition.name for definition in definitions]

    assert names == [
        "project.get_summary",
        "project.validate",
        "project_create_mobile_robot",
    ]
    assert "project_add_requirement" not in names
    assert "project.preview_changeset" not in names
    assert "project.apply_changeset" not in names
    assert all("generate" not in name for name in names)
    assert all("build" not in name for name in names)
    assert all("simulate" not in name for name in names)
    assert all(name.replace(".", "_").replace("_", "").isalnum() for name in names)

    banned = {"$ref", "$defs", "oneOf", "anyOf", "allOf", "discriminator"}
    for definition in definitions:
        schema = definition.input_schema
        assert schema.get("type") == "object"
        assert schema.get("additionalProperties") is False
        for key in banned:
            assert not _contains_key(schema, key), definition.name
        assert _schema_depth(schema) <= 5


def test_design_workflow_rejects_disallowed_mutation_before_preview_or_write(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / "hephaestus.project.json"
    before_bytes = project_file.read_bytes()
    executor = ProjectToolExecutor(
        default_project_tool_registry(),
        ToolContext(project_dir),
        workflow_policy=mobile_robot_design_tool_policy(),
    )
    confirm_calls: list[dict] = []

    result = executor.execute(
        ToolCall("wrong", "project_add_requirement", _add_requirement_args()),
        allow_mutation=True,
        confirm=lambda preview: confirm_calls.append(preview) or True,
    )

    assert result["status"] == "wrong_operation_rejected"
    assert result["terminal"] is True
    assert result["workflow"] == "design"
    assert result["operation_family"] == "requirement"
    assert confirm_calls == []
    assert project_file.read_bytes() == before_bytes

def test_project_tools_preview_apply_and_permission(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    registry = default_project_tool_registry()
    executor = ProjectToolExecutor(registry, ToolContext(project_dir))

    summary = registry.execute(ToolCall("1", "project.get_summary", {}), ToolContext(project_dir))
    assert summary["status"] == "ok"
    assert summary["result"]["revision"] == 0

    listed = registry.execute(
        ToolCall("2", "project.list_entities", {"limit": 1}), ToolContext(project_dir)
    )
    assert listed["result"]["truncated"] is True

    missing = registry.execute(
        ToolCall("3", "project.get_entity", {"entity_id": "entity:none"}), ToolContext(project_dir)
    )
    assert missing["status"] == "error"

    preview = registry.execute(
        ToolCall("4", "project.preview_changeset", {"changeset": _changeset()}),
        ToolContext(project_dir),
    )
    assert preview["status"] == "ok"
    assert load_project(project_dir).revision == 0

    denied_raw = registry.execute(
        ToolCall("5", "project.apply_changeset", {"changeset": _changeset()}),
        ToolContext(project_dir),
        allow_mutation=False,
    )
    assert denied_raw["status"] == "error"

    applied_raw = registry.execute(
        ToolCall("6", "project.apply_changeset", {"changeset": _changeset()}),
        ToolContext(project_dir),
        allow_mutation=True,
    )
    assert applied_raw["status"] == "ok"
    assert load_project(project_dir).revision == 1

    hidden = executor.execute(
        ToolCall("7", "project.apply_changeset", {"changeset": _changeset()}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert hidden["status"] == "error"
    assert "not available" in hidden["error"]


def test_project_add_requirement_operation_previews_applies_and_injects_revision(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    save_project(replace(representative_project(), revision=4), project_dir)
    registry = default_project_tool_registry()
    executor = ProjectToolExecutor(registry, ToolContext(project_dir))
    args = _add_requirement_args()

    previews: list[dict] = []
    denied = executor.execute(
        ToolCall("req_apply_denied", "project_add_requirement", args),
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or False,
    )

    assert previews and previews[0]["status"] == "ok"
    assert previews[0]["result"]["dry_run"] is True
    assert previews[0]["result"]["revision"] == 5
    assert previews[0]["result"]["human_preview"]["validation"] == "passed"
    assert load_project(project_dir).revision == 4
    assert all(
        requirement.id != "requirement:the_robot_must_carry_a_payload_of_at_least_25_kg"
        for requirement in load_project(project_dir).requirements
    )

    assert denied["status"] == "denied"
    assert denied["preview"]["not_applied"] is True
    assert "after_fingerprint" not in denied["preview"]
    assert "revision" not in denied["preview"]
    assert load_project(project_dir).revision == 4

    applied = executor.execute(
        ToolCall("req_apply", "project_add_requirement", args),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    project = load_project(project_dir)
    inserted = next(
        requirement
        for requirement in project.requirements
        if requirement.id == "requirement:the_robot_must_carry_a_payload_of_at_least_25_kg"
    )
    assert applied["status"] == "ok"
    assert applied["result"]["revision"] == 5
    assert project.revision == 5
    assert inserted.kind.value == "payload"
    assert inserted.priority.value == "high"
    assert inserted.value is not None
    assert inserted.value.to_dict() == {
        "minimum": {"dimension": "mass", "unit": "kg", "value": 25.0}
    }
    assert inserted.provenance is not None
    assert inserted.provenance.source.value == "generated"


def test_project_add_requirement_rejects_invalid_model_facing_args(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    executor = ProjectToolExecutor(default_project_tool_registry(), ToolContext(project_dir))

    negative = {**_add_requirement_args(), "quantity_value": -1.0}
    negative_result = executor.execute(
        ToolCall("negative", "project_add_requirement", negative),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert negative_result["status"] == "error"
    assert negative_result["error"] == "invalid arguments"

    unsupported = {**_add_requirement_args(), "kind": "environment"}
    unsupported_result = executor.execute(
        ToolCall("unsupported", "project_add_requirement", unsupported),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert unsupported_result["status"] == "error"
    assert unsupported_result["error"] == "invalid arguments"
    assert load_project(project_dir).revision == 0


def test_project_create_mobile_robot_preview_apply_and_rejects_family(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    executor = ProjectToolExecutor(default_project_tool_registry(), ToolContext(project_dir))

    previews: list[dict] = []
    denied = executor.execute(
        ToolCall("design_denied", "project_create_mobile_robot", _mobile_robot_args()),
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or False,
    )
    assert denied["status"] == "denied"
    assert previews[0]["result"]["operation"] == "create_mobile_robot"
    assert previews[0]["result"]["dry_run"] is True
    assert previews[0]["result"]["human_preview"]["validation"] == "passed"
    assert load_project(project_dir).revision == 0

    applied = executor.execute(
        ToolCall("design_apply", "project_create_mobile_robot", _mobile_robot_args()),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    project = load_project(project_dir)
    assert applied["status"] == "ok"
    assert applied["result"]["revision"] == 1
    assert project.revision == 1
    assert any(entity.id == "entity:aisle_scout" for entity in project.entities)
    assert any(
        scenario.environment == "indoor_warehouse" for scenario in project.simulation_scenarios
    )

    unsupported = executor.execute(
        ToolCall(
            "design_unsupported",
            "project_create_mobile_robot",
            {**_mobile_robot_args(), "robot_family": "tracked", "robot_name": "Second Scout"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert unsupported["status"] == "error"
    assert "unsupported robot family" in unsupported["error"]


def test_mutation_preview_without_fingerprint_fails_closed(tmp_path: Path) -> None:
    registry = ToolRegistry()
    confirmed: list[bool] = []

    def preview(args: EmptyArgs, context: ToolContext) -> dict:
        return {"summary": "bad preview"}

    def apply(args: EmptyArgs, context: ToolContext) -> dict:
        raise AssertionError("apply should not run without preview fingerprint")

    registry.register(
        RegisteredTool(
            "bad_mutation",
            "bad",
            PermissionClass.PROJECT_MUTATION,
            EmptyArgs,
            apply,
            preview_handler=preview,
        )
    )
    result = ProjectToolExecutor(registry, ToolContext(tmp_path)).execute(
        ToolCall("bad", "bad_mutation", {}),
        allow_mutation=True,
        confirm=lambda preview: confirmed.append(True) or True,
    )

    assert result["status"] == "error"
    assert result["error"] == "mutation preview did not include before_fingerprint"
    assert confirmed == []




def test_mutation_handler_rechecks_expected_preview_fingerprint(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    before = fingerprint_project(load_project(project_dir))
    changed = replace(
        load_project(project_dir),
        metadata=ProjectMetadata("Changed Project", "changed after preview"),
    )
    save_project(changed, project_dir, overwrite=True)

    result = default_project_tool_registry().execute(
        ToolCall("stale", "project_create_mobile_robot", _mobile_robot_args()),
        ToolContext(project_dir, expected_fingerprint=before),
        allow_mutation=True,
    )

    assert result["status"] == "error"
    assert result["error"] == "active project changed after preview"
    assert load_project(project_dir).metadata.name == "Changed Project"
    assert load_project(project_dir).revision == 0

def test_confirmed_mutation_refuses_changed_project_after_preview(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(representative_project(), project_dir)
    executor = ProjectToolExecutor(default_project_tool_registry(), ToolContext(project_dir))

    def mutate_after_preview(preview: dict) -> bool:
        project = load_project(project_dir)
        changed = replace(
            project,
            metadata=ProjectMetadata(project.metadata.name, "changed after preview"),
        )
        save_project(changed, project_dir, overwrite=True)
        return True

    result = executor.execute(
        ToolCall("7", "project_add_requirement", _add_requirement_args()),
        allow_mutation=True,
        confirm=mutate_after_preview,
    )

    assert result["status"] == "error"
    assert result["error"] == "active project changed after preview"
    assert load_project(project_dir).revision == 0
    assert all(
        requirement.id != "requirement:the_robot_must_carry_a_payload_of_at_least_25_kg"
        for requirement in load_project(project_dir).requirements
    )


def test_tool_load_failure_is_sanitized(tmp_path: Path) -> None:
    missing = tmp_path / "missing_project"
    registry = default_project_tool_registry()
    result = registry.execute(ToolCall("8", "project.get_summary", {}), ToolContext(missing))
    assert result["status"] == "error"
    assert str(missing) not in result["error"]
