from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from hephaestus.changes import fingerprint_project
from hephaestus.concept_robot import (
    CONCEPT_EXTENSION_KEY,
    ConceptBodySpec,
    ConceptColor,
    ConceptError,
    ConceptJointLimit,
    ConceptJointSpec,
    ConceptJointType,
    ConceptPose,
    ConceptRobotSpec,
    ConceptWorld,
    FunctionalRole,
    LocomotionMode,
    PrimitiveType,
    WheelRole,
    compile_concept_robot_design,
    primitive_inertia,
    semantic_id_map,
)
from hephaestus.fixtures import (
    blocky_yellow_warehouse_utility_robot_project,
    blocky_yellow_warehouse_utility_robot_spec,
    edited_blocky_yellow_warehouse_utility_robot_spec,
)
from hephaestus.model import EngineeringProject, Entity
from hephaestus.persistence import canonical_project_json


def _color() -> ConceptColor:
    return ConceptColor(r=0.98, g=0.78, b=0.08, a=1.0)


def _body(key: str = "base", *, mass_kg: float = 8.0) -> ConceptBodySpec:
    return ConceptBodySpec(
        semantic_key=key,
        display_name=key.replace("_", " ").title(),
        primitive_type=PrimitiveType.BOX,
        size_x_m=0.5,
        size_y_m=0.3,
        size_z_m=0.2,
        pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.1),
        mass_kg=mass_kg,
        visual_color=_color(),
        functional_role=FunctionalRole.CHASSIS,
    )


def _wheel(key: str, y_m: float, role: WheelRole) -> ConceptBodySpec:
    return ConceptBodySpec(
        semantic_key=key,
        display_name=key.replace("_", " ").title(),
        primitive_type=PrimitiveType.CYLINDER,
        radius_m=0.09,
        length_m=0.05,
        pose=ConceptPose(x_m=-0.1, y_m=y_m, z_m=0.09),
        mass_kg=1.2,
        visual_color=ConceptColor(r=0.04, g=0.04, b=0.04, a=1.0),
        functional_role=FunctionalRole.DRIVE_WHEEL,
    )


def _drive_joint(key: str, child: str, role: WheelRole) -> ConceptJointSpec:
    return ConceptJointSpec(
        semantic_key=key,
        parent_body_key="base",
        child_body_key=child,
        joint_type=ConceptJointType.CONTINUOUS,
        origin=ConceptPose(x_m=-0.1, y_m=0.0, z_m=0.09),
        axis_y=1.0,
        axis_z=0.0,
        limits=ConceptJointLimit(),
        wheel_role=role,
    )


def _minimal_spec(**overrides: object) -> ConceptRobotSpec:
    values = {
        "spec_version": "hephaestus.concept_robot.v1",
        "robot_name": "Utility Bot",
        "design_summary": "Move generic warehouse payloads.",
        "locomotion_mode": LocomotionMode.STATIC,
        "bodies": (_body(),),
        "joints": (),
        "sensors": (),
        "world_template": ConceptWorld.WAREHOUSE,
        "assumptions": (),
        "engineering_warnings": (),
    }
    values.update(overrides)
    return ConceptRobotSpec(**values)


def _drive_spec() -> ConceptRobotSpec:
    return _minimal_spec(
        locomotion_mode=LocomotionMode.DIFFERENTIAL_DRIVE,
        bodies=(
            _body(),
            _wheel("left_wheel", 0.28, WheelRole.LEFT_DRIVE),
            _wheel("right_wheel", -0.28, WheelRole.RIGHT_DRIVE),
        ),
        joints=(
            _drive_joint("left_drive", "left_wheel", WheelRole.LEFT_DRIVE),
            _drive_joint("right_drive", "right_wheel", WheelRole.RIGHT_DRIVE),
        ),
    )


def test_concept_robot_spec_validation_is_strict_and_limited() -> None:
    with pytest.raises(ValidationError):
        _body("bad-key")
    with pytest.raises(ValidationError):
        _body(mass_kg=0.0)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ConceptRobotSpec.model_validate({**_minimal_spec().model_dump(), "extra": True})
    with pytest.raises(ValidationError, match="duplicate body semantic key"):
        _minimal_spec(bodies=(_body("base"), _body("base")))


def test_semantic_key_mapping_is_deterministic_and_project_scoped() -> None:
    spec = _drive_spec()
    mapping = semantic_id_map(spec, project_scope="warehouse")

    assert mapping["bodies"]["base"]["body"] == "body:warehouse_concept_base"
    assert mapping["joints"]["left_drive"] == "joint:warehouse_concept_left_drive"
    assert mapping == semantic_id_map(spec, project_scope="warehouse")


def test_primitive_inertia_calculations_cover_box_cylinder_and_sphere() -> None:
    box = ConceptBodySpec(
        semantic_key="box",
        display_name="Box",
        primitive_type=PrimitiveType.BOX,
        size_x_m=2.0,
        size_y_m=3.0,
        size_z_m=4.0,
        pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.0),
        mass_kg=12.0,
        visual_color=_color(),
        functional_role=FunctionalRole.OTHER,
    )
    box_inertia = primitive_inertia(box)
    assert box_inertia[0][0] == pytest.approx(25.0)
    assert box_inertia[1][1] == pytest.approx(20.0)
    assert box_inertia[2][2] == pytest.approx(13.0)

    cylinder = ConceptBodySpec(
        semantic_key="wheel",
        display_name="Wheel",
        primitive_type=PrimitiveType.CYLINDER,
        radius_m=0.2,
        length_m=0.1,
        pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.0),
        mass_kg=3.0,
        visual_color=_color(),
        functional_role=FunctionalRole.DRIVE_WHEEL,
    )
    axis_inertia = 0.5 * 3.0 * 0.2 * 0.2
    other_inertia = 3.0 * (3.0 * 0.2 * 0.2 + 0.1 * 0.1) / 12.0
    cylinder_inertia = primitive_inertia(cylinder)
    assert cylinder_inertia[0][0] == pytest.approx(other_inertia)
    assert cylinder_inertia[1][1] == pytest.approx(axis_inertia)
    assert cylinder_inertia[2][2] == pytest.approx(other_inertia)

    sphere = ConceptBodySpec(
        semantic_key="ball",
        display_name="Ball",
        primitive_type=PrimitiveType.SPHERE,
        radius_m=0.3,
        pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.0),
        mass_kg=2.0,
        visual_color=_color(),
        functional_role=FunctionalRole.PASSIVE_WHEEL,
    )
    assert primitive_inertia(sphere)[0][0] == pytest.approx(0.4 * 2.0 * 0.3 * 0.3)


def test_warnings_are_reported_for_bad_but_representable_designs() -> None:
    tall = _body()
    tall_data = tall.model_dump()
    tall_data.update({"size_z_m": 1.2, "pose": {**tall.pose.model_dump(), "z_m": 0.7}})
    spec = _minimal_spec(
        bodies=(ConceptBodySpec.model_validate(tall_data),),
        engineering_warnings=("User marked this design as concept-only.",),
    )

    compiled = compile_concept_robot_design(EngineeringProject.empty(name="Warnings"), spec)

    assert "User marked this design as concept-only." in compiled.warnings
    assert "High center-mounted structures may make the robot likely to tip." in compiled.warnings
    assert compiled.validation.is_valid


def test_invalid_specs_and_id_collisions_fail_cleanly() -> None:
    with pytest.raises(ValidationError, match="joint references an unknown body"):
        _minimal_spec(
            joints=(
                ConceptJointSpec(
                    semantic_key="bad_mount",
                    parent_body_key="base",
                    child_body_key="missing",
                    joint_type=ConceptJointType.FIXED,
                    origin=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.0),
                ),
            )
        )
    with pytest.raises(ValidationError, match="requires a left drive wheel"):
        _minimal_spec(locomotion_mode=LocomotionMode.DIFFERENTIAL_DRIVE)
    with pytest.raises(ConceptError, match="duplicate existing canonical IDs"):
        compile_concept_robot_design(
            replace(
                EngineeringProject.empty(name="Collision"),
                entities=(Entity("entity:project_hephaestus_concept_robot", "Existing robot"),),
            ),
            _minimal_spec(),
        )


def test_blocky_yellow_fixture_compiles_to_valid_deterministic_project() -> None:
    spec = blocky_yellow_warehouse_utility_robot_spec()
    compiled = compile_concept_robot_design(
        EngineeringProject.empty(name="Blocky Yellow Warehouse Utility Robot"), spec
    )
    project = compiled.project

    assert compiled.validation.is_valid
    assert project.revision == 1
    assert len(project.bodies) == 14
    assert len(project.joints) == 13
    assert {sensor.modality.value for sensor in project.sensors} == {"lidar_2d", "imu"}
    assert CONCEPT_EXTENSION_KEY in project.extensions
    assert project.extensions[CONCEPT_EXTENSION_KEY]["source_spec"]["robot_name"] == (
        "Blocky Yellow Warehouse Utility Robot"
    )
    colors = [material.extensions.get("color_rgba") for material in project.materials]
    assert [0.98, 0.78, 0.08, 1.0] in colors

    second = blocky_yellow_warehouse_utility_robot_project()
    assert canonical_project_json(project) == canonical_project_json(second)
    assert fingerprint_project(project) == fingerprint_project(second)


def test_concept_robot_edit_replaces_generated_slice_and_increments_revision() -> None:
    base_compiled = compile_concept_robot_design(
        EngineeringProject.empty(name="Fixture"), blocky_yellow_warehouse_utility_robot_spec()
    )
    before = base_compiled.project
    before_fingerprint = fingerprint_project(before)

    edited = compile_concept_robot_design(
        before,
        edited_blocky_yellow_warehouse_utility_robot_spec(),
        previous_spec=blocky_yellow_warehouse_utility_robot_spec(),
    )
    after = edited.project

    assert edited.before_fingerprint == before_fingerprint
    assert edited.after_fingerprint != before_fingerprint
    assert after.revision == 2
    assert len({body.id for body in after.bodies}) == len(after.bodies)
    assert len(after.bodies) == len(before.bodies)
    assert "changed body torso" in edited.semantic_diff
    assert "changed body lidar_puck" in edited.semantic_diff
    torso_body = next(body for body in after.bodies if body.id.endswith("_torso"))
    assert torso_body.mass_properties is not None
    assert torso_body.mass_properties.mass.value == 4.8
    torso_geometry = next(
        geometry for geometry in after.geometries if geometry.id.endswith("_torso")
    )
    assert [dimension.value for dimension in torso_geometry.dimensions] == [0.34, 0.3, 0.36]
    assert after.extensions[CONCEPT_EXTENSION_KEY]["source_spec"]["design_summary"].startswith(
        "Move inspection payloads with a taller torso"
    )
