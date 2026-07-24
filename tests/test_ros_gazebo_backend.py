from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from hephaestus.artifacts import ArtifactStager, artifact_root, load_manifest
from hephaestus.backends.cadquery_backend import FakeCadAdapter, generate_cad_artifacts
from hephaestus.backends.ros_gazebo import (
    GENERATOR_VERSION,
    RosGazeboGenerationError,
    default_workspace_dir,
    generate_and_write_ros_gazebo_workspace,
    generate_ros_gazebo_artifacts,
)
from hephaestus.concept_robot import LocomotionMode, compile_concept_robot_design
from hephaestus.fixtures import blocky_yellow_warehouse_utility_robot_spec
from hephaestus.mobile_robot import DESIGN_EXTENSION_KEY
from hephaestus.model import EngineeringProject
from hephaestus.persistence import save_project


def _mobile_robot_project() -> EngineeringProject:
    ids = {
        "robot_entity": "entity:ros_gazebo_bot",
        "chassis_body": "body:ros_gazebo_bot_chassis",
        "chassis_geometry": "geometry:ros_gazebo_bot_chassis",
        "left_wheel_body": "body:ros_gazebo_bot_left_wheel",
        "right_wheel_body": "body:ros_gazebo_bot_right_wheel",
        "wheel_geometry": "geometry:ros_gazebo_bot_wheel",
        "caster_body": "body:ros_gazebo_bot_caster",
        "caster_geometry": "geometry:ros_gazebo_bot_caster",
    }
    values = {
        "chassis_length": {"value": 0.7, "unit": "m", "dimension": "length"},
        "chassis_width": {"value": 0.48, "unit": "m", "dimension": "length"},
        "chassis_height": {"value": 0.22, "unit": "m", "dimension": "length"},
        "wheel_radius": {"value": 0.08, "unit": "m", "dimension": "length"},
        "wheel_width": {"value": 0.04, "unit": "m", "dimension": "length"},
        "wheel_separation": {"value": 0.42, "unit": "m", "dimension": "length"},
        "caster_radius": {"value": 0.035, "unit": "m", "dimension": "length"},
        "ground_clearance": {"value": 0.04, "unit": "m", "dimension": "length"},
        "chassis_mass": {"value": 10.0, "unit": "kg", "dimension": "mass"},
        "wheel_mass": {"value": 0.8, "unit": "kg", "dimension": "mass"},
        "caster_mass": {"value": 0.3, "unit": "kg", "dimension": "mass"},
        "lidar_update_rate": {"value": 10.0, "unit": "Hz", "dimension": "frequency"},
        "imu_update_rate": {"value": 100.0, "unit": "Hz", "dimension": "frequency"},
        "simulation_update_rate": {"value": 100.0, "unit": "Hz", "dimension": "frequency"},
    }
    extension = {
        "schema": "hephaestus.mobile_robot.v1",
        "supported_family": "differential_drive",
        "robot_name": "ROS Gazebo Bot",
        "mission_summary": "Inspect warehouse aisles offline.",
        "values": values,
        "ids": ids,
    }
    base = EngineeringProject.empty(name="ROS Gazebo Bot", project_id="project:ros_gazebo_bot")
    return replace(base, extensions={DESIGN_EXTENSION_KEY: extension})


def _stage_cad_and_ros(tmp_path: Path) -> tuple[EngineeringProject, ArtifactStager, set[str]]:
    project = _mobile_robot_project()
    save_project(project, tmp_path)
    stager = ArtifactStager(tmp_path, project, backend_versions={"cadquery": "fake"})
    cad_records = generate_cad_artifacts(project, stager, adapter=FakeCadAdapter())
    ros_records = generate_ros_gazebo_artifacts(project, stager)
    paths = {record.path for record in (*cad_records, *ros_records)}
    return project, stager, paths


def test_ros_gazebo_artifacts_parse_and_reference_robot_meshes(tmp_path: Path) -> None:
    _project, stager, paths = _stage_cad_and_ros(tmp_path)
    root = stager.staging

    assert "ros2_ws/src/ros_gazebo_bot_description/urdf/ros_gazebo_bot.urdf.xacro" in paths
    assert "ros2_ws/src/ros_gazebo_bot_bringup/config/controllers.yaml" in paths
    assert "ros2_ws/src/ros_gazebo_bot_bringup/config/bridge.yaml" in paths
    assert "ros2_ws/src/ros_gazebo_bot_simulation/worlds/warehouse_inspection.sdf" in paths

    urdf = ET.parse(
        root / "ros2_ws/src/ros_gazebo_bot_description/urdf/ros_gazebo_bot.urdf"
    ).getroot()
    assert urdf.tag == "robot"
    assert urdf.find("ros2_control") is not None

    model_sdf = (
        root / "ros2_ws/src/ros_gazebo_bot_simulation/models/ros_gazebo_bot/model.sdf"
    ).read_text(encoding="utf-8")
    model_root = ET.fromstring(model_sdf)
    assert "model://ros_gazebo_bot/meshes/visual/chassis.stl" in model_sdf
    assert "model://ros_gazebo_bot/meshes/visual/wheel.stl" in model_sdf
    assert "model:///meshes/visual/wheel.stl" not in model_sdf
    assert model_sdf.count("<inertial>") == 4
    assert "<pose>0 0 0 1.570796326795 0 0</pose><geometry><cylinder>" in model_sdf
    control_plugin = next(
        (
            plugin
            for plugin in model_root.findall("./model/plugin")
            if plugin.attrib.get("name") == "gz_ros2_control::GazeboSimROS2ControlPlugin"
        ),
        None,
    )
    assert control_plugin is not None
    assert control_plugin.findtext("parameters") == (
        "install/share/ros_gazebo_bot_bringup/config/controllers.yaml"
    )

    wheel_collision_origin = urdf.find("./link[@name='left_wheel_link']/collision/origin")
    assert wheel_collision_origin is not None
    assert wheel_collision_origin.attrib["rpy"] == "1.570796326795 0 0"

    world_path = root / "ros2_ws/src/ros_gazebo_bot_simulation/worlds/warehouse_inspection.sdf"
    world_text = world_path.read_text(encoding="utf-8")
    world = ET.parse(world_path).getroot()
    assert world.find("world/include/uri").text == "model://ros_gazebo_bot"

    launch = (
        root / "ros2_ws/src/ros_gazebo_bot_bringup/launch/gazebo_headless.launch.py"
    ).read_text(encoding="utf-8")
    assert "DeclareLaunchArgument('world'" in launch
    assert "LaunchConfiguration('world')" in launch
    assert "executable='create'" not in launch
    assert "executable='parameter_bridge'" in launch
    assert launch.count("executable='spawner'") == 1
    assert "TimerAction" not in launch
    assert (
        "arguments=['joint_state_broadcaster', 'diff_drive_controller', '--controller-manager', "
        "'/controller_manager', '--controller-manager-timeout', '30', '--service-call-timeout', "
        "'30', '--switch-timeout', '30', '--activate-as-group']"
    ) in launch
    compile(launch, "gazebo_headless.launch.py", "exec")
    assert "robot_state_publisher.launch.py" in launch

    bridge = (root / "ros2_ws/src/ros_gazebo_bot_bringup/config/bridge.yaml").read_text(
        encoding="utf-8"
    )
    assert 'ros_topic_name: "/clock"' in bridge
    assert 'gz_topic_name: "/clock"' in bridge
    assert "qos_profile: CLOCK" in bridge
    assert "gz-sim-user-commands-system" in world_text
    assert "gz-sim-imu-system" in world_text
    stager.cleanup()


def test_concept_launch_uses_grouped_spawner_and_static_omits_diff_drive(tmp_path: Path) -> None:
    spec = blocky_yellow_warehouse_utility_robot_spec()
    drive_project = compile_concept_robot_design(
        EngineeringProject.empty(name="Blocky Yellow Warehouse Utility Robot"), spec
    ).project
    drive_stager = ArtifactStager(tmp_path / "drive", drive_project)
    generate_ros_gazebo_artifacts(drive_project, drive_stager)
    drive_launch = (
        drive_stager.staging
        / (
            "ros2_ws/src/blocky_yellow_warehouse_utility_robot_bringup/launch/"
            "gazebo_headless.launch.py"
        )
    ).read_text(encoding="utf-8")

    assert drive_launch.count("executable='spawner'") == 1
    assert "TimerAction" not in drive_launch
    assert (
        "arguments=['joint_state_broadcaster', 'diff_drive_controller', '--controller-manager', "
        "'/controller_manager', '--controller-manager-timeout', '30', '--service-call-timeout', "
        "'30', '--switch-timeout', '30', '--activate-as-group']"
    ) in drive_launch
    compile(drive_launch, "gazebo_headless.launch.py", "exec")

    static_spec = spec.model_copy(
        update={
            "robot_name": "Static Concept Robot",
            "locomotion_mode": LocomotionMode.STATIC,
            "bodies": (spec.bodies[0],),
            "joints": (),
            "sensors": (),
        }
    )
    static_project = compile_concept_robot_design(
        EngineeringProject.empty(name="Static Concept Robot"), static_spec
    ).project
    static_stager = ArtifactStager(tmp_path / "static", static_project)
    generate_ros_gazebo_artifacts(static_project, static_stager)
    static_launch = (
        static_stager.staging
        / "ros2_ws/src/static_concept_robot_bringup/launch/gazebo_headless.launch.py"
    ).read_text(encoding="utf-8")
    static_controllers = (
        static_stager.staging / "ros2_ws/src/static_concept_robot_bringup/config/controllers.yaml"
    ).read_text(encoding="utf-8")

    assert (
        "arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'"
        in static_launch
    )
    assert "diff_drive_controller" not in static_launch
    assert "diff_drive_controller" not in static_controllers
    compile(static_launch, "gazebo_headless.launch.py", "exec")
    drive_stager.cleanup()
    static_stager.cleanup()


def test_generated_setup_py_installs_package_data_without_flattening_models(tmp_path: Path) -> None:
    _project, stager, _paths = _stage_cad_and_ros(tmp_path)

    description_setup = (
        stager.staging / "ros2_ws/src/ros_gazebo_bot_description/setup.py"
    ).read_text(encoding="utf-8")
    simulation_setup = (
        stager.staging / "ros2_ws/src/ros_gazebo_bot_simulation/setup.py"
    ).read_text(encoding="utf-8")
    bringup_setup = (stager.staging / "ros2_ws/src/ros_gazebo_bot_bringup/setup.py").read_text(
        encoding="utf-8"
    )

    assert "('urdf/*', 'urdf')" in description_setup
    assert "('meshes/visual/*', 'meshes/visual')" in description_setup
    assert "models_root = Path('models')" in simulation_setup
    assert "models_root.rglob('*')" in simulation_setup
    assert "('worlds/*', 'worlds')" in simulation_setup
    assert "('config/*', 'config')" in bringup_setup
    assert "('launch/*', 'launch')" in bringup_setup
    stager.cleanup()


def test_bringup_package_declares_simulation_package_dependency(tmp_path: Path) -> None:
    _project, stager, _paths = _stage_cad_and_ros(tmp_path)

    bringup_package = (stager.staging / "ros2_ws/src/ros_gazebo_bot_bringup/package.xml").read_text(
        encoding="utf-8"
    )

    assert "<exec_depend>ros_gazebo_bot_description</exec_depend>" in bringup_package
    assert "<exec_depend>ros_gazebo_bot_simulation</exec_depend>" in bringup_package
    stager.cleanup()


def test_ros_gazebo_publish_manifest_records_fresh_ros_artifacts(tmp_path: Path) -> None:
    _project, stager, paths = _stage_cad_and_ros(tmp_path)
    records = []
    for path in sorted(paths):
        role = "ros" if path.startswith("ros2_ws/") else "cad"
        from hephaestus.artifacts import record_for_file

        records.append(
            record_for_file(stager.staging, path, role=role, owning_ids=("entity:test",))
        )
    manifest = stager.publish(
        records, generator_versions={"hephaestus.ros_gazebo_backend": GENERATOR_VERSION}
    )

    loaded, status = load_manifest(tmp_path)
    assert loaded is not None
    assert status == "fresh"
    assert loaded.project_fingerprint == manifest.project_fingerprint
    assert (
        artifact_root(tmp_path) / "ros2_ws/src/ros_gazebo_bot_bringup/launch/gazebo.launch.py"
    ).exists()


def test_generate_and_write_ros_gazebo_workspace_requires_project_scoped_workspace(
    tmp_path: Path,
) -> None:
    project = _mobile_robot_project()
    save_project(project, tmp_path)

    with pytest.raises(RosGazeboGenerationError, match="workspace escapes project scope"):
        generate_and_write_ros_gazebo_workspace(project, tmp_path, tmp_path.parent / "outside")

    assert default_workspace_dir(tmp_path) == artifact_root(tmp_path) / "ros2_ws"
