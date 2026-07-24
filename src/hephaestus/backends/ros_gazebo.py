"""ROS 2 Jazzy and Gazebo Harmonic artifact generation."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.artifacts import (
    ArtifactRecord,
    ArtifactStager,
    artifact_root,
    record_for_file,
    sanitized_name,
)
from hephaestus.concept_robot import ConceptError, PrimitiveType, concept_backend_spec
from hephaestus.mobile_robot import mobile_robot_spec
from hephaestus.model import EngineeringProject

GENERATOR_VERSION = "hephaestus.ros_gazebo_backend.v1"
DESCRIPTION_PACKAGE = "robot_description"
BRINGUP_PACKAGE = "robot_bringup"
SIMULATION_PACKAGE = "robot_simulation"
CONTROLLER_PACKAGE = "robot_control"


@dataclass(frozen=True, slots=True)
class GeneratedWorkspace:
    workspace_dir: Path
    model_name: str
    files: tuple[object, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace_dir),
            "model_name": self.model_name,
            "files": [str(getattr(item, "relative_path", item)) for item in self.files],
            "warnings": list(self.warnings),
        }


class RosGazeboGenerationError(RuntimeError):
    """Raised when ROS/Gazebo artifacts cannot be generated."""


def default_workspace_dir(project_dir: Path) -> Path:
    return artifact_root(project_dir) / "ros2_ws"


def generate_ros_gazebo_workspace(
    project: EngineeringProject, workspace_dir: Path
) -> GeneratedWorkspace:
    from hephaestus.artifacts import ArtifactStager

    project_dir = (
        workspace_dir.parent.parent
        if workspace_dir.parent.name == "generated"
        else workspace_dir.parent
    )
    stager = ArtifactStager(project_dir, project)
    try:
        _copy_existing_cad_meshes(project_dir, stager)
        records = generate_ros_gazebo_artifacts(project, stager)
        source_workspace = stager.staging / "ros2_ws"
        if not source_workspace.exists():
            raise RosGazeboGenerationError("ROS/Gazebo staging did not produce ros2_ws")
        for child in source_workspace.iterdir():
            target = workspace_dir / child.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        try:
            backend_spec = concept_backend_spec(project)
            model_name = sanitized_name(str(backend_spec["spec"]["robot_name"]))
        except ConceptError:
            spec = mobile_robot_spec(project)
            model_name = sanitized_name(str(spec["robot_name"]))
        return GeneratedWorkspace(
            workspace_dir=workspace_dir.resolve(),
            model_name=model_name,
            files=records,
        )
    finally:
        stager.cleanup()


def generate_and_write_ros_gazebo_workspace(
    project: EngineeringProject,
    project_dir: Path,
    workspace_dir: Path | None = None,
) -> GeneratedWorkspace:
    target = workspace_dir if workspace_dir is not None else default_workspace_dir(project_dir)
    resolved_project = project_dir.resolve()
    resolved_target_parent = target.parent.resolve() if target.parent.exists() else target.parent
    try:
        resolved_target_parent.relative_to(resolved_project)
    except ValueError as exc:
        raise RosGazeboGenerationError(f"workspace escapes project scope: {target}") from exc
    return generate_ros_gazebo_workspace(project, target)


def _copy_existing_cad_meshes(project_dir: Path, stager: ArtifactStager) -> None:
    source_root = artifact_root(project_dir) / "cad" / "stl"
    if not source_root.exists():
        return
    for role in ("visual", "collision"):
        for name in ("chassis", "wheel", "caster"):
            source = source_root / role / f"{name}.stl"
            if source.exists():
                shutil.copy2(source, stager.path(f"cad/stl/{role}/{name}.stl"))


def generate_ros_gazebo_artifacts(
    project: EngineeringProject,
    stager: ArtifactStager,
) -> tuple[ArtifactRecord, ...]:
    try:
        concept = concept_backend_spec(project)
    except ConceptError:
        pass
    else:
        return _generate_concept_ros_gazebo_artifacts(concept, stager)
    spec = mobile_robot_spec(project)
    robot_name = sanitized_name(spec["robot_name"])
    desc_pkg = f"{robot_name}_description"
    bringup_pkg = f"{robot_name}_bringup"
    sim_pkg = f"{robot_name}_simulation"
    values = spec["values"]
    ids = spec["ids"]
    records: list[ArtifactRecord] = []

    _copy_meshes(stager, desc_pkg, sim_pkg, robot_name)
    files = {
        f"ros2_ws/src/{desc_pkg}/package.xml": _description_package_xml(desc_pkg),
        f"ros2_ws/src/{desc_pkg}/setup.py": _setup_py(desc_pkg),
        f"ros2_ws/src/{desc_pkg}/setup.cfg": _setup_cfg(desc_pkg),
        f"ros2_ws/src/{desc_pkg}/resource/{desc_pkg}": "",
        f"ros2_ws/src/{desc_pkg}/{desc_pkg}/__init__.py": "",
        f"ros2_ws/src/{desc_pkg}/urdf/{robot_name}.urdf.xacro": _xacro(
            robot_name, desc_pkg, bringup_pkg, values
        ),
        f"ros2_ws/src/{desc_pkg}/urdf/{robot_name}.urdf": _urdf(robot_name, desc_pkg, values),
        f"ros2_ws/src/{bringup_pkg}/package.xml": _bringup_package_xml(
            bringup_pkg, desc_pkg, sim_pkg
        ),
        f"ros2_ws/src/{bringup_pkg}/setup.py": _setup_py(bringup_pkg),
        f"ros2_ws/src/{bringup_pkg}/setup.cfg": _setup_cfg(bringup_pkg),
        f"ros2_ws/src/{bringup_pkg}/resource/{bringup_pkg}": "",
        f"ros2_ws/src/{bringup_pkg}/{bringup_pkg}/__init__.py": "",
        f"ros2_ws/src/{bringup_pkg}/config/controllers.yaml": _controllers_yaml(robot_name, values),
        f"ros2_ws/src/{bringup_pkg}/config/bridge.yaml": _bridge_yaml(robot_name),
        f"ros2_ws/src/{bringup_pkg}/launch/robot_state_publisher.launch.py": _rsp_launch(
            desc_pkg, robot_name
        ),
        f"ros2_ws/src/{bringup_pkg}/launch/gazebo.launch.py": _gazebo_launch(
            desc_pkg,
            bringup_pkg,
            sim_pkg,
            robot_name,
            headless=False,
            controller_names=("joint_state_broadcaster", "diff_drive_controller"),
        ),
        f"ros2_ws/src/{bringup_pkg}/launch/gazebo_headless.launch.py": _gazebo_launch(
            desc_pkg,
            bringup_pkg,
            sim_pkg,
            robot_name,
            headless=True,
            controller_names=("joint_state_broadcaster", "diff_drive_controller"),
        ),
        f"ros2_ws/src/{sim_pkg}/package.xml": _simulation_package_xml(sim_pkg, desc_pkg),
        f"ros2_ws/src/{sim_pkg}/setup.py": _setup_py(sim_pkg),
        f"ros2_ws/src/{sim_pkg}/setup.cfg": _setup_cfg(sim_pkg),
        f"ros2_ws/src/{sim_pkg}/resource/{sim_pkg}": "",
        f"ros2_ws/src/{sim_pkg}/{sim_pkg}/__init__.py": "",
        f"ros2_ws/src/{sim_pkg}/models/{robot_name}/model.config": _model_config(robot_name),
        f"ros2_ws/src/{sim_pkg}/models/{robot_name}/model.sdf": _sdf_model(
            robot_name, bringup_pkg, values
        ),
        f"ros2_ws/src/{sim_pkg}/worlds/warehouse_inspection.sdf": _world(robot_name),
        f"gazebo/models/{robot_name}/model.config": _model_config(robot_name),
        f"gazebo/models/{robot_name}/model.sdf": _sdf_model(robot_name, bringup_pkg, values),
        "gazebo/worlds/warehouse_inspection.sdf": _world(robot_name),
    }
    for rel, content in files.items():
        stager.write_text(rel, content)
    _validate_xml(stager.staging / f"ros2_ws/src/{desc_pkg}/urdf/{robot_name}.urdf")
    _validate_xml(stager.staging / f"ros2_ws/src/{sim_pkg}/models/{robot_name}/model.sdf")
    _validate_xml(stager.staging / f"ros2_ws/src/{sim_pkg}/worlds/warehouse_inspection.sdf")

    for rel in sorted(files):
        role = _role_for_path(rel)
        records.append(
            record_for_file(stager.staging, rel, role=role, owning_ids=(ids["robot_entity"],))
        )
    for rel in _mesh_records(desc_pkg, sim_pkg, robot_name):
        role = "visual_mesh_copy" if "/visual/" in rel else "collision_mesh_copy"
        records.append(
            record_for_file(
                stager.staging,
                rel,
                role=role,
                owning_ids=(ids["robot_entity"],),
                units="mm mesh vertices; URDF/SDF scale 0.001",
            )
        )
    report = _validation_report(robot_name, desc_pkg, sim_pkg, values)
    report_rel = "reports/ros_gazebo_report.json"
    stager.write_text(
        report_rel, json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    records.append(
        record_for_file(
            stager.staging, report_rel, role="ros_gazebo_report", owning_ids=(ids["robot_entity"],)
        )
    )
    return tuple(records)


def _generate_concept_ros_gazebo_artifacts(
    concept: dict[str, Any], stager: ArtifactStager
) -> tuple[ArtifactRecord, ...]:
    spec = concept["spec"]
    ids = concept["ids"]
    robot_name = sanitized_name(str(spec["robot_name"]))
    desc_pkg = f"{robot_name}_description"
    bringup_pkg = f"{robot_name}_bringup"
    sim_pkg = f"{robot_name}_simulation"
    records: list[ArtifactRecord] = []
    files = {
        f"ros2_ws/src/{desc_pkg}/package.xml": _description_package_xml(desc_pkg),
        f"ros2_ws/src/{desc_pkg}/setup.py": _setup_py(desc_pkg),
        f"ros2_ws/src/{desc_pkg}/setup.cfg": _setup_cfg(desc_pkg),
        f"ros2_ws/src/{desc_pkg}/resource/{desc_pkg}": "",
        f"ros2_ws/src/{desc_pkg}/{desc_pkg}/__init__.py": "",
        f"ros2_ws/src/{desc_pkg}/urdf/{robot_name}.urdf.xacro": _concept_xacro(
            robot_name, bringup_pkg, spec
        ),
        f"ros2_ws/src/{desc_pkg}/urdf/{robot_name}.urdf": _concept_urdf(
            robot_name, spec, include_control=False
        ),
        f"ros2_ws/src/{bringup_pkg}/package.xml": _bringup_package_xml(
            bringup_pkg, desc_pkg, sim_pkg
        ),
        f"ros2_ws/src/{bringup_pkg}/setup.py": _setup_py(bringup_pkg),
        f"ros2_ws/src/{bringup_pkg}/setup.cfg": _setup_cfg(bringup_pkg),
        f"ros2_ws/src/{bringup_pkg}/resource/{bringup_pkg}": "",
        f"ros2_ws/src/{bringup_pkg}/{bringup_pkg}/__init__.py": "",
        f"ros2_ws/src/{bringup_pkg}/config/controllers.yaml": _concept_controllers_yaml(spec),
        f"ros2_ws/src/{bringup_pkg}/config/bridge.yaml": _bridge_yaml(robot_name),
        f"ros2_ws/src/{bringup_pkg}/launch/robot_state_publisher.launch.py": _rsp_launch(
            desc_pkg, robot_name
        ),
        f"ros2_ws/src/{bringup_pkg}/launch/gazebo.launch.py": _gazebo_launch(
            desc_pkg,
            bringup_pkg,
            sim_pkg,
            robot_name,
            headless=False,
            controller_names=_concept_launch_controllers(spec),
        ),
        f"ros2_ws/src/{bringup_pkg}/launch/gazebo_headless.launch.py": _gazebo_launch(
            desc_pkg,
            bringup_pkg,
            sim_pkg,
            robot_name,
            headless=True,
            controller_names=_concept_launch_controllers(spec),
        ),
        f"ros2_ws/src/{sim_pkg}/package.xml": _simulation_package_xml(sim_pkg, desc_pkg),
        f"ros2_ws/src/{sim_pkg}/setup.py": _setup_py(sim_pkg),
        f"ros2_ws/src/{sim_pkg}/setup.cfg": _setup_cfg(sim_pkg),
        f"ros2_ws/src/{sim_pkg}/resource/{sim_pkg}": "",
        f"ros2_ws/src/{sim_pkg}/{sim_pkg}/__init__.py": "",
        f"ros2_ws/src/{sim_pkg}/models/{robot_name}/model.config": _model_config(robot_name),
        f"ros2_ws/src/{sim_pkg}/models/{robot_name}/model.sdf": _concept_sdf_model(
            robot_name, bringup_pkg, spec
        ),
        f"ros2_ws/src/{sim_pkg}/worlds/warehouse_inspection.sdf": _concept_world(
            robot_name, str(spec["world_template"])
        ),
        f"gazebo/models/{robot_name}/model.config": _model_config(robot_name),
        f"gazebo/models/{robot_name}/model.sdf": _concept_sdf_model(robot_name, bringup_pkg, spec),
        "gazebo/worlds/warehouse_inspection.sdf": _concept_world(
            robot_name, str(spec["world_template"])
        ),
    }
    for rel, content in files.items():
        stager.write_text(rel, content)
    _validate_xml(stager.staging / f"ros2_ws/src/{desc_pkg}/urdf/{robot_name}.urdf")
    _validate_xml(stager.staging / f"ros2_ws/src/{sim_pkg}/models/{robot_name}/model.sdf")
    _validate_xml(stager.staging / f"ros2_ws/src/{sim_pkg}/worlds/warehouse_inspection.sdf")
    owner = (str(ids.get("robot_entity", "entity:concept_robot")),)
    for rel in sorted(files):
        records.append(
            record_for_file(stager.staging, rel, role=_role_for_path(rel), owning_ids=owner)
        )
    report = _concept_validation_report(robot_name, desc_pkg, sim_pkg, spec, concept)
    report_rel = "reports/ros_gazebo_report.json"
    stager.write_text(
        report_rel, json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    records.append(
        record_for_file(stager.staging, report_rel, role="ros_gazebo_report", owning_ids=owner)
    )
    return tuple(records)


def _concept_xacro(robot_name: str, bringup_pkg: str, spec: dict[str, Any]) -> str:
    urdf = _concept_urdf(robot_name, spec, include_control=True)
    plugin = f"""
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>install/share/{bringup_pkg}/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>
"""
    return urdf.replace("</robot>\n", plugin + "</robot>\n")


def _concept_urdf(robot_name: str, spec: dict[str, Any], *, include_control: bool) -> str:
    bodies = {str(body["semantic_key"]): body for body in spec["bodies"]}
    root_key = _concept_root_body_key(spec)
    body_links = {key: ("base_link" if key == root_key else f"{key}_link") for key in bodies}
    root = bodies[root_key]
    parts = [
        f'<?xml version="1.0"?>\n<robot name="{robot_name}">',
        '  <link name="base_footprint"/>',
    ]
    parts.append(f'''  <joint name="base_footprint_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="{_pose_xyz(root["pose"])}" rpy="{_pose_rpy(root["pose"])}"/>
  </joint>''')
    for key in sorted(bodies):
        parts.append(_concept_urdf_link(body_links[key], bodies[key]))
    for joint in sorted(spec["joints"], key=lambda item: item["semantic_key"]):
        origin = joint["origin"]
        parts.append(f'''  <joint name="{joint["semantic_key"]}" type="{_urdf_joint_type(str(joint["joint_type"]))}">
    <parent link="{body_links[str(joint["parent_body_key"])]}"/>
    <child link="{body_links[str(joint["child_body_key"])]}"/>
    <origin xyz="{_pose_xyz(origin)}" rpy="{_pose_rpy(origin)}"/>
{_urdf_axis(joint)}  </joint>''')
    for sensor in sorted(spec["sensors"], key=lambda item: item["semantic_key"]):
        origin = sensor["pose"]
        sensor_link = f"{sensor['semantic_key']}_link"
        parts.append(f'  <link name="{sensor_link}"/>')
        parts.append(f'''  <joint name="{sensor["semantic_key"]}_joint" type="fixed">
    <parent link="{body_links[str(sensor["parent_body_key"])]}"/>
    <child link="{sensor_link}"/>
    <origin xyz="{_pose_xyz(origin)}" rpy="{_pose_rpy(origin)}"/>
  </joint>''')
    if include_control and spec["locomotion_mode"] != "static":
        controls = "\n".join(
            _control_joint(str(joint["semantic_key"])) for joint in _concept_drive_joints(spec)
        )
        parts.append(f"""  <ros2_control name="GazeboSimSystem" type="system">
    <hardware>
      <plugin>gz_ros2_control/GazeboSimSystem</plugin>
    </hardware>
{controls}
  </ros2_control>""")
    parts.append("</robot>\n")
    return "\n".join(parts)


def _concept_urdf_link(link_name: str, body: dict[str, Any]) -> str:
    mass = float(body["mass_kg"])
    inertia = _concept_inertia_diag(body)
    color = body["visual_color"]
    material = f"{float(color['r']):.4f} {float(color['g']):.4f} {float(color['b']):.4f} {float(color['a']):.4f}"
    geometry = _concept_urdf_geometry(body)
    visual_origin = _cylinder_visual_origin(body)
    return f'''  <link name="{link_name}">
    {_inertial(mass, inertia)}
    <visual>{visual_origin}<geometry>{geometry}</geometry><material name="{body["semantic_key"]}_mat"><color rgba="{material}"/></material></visual>
    <collision>{visual_origin}<geometry>{geometry}</geometry></collision>
  </link>'''


def _concept_urdf_geometry(body: dict[str, Any]) -> str:
    primitive = str(body["primitive_type"])
    if primitive == "box":
        return f'<box size="{float(body["size_x_m"]):.6f} {float(body["size_y_m"]):.6f} {float(body["size_z_m"]):.6f}"/>'
    if primitive == "cylinder":
        return f'<cylinder radius="{float(body["radius_m"]):.6f}" length="{float(body["length_m"]):.6f}"/>'
    return f'<sphere radius="{float(body["radius_m"]):.6f}"/>'


def _concept_sdf_model(robot_name: str, bringup_pkg: str, spec: dict[str, Any]) -> str:
    bodies = {str(body["semantic_key"]): body for body in spec["bodies"]}
    root_key = _concept_root_body_key(spec)
    body_links = {key: ("base_link" if key == root_key else f"{key}_link") for key in bodies}
    root = bodies[root_key]
    absolute_poses = _absolute_body_poses(spec, bodies, root_key)
    parts = [
        f'''<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{robot_name}">
    <pose>{_pose_sdf(root["pose"])}</pose>'''
    ]
    for key in sorted(bodies):
        pose = (
            ""
            if key == root_key
            else f'<pose relative_to="base_link">{_pose_sdf(_relative_pose(absolute_poses[key], absolute_poses[root_key]))}</pose>'
        )
        parts.append(_concept_sdf_link(body_links[key], bodies[key], pose, spec))
    for joint in sorted(spec["joints"], key=lambda item: item["semantic_key"]):
        parts.append(_concept_sdf_joint(joint, body_links))
    parts.append(f"""    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>install/share/{bringup_pkg}/config/controllers.yaml</parameters>
    </plugin>
  </model>
</sdf>
""")
    return "\n".join(parts)


def _concept_sdf_link(
    link_name: str, body: dict[str, Any], pose_xml: str, spec: dict[str, Any]
) -> str:
    mass = float(body["mass_kg"])
    inertia = _concept_inertia_diag(body)
    color = body["visual_color"]
    material = f"{float(color['r']):.4f} {float(color['g']):.4f} {float(color['b']):.4f} {float(color['a']):.4f}"
    geom = _concept_sdf_geometry(body)
    geom_pose = _cylinder_sdf_pose(body)
    sensors = "".join(
        _concept_sdf_sensor(sensor, body)
        for sensor in spec["sensors"]
        if sensor["parent_body_key"] == body["semantic_key"]
    )
    return f'''    <link name="{link_name}">{pose_xml}
      {_sdf_inertial(mass, inertia)}
      <visual name="{body["semantic_key"]}_visual">{geom_pose}<geometry>{geom}</geometry><material><ambient>{material}</ambient><diffuse>{material}</diffuse></material></visual>
      <collision name="{body["semantic_key"]}_collision">{geom_pose}<geometry>{geom}</geometry></collision>
{sensors}    </link>'''


def _concept_sdf_sensor(sensor: dict[str, Any], parent: dict[str, Any]) -> str:
    del parent
    rel = sensor["pose"]
    update = float(sensor["update_rate_hz"])
    topic = str(sensor["topic"])
    if sensor["sensor_type"] == "lidar_2d":
        return f'''      <sensor name="{sensor["semantic_key"]}" type="gpu_lidar">
        <pose>{_pose_sdf(rel)}</pose>
        <topic>{topic}</topic>
        <update_rate>{update:.1f}</update_rate>
        <lidar><scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan><range><min>0.05</min><max>12.0</max><resolution>0.01</resolution></range><noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise></lidar>
      </sensor>
'''
    return f'''      <sensor name="{sensor["semantic_key"]}" type="imu">
        <pose>{_pose_sdf(rel)}</pose>
        <topic>{topic}</topic>
        <update_rate>{update:.1f}</update_rate>
      </sensor>
'''


def _concept_sdf_geometry(body: dict[str, Any]) -> str:
    primitive = str(body["primitive_type"])
    if primitive == "box":
        return f"<box><size>{float(body['size_x_m']):.6f} {float(body['size_y_m']):.6f} {float(body['size_z_m']):.6f}</size></box>"
    if primitive == "cylinder":
        return f"<cylinder><radius>{float(body['radius_m']):.6f}</radius><length>{float(body['length_m']):.6f}</length></cylinder>"
    return f"<sphere><radius>{float(body['radius_m']):.6f}</radius></sphere>"


def _concept_sdf_joint(joint: dict[str, Any], body_links: dict[str, str]) -> str:
    axis = ""
    if str(joint["joint_type"]) == "continuous":
        axis = f"<axis><xyz>{float(joint['axis_x']):.6f} {float(joint['axis_y']):.6f} {float(joint['axis_z']):.6f}</xyz></axis>"
    sdf_kind = "revolute" if str(joint["joint_type"]) == "continuous" else "fixed"
    return f'''    <joint name="{joint["semantic_key"]}" type="{sdf_kind}"><parent>{body_links[str(joint["parent_body_key"])]}</parent><child>{body_links[str(joint["child_body_key"])]}</child>{axis}</joint>'''


def _concept_controllers_yaml(spec: dict[str, Any]) -> str:
    drive = _concept_drive_joints(spec)
    bodies = {str(body["semantic_key"]): body for body in spec["bodies"]}
    left = [str(joint["semantic_key"]) for joint in drive if joint["wheel_role"] == "left_drive"]
    right = [str(joint["semantic_key"]) for joint in drive if joint["wheel_role"] == "right_drive"]
    if not left or not right:
        return """controller_manager:
  ros__parameters:
    update_rate: 250
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
"""
    left_y = [
        abs(float(bodies[str(joint["child_body_key"])]["pose"]["y_m"]))
        for joint in drive
        if joint["wheel_role"] == "left_drive"
    ]
    right_y = [
        abs(float(bodies[str(joint["child_body_key"])]["pose"]["y_m"]))
        for joint in drive
        if joint["wheel_role"] == "right_drive"
    ]
    radii = [
        float(bodies[str(joint["child_body_key"])]["radius_m"])
        for joint in drive
        if bodies[str(joint["child_body_key"])]["primitive_type"] == "cylinder"
    ]
    separation = (
        (sum(left_y) / len(left_y) + sum(right_y) / len(right_y)) if left_y and right_y else 0.5
    )
    radius = sum(radii) / len(radii) if radii else 0.08
    return f"""controller_manager:
  ros__parameters:
    update_rate: 250
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    diff_drive_controller:
      type: diff_drive_controller/DiffDriveController

diff_drive_controller:
  ros__parameters:
    left_wheel_names: {json.dumps(left)}
    right_wheel_names: {json.dumps(right)}
    wheel_separation: {separation:.6f}
    wheel_radius: {radius:.6f}
    base_frame_id: base_link
    odom_frame_id: odom
    publish_rate: 50.0
    cmd_vel_timeout: 0.5
    use_stamped_vel: true
    enable_odom_tf: true
    robot_base_frame: base_link
    tf_frame_prefix_enable: false
"""


def _concept_launch_controllers(spec: dict[str, Any]) -> tuple[str, ...]:
    drive = _concept_drive_joints(spec)
    has_left = any(joint["wheel_role"] == "left_drive" for joint in drive)
    has_right = any(joint["wheel_role"] == "right_drive" for joint in drive)
    if has_left and has_right:
        return ("joint_state_broadcaster", "diff_drive_controller")
    return ("joint_state_broadcaster",)


def _concept_world(robot_name: str, template: str) -> str:
    extras = {
        "warehouse": "".join(
            f"""    <model name="shelf_{i}"><static>true</static><pose>{x} {y} 1.0 0 0 0</pose><link name="link"><visual name="visual"><geometry><box><size>0.25 2.5 2.0</size></box></geometry></visual><collision name="collision"><geometry><box><size>0.25 2.5 2.0</size></box></geometry></collision></link></model>\n"""
            for i, (x, y) in enumerate(((-2, 1.8), (2, 1.8), (-2, -1.8), (2, -1.8)), start=1)
        ),
        "engineering_lab": '    <model name="bench"><static>true</static><pose>2.0 0 0.45 0 0 0</pose><link name="link"><visual name="visual"><geometry><box><size>1.2 0.5 0.9</size></box></geometry></visual><collision name="collision"><geometry><box><size>1.2 0.5 0.9</size></box></geometry></collision></link></model>\n',
        "ramp_course": '    <model name="low_ramp"><static>true</static><pose>1.8 0 0.08 0 0.18 0</pose><link name="link"><visual name="visual"><geometry><box><size>1.0 1.2 0.16</size></box></geometry></visual><collision name="collision"><geometry><box><size>1.0 1.2 0.16</size></box></geometry></collision></link></model>\n',
        "rough_terrain": '    <model name="small_block"><static>true</static><pose>1.5 0.4 0.035 0 0 0</pose><link name="link"><visual name="visual"><geometry><box><size>0.25 0.25 0.07</size></box></geometry></visual><collision name="collision"><geometry><box><size>0.25 0.25 0.07</size></box></geometry></collision></link></model>\n',
    }.get(template, "")
    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="warehouse_inspection">
    <physics type="ode"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor><real_time_update_rate>250</real_time_update_rate></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <light name="ceiling_light" type="directional"><pose>0 0 6 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse><direction>-0.4 0.2 -1</direction></light>
    <model name="ground_plane"><static>true</static><link name="link"><collision name="collision"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></collision><visual name="visual"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></visual></link></model>
{extras}    <include><uri>model://{robot_name}</uri><pose>0 0 0 0 0 0</pose></include>
  </world>
</sdf>
"""


def _concept_validation_report(
    robot_name: str, desc_pkg: str, sim_pkg: str, spec: dict[str, Any], concept: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "hephaestus.ros_gazebo_report.v1",
        "status": "passed",
        "robot_name": robot_name,
        "description_package": desc_pkg,
        "simulation_package": sim_pkg,
        "body_count": len(spec["bodies"]),
        "joint_count": len(spec["joints"]),
        "sensor_count": len(spec["sensors"]),
        "bridged_topics": ["/clock", "/scan", "/imu"],
        "controllers": ["joint_state_broadcaster", "diff_drive_controller"]
        if spec["locomotion_mode"] != "static"
        else ["joint_state_broadcaster"],
        "cmd_vel_topic": "/diff_drive_controller/cmd_vel",
        "cmd_vel_type": "geometry_msgs/msg/TwistStamped",
        "odom_topic": "/diff_drive_controller/odom",
        "warnings": list(concept.get("warnings", [])),
    }


def _concept_root_body_key(spec: dict[str, Any]) -> str:
    children = {str(joint["child_body_key"]) for joint in spec["joints"]}
    for body in spec["bodies"]:
        key = str(body["semantic_key"])
        if key not in children:
            return key
    raise RosGazeboGenerationError("concept has no root body")


def _concept_drive_joints(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        joint
        for joint in spec["joints"]
        if joint.get("wheel_role") in {"left_drive", "right_drive"}
    ]


def _concept_inertia_diag(body: dict[str, Any]) -> tuple[float, float, float]:
    mass = float(body["mass_kg"])
    primitive = PrimitiveType(str(body["primitive_type"]))
    if primitive is PrimitiveType.BOX:
        x, y, z = float(body["size_x_m"]), float(body["size_y_m"]), float(body["size_z_m"])
        return _box_inertia(mass, x, y, z)
    if primitive is PrimitiveType.CYLINDER:
        return _cylinder_y_inertia(mass, float(body["radius_m"]), float(body["length_m"]))
    return _sphere_inertia(mass, float(body["radius_m"]))


def _urdf_joint_type(kind: str) -> str:
    return "continuous" if kind == "continuous" else "fixed"


def _urdf_axis(joint: dict[str, Any]) -> str:
    if str(joint["joint_type"]) != "continuous":
        return ""
    return f'    <axis xyz="{float(joint["axis_x"]):.6f} {float(joint["axis_y"]):.6f} {float(joint["axis_z"]):.6f}"/>\n'


def _absolute_body_poses(
    spec: dict[str, Any], bodies: dict[str, dict[str, Any]], root_key: str
) -> dict[str, dict[str, float]]:
    children = {str(joint["child_body_key"]): joint for joint in spec["joints"]}
    cache: dict[str, dict[str, float]] = {}

    def resolve(key: str) -> dict[str, float]:
        if key in cache:
            return cache[key]
        if key == root_key:
            pose = bodies[key]["pose"]
            cache[key] = {
                "x_m": float(pose["x_m"]),
                "y_m": float(pose["y_m"]),
                "z_m": float(pose["z_m"]),
                "roll_rad": float(pose.get("roll_rad", 0.0)),
                "pitch_rad": float(pose.get("pitch_rad", 0.0)),
                "yaw_rad": float(pose.get("yaw_rad", 0.0)),
            }
            return cache[key]
        joint = children[key]
        parent = resolve(str(joint["parent_body_key"]))
        origin = joint["origin"]
        cache[key] = {
            "x_m": parent["x_m"] + float(origin["x_m"]),
            "y_m": parent["y_m"] + float(origin["y_m"]),
            "z_m": parent["z_m"] + float(origin["z_m"]),
            "roll_rad": float(origin.get("roll_rad", 0.0)),
            "pitch_rad": float(origin.get("pitch_rad", 0.0)),
            "yaw_rad": float(origin.get("yaw_rad", 0.0)),
        }
        return cache[key]

    for key in bodies:
        resolve(key)
    return cache


def _relative_pose(pose: dict[str, Any], parent_pose: dict[str, Any]) -> dict[str, float]:
    return {
        "x_m": float(pose["x_m"]) - float(parent_pose["x_m"]),
        "y_m": float(pose["y_m"]) - float(parent_pose["y_m"]),
        "z_m": float(pose["z_m"]) - float(parent_pose["z_m"]),
        "roll_rad": float(pose.get("roll_rad", 0.0)),
        "pitch_rad": float(pose.get("pitch_rad", 0.0)),
        "yaw_rad": float(pose.get("yaw_rad", 0.0)),
    }


def _pose_xyz(pose: dict[str, Any]) -> str:
    return f"{float(pose['x_m']):.6f} {float(pose['y_m']):.6f} {float(pose['z_m']):.6f}"


def _pose_rpy(pose: dict[str, Any]) -> str:
    return f"{float(pose.get('roll_rad', 0.0)):.6f} {float(pose.get('pitch_rad', 0.0)):.6f} {float(pose.get('yaw_rad', 0.0)):.6f}"


def _pose_sdf(pose: dict[str, Any]) -> str:
    return f"{_pose_xyz(pose)} {_pose_rpy(pose)}"


def _cylinder_visual_origin(body: dict[str, Any]) -> str:
    if str(body["primitive_type"]) != "cylinder":
        return ""
    return '<origin xyz="0 0 0" rpy="1.570796326795 0 0"/>'


def _cylinder_sdf_pose(body: dict[str, Any]) -> str:
    if str(body["primitive_type"]) != "cylinder":
        return ""
    return "<pose>0 0 0 1.570796326795 0 0</pose>"


def _copy_meshes(stager: ArtifactStager, desc_pkg: str, sim_pkg: str, robot_name: str) -> None:
    for role in ("visual", "collision"):
        for name in ("chassis", "wheel", "caster"):
            source = stager.staging / f"cad/stl/{role}/{name}.stl"
            if not source.exists():
                raise RosGazeboGenerationError(f"CAD mesh missing before ROS generation: {source}")
            for target_rel in (
                f"ros2_ws/src/{desc_pkg}/meshes/{role}/{name}.stl",
                f"ros2_ws/src/{sim_pkg}/models/{robot_name}/meshes/{role}/{name}.stl",
                f"gazebo/models/{robot_name}/meshes/{role}/{name}.stl",
            ):
                target = stager.path(target_rel)
                shutil.copy2(source, target)


def _mesh_records(desc_pkg: str, sim_pkg: str, robot_name: str) -> tuple[str, ...]:
    records: list[str] = []
    for role in ("visual", "collision"):
        for name in ("chassis", "wheel", "caster"):
            records.extend(
                (
                    f"ros2_ws/src/{desc_pkg}/meshes/{role}/{name}.stl",
                    f"ros2_ws/src/{sim_pkg}/models/{robot_name}/meshes/{role}/{name}.stl",
                    f"gazebo/models/{robot_name}/meshes/{role}/{name}.stl",
                )
            )
    return tuple(records)


def _xacro(robot_name: str, desc_pkg: str, bringup_pkg: str, values: dict[str, Any]) -> str:
    urdf = _urdf(robot_name, desc_pkg, values)
    plugin = f"""
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>install/share/{bringup_pkg}/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>
"""
    return urdf.replace("</robot>\n", plugin + "</robot>\n")


def _urdf(robot_name: str, desc_pkg: str, values: dict[str, Any]) -> str:
    length = _value(values, "chassis_length")
    width = _value(values, "chassis_width")
    height = _value(values, "chassis_height")
    wheel_radius = _value(values, "wheel_radius")
    wheel_width = _value(values, "wheel_width")
    wheel_sep = _value(values, "wheel_separation")
    ground = _value(values, "ground_clearance")
    base_z = ground + height / 2.0
    wheel_x = -length * 0.22
    caster_x = length * 0.34
    left_y = wheel_sep / 2.0
    right_y = -wheel_sep / 2.0
    chassis_mass = _value(values, "chassis_mass")
    wheel_mass = _value(values, "wheel_mass")
    caster_mass = _value(values, "caster_mass")
    chassis_i = _box_inertia(chassis_mass, length, width, height)
    wheel_i = _cylinder_y_inertia(wheel_mass, wheel_radius, wheel_width)
    caster_i = _sphere_inertia(caster_mass, _value(values, "caster_radius"))
    return f"""<?xml version="1.0"?>
<robot name="{robot_name}">
  <link name="base_footprint"/>
  <joint name="base_footprint_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 {base_z:.6f}" rpy="0 0 0"/>
  </joint>
  <link name="base_link">
    {_inertial(chassis_mass, chassis_i)}
    <visual>
      <geometry><mesh filename="package://{desc_pkg}/meshes/visual/chassis.stl" scale="0.001 0.001 0.001"/></geometry>
    </visual>
    <collision>
      <geometry><box size="{length:.6f} {width:.6f} {height:.6f}"/></geometry>
    </collision>
  </link>
  {_wheel_link("left_wheel_link", wheel_mass, wheel_i, desc_pkg, wheel_radius, wheel_width)}
  {_wheel_joint("left_wheel_joint", "left_wheel_link", wheel_x, left_y, wheel_radius - base_z)}
  {_wheel_link("right_wheel_link", wheel_mass, wheel_i, desc_pkg, wheel_radius, wheel_width)}
  {_wheel_joint("right_wheel_joint", "right_wheel_link", wheel_x, right_y, wheel_radius - base_z)}
  <link name="caster_link">
    {_inertial(caster_mass, caster_i)}
    <visual><geometry><mesh filename="package://{desc_pkg}/meshes/visual/caster.stl" scale="0.001 0.001 0.001"/></geometry></visual>
    <collision><geometry><sphere radius="{_value(values, "caster_radius"):.6f}"/></geometry></collision>
  </link>
  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_link"/>
    <origin xyz="{caster_x:.6f} 0 {_value(values, "caster_radius") - base_z:.6f}" rpy="0 0 0"/>
  </joint>
  <link name="lidar_link"/>
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="{length * 0.18:.6f} 0 {height / 2.0 + 0.08:.6f}" rpy="0 0 0"/>
  </joint>
  <link name="imu_link"/>
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.02" rpy="0 0 0"/>
  </joint>
  <ros2_control name="GazeboSimSystem" type="system">
    <hardware>
      <plugin>gz_ros2_control/GazeboSimSystem</plugin>
    </hardware>
    {_control_joint("left_wheel_joint")}
    {_control_joint("right_wheel_joint")}
  </ros2_control>
</robot>
"""


def _wheel_link(
    name: str,
    mass: float,
    inertia: tuple[float, float, float],
    desc_pkg: str,
    radius: float,
    width: float,
) -> str:
    return f"""<link name="{name}">
    {_inertial(mass, inertia)}
    <visual><geometry><mesh filename="package://{desc_pkg}/meshes/visual/wheel.stl" scale="0.001 0.001 0.001"/></geometry></visual>
    <collision><origin xyz="0 0 0" rpy="1.570796326795 0 0"/><geometry><cylinder radius="{radius:.6f}" length="{width:.6f}"/></geometry></collision>
  </link>"""


def _wheel_joint(name: str, child: str, x: float, y: float, z: float) -> str:
    return f"""<joint name="{name}" type="continuous">
    <parent link="base_link"/>
    <child link="{child}"/>
    <origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>"""


def _control_joint(name: str) -> str:
    return f"""<joint name="{name}">
      <command_interface name="velocity"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>"""


def _inertial(mass: float, inertia: tuple[float, float, float]) -> str:
    return f"""<inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.9f}"/>
      <inertia ixx="{inertia[0]:.12f}" ixy="0" ixz="0" iyy="{inertia[1]:.12f}" iyz="0" izz="{inertia[2]:.12f}"/>
    </inertial>"""


def _sdf_model(robot_name: str, bringup_pkg: str, values: dict[str, Any]) -> str:
    length = _value(values, "chassis_length")
    width = _value(values, "chassis_width")
    height = _value(values, "chassis_height")
    wheel_radius = _value(values, "wheel_radius")
    wheel_width = _value(values, "wheel_width")
    wheel_sep = _value(values, "wheel_separation")
    ground = _value(values, "ground_clearance")
    base_z = ground + height / 2.0
    wheel_x = -length * 0.22
    caster_x = length * 0.34
    chassis_mass = _value(values, "chassis_mass")
    wheel_mass = _value(values, "wheel_mass")
    caster_mass = _value(values, "caster_mass")
    chassis_i = _box_inertia(chassis_mass, length, width, height)
    wheel_i = _cylinder_y_inertia(wheel_mass, wheel_radius, wheel_width)
    caster_i = _sphere_inertia(caster_mass, _value(values, "caster_radius"))
    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{robot_name}">
    <pose>0 0 {base_z:.6f} 0 0 0</pose>
    <link name="base_link">
      {_sdf_inertial(chassis_mass, chassis_i)}
      <visual name="chassis_visual"><geometry><mesh><uri>model://{robot_name}/meshes/visual/chassis.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry></visual>
      <collision name="chassis_collision"><geometry><box><size>{length:.6f} {width:.6f} {height:.6f}</size></box></geometry></collision>
      <sensor name="lidar" type="gpu_lidar">
        <pose>{length * 0.18:.6f} 0 {height / 2.0 + 0.08:.6f} 0 0 0</pose>
        <topic>/scan</topic>
        <update_rate>{_value(values, "lidar_update_rate"):.1f}</update_rate>
        <lidar><scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan><range><min>0.05</min><max>12.0</max><resolution>0.01</resolution></range><noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise></lidar>
      </sensor>
      <sensor name="imu" type="imu">
        <pose>0 0 0.02 0 0 0</pose>
        <topic>/imu</topic>
        <update_rate>{_value(values, "imu_update_rate"):.1f}</update_rate>
      </sensor>
    </link>
    {_sdf_wheel(robot_name, "left_wheel_link", "left_wheel_joint", wheel_x, wheel_sep / 2.0, wheel_radius - base_z, wheel_radius, wheel_width, wheel_mass, wheel_i)}
    {_sdf_wheel(robot_name, "right_wheel_link", "right_wheel_joint", wheel_x, -wheel_sep / 2.0, wheel_radius - base_z, wheel_radius, wheel_width, wheel_mass, wheel_i)}
    <link name="caster_link"><pose relative_to="base_link">{caster_x:.6f} 0 {_value(values, "caster_radius") - base_z:.6f} 0 0 0</pose>{_sdf_inertial(caster_mass, caster_i)}<visual name="caster_visual"><geometry><mesh><uri>model://{robot_name}/meshes/visual/caster.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry></visual><collision name="caster_collision"><geometry><sphere><radius>{_value(values, "caster_radius"):.6f}</radius></sphere></geometry></collision></link>
    <joint name="caster_joint" type="fixed"><parent>base_link</parent><child>caster_link</child></joint>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>install/share/{bringup_pkg}/config/controllers.yaml</parameters>
    </plugin>
  </model>
</sdf>
"""


def _sdf_inertial(mass: float, inertia: tuple[float, float, float]) -> str:
    return f"""<inertial><pose>0 0 0 0 0 0</pose><mass>{mass:.9f}</mass><inertia><ixx>{inertia[0]:.12f}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{inertia[1]:.12f}</iyy><iyz>0</iyz><izz>{inertia[2]:.12f}</izz></inertia></inertial>"""


def _sdf_wheel(
    robot_name: str,
    name: str,
    joint: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    width: float,
    mass: float,
    inertia: tuple[float, float, float],
) -> str:
    return f"""<link name="{name}"><pose relative_to="base_link">{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>{_sdf_inertial(mass, inertia)}<visual name="{name}_visual"><geometry><mesh><uri>model://{robot_name}/meshes/visual/wheel.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry></visual><collision name="{name}_collision"><pose>0 0 0 1.570796326795 0 0</pose><geometry><cylinder><radius>{radius:.6f}</radius><length>{width:.6f}</length></cylinder></geometry></collision></link><joint name="{joint}" type="revolute"><parent>base_link</parent><child>{name}</child><axis><xyz>0 1 0</xyz></axis></joint>"""


def _world(robot_name: str) -> str:
    shelves = "\n".join(
        f"""    <model name="shelf_{i}"><static>true</static><pose>{x} {y} 1.0 0 0 0</pose><link name="link"><visual name="visual"><geometry><box><size>0.25 2.5 2.0</size></box></geometry></visual><collision name="collision"><geometry><box><size>0.25 2.5 2.0</size></box></geometry></collision></link></model>"""
        for i, (x, y) in enumerate(((-2, 1.8), (2, 1.8), (-2, -1.8), (2, -1.8)), start=1)
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="warehouse_inspection">
    <physics type="ode"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor><real_time_update_rate>250</real_time_update_rate></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <light name="ceiling_light" type="directional"><pose>0 0 6 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse><direction>-0.4 0.2 -1</direction></light>
    <model name="ground_plane"><static>true</static><link name="link"><collision name="collision"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></collision><visual name="visual"><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></visual></link></model>
{shelves}
    <include><uri>model://{robot_name}</uri><pose>0 0 0 0 0 0</pose></include>
  </world>
</sdf>
"""


def _controllers_yaml(robot_name: str, values: dict[str, Any]) -> str:
    return f"""controller_manager:
  ros__parameters:
    update_rate: {int(_value(values, "simulation_update_rate"))}
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    diff_drive_controller:
      type: diff_drive_controller/DiffDriveController

diff_drive_controller:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]
    wheel_separation: {_value(values, "wheel_separation"):.6f}
    wheel_radius: {_value(values, "wheel_radius"):.6f}
    base_frame_id: base_link
    odom_frame_id: odom
    publish_rate: 50.0
    cmd_vel_timeout: 0.5
    use_stamped_vel: false
    enable_odom_tf: true
    robot_base_frame: base_link
    tf_frame_prefix_enable: false
    topic_name: /{robot_name}/cmd_vel
"""


def _bridge_yaml(robot_name: str) -> str:
    del robot_name
    return """- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
  qos_profile: CLOCK
- ros_topic_name: "/scan"
  gz_topic_name: "/scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS
- ros_topic_name: "/imu"
  gz_topic_name: "/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS
"""


def _rsp_launch(desc_pkg: str, robot_name: str) -> str:
    return f"""from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    xacro_file = PathJoinSubstitution([FindPackageShare('{desc_pkg}'), 'urdf', '{robot_name}.urdf.xacro'])
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{{'robot_description': Command(['xacro ', xacro_file]), 'use_sim_time': True}}],
        )
    ])
"""


def _gazebo_launch(
    desc_pkg: str,
    bringup_pkg: str,
    sim_pkg: str,
    robot_name: str,
    *,
    headless: bool,
    controller_names: tuple[str, ...],
) -> str:
    del desc_pkg, robot_name
    if headless:
        gazebo_launch = """        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch), launch_arguments={'gz_args': ['-r -s ', world], 'on_exit_shutdown': 'true'}.items()),"""
    else:
        gazebo_launch = """        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch), launch_arguments={'gz_args': ['-r ', world], 'on_exit_shutdown': 'true'}.items()),"""
    controller_args = ", ".join(
        repr(argument)
        for argument in (
            *controller_names,
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "30",
            "--service-call-timeout",
            "30",
            "--switch-timeout",
            "30",
            "--activate-as-group",
        )
    )
    gz_launch_file = "gz_sim.launch.py"
    return f"""from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_world = PathJoinSubstitution([FindPackageShare('{sim_pkg}'), 'worlds', 'warehouse_inspection.sdf'])
    world = LaunchConfiguration('world')
    bridge_config = PathJoinSubstitution([FindPackageShare('{bringup_pkg}'), 'config', 'bridge.yaml'])
    rsp_launch = PathJoinSubstitution([FindPackageShare('{bringup_pkg}'), 'launch', 'robot_state_publisher.launch.py'])
    gz_launch = PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', '{gz_launch_file}'])
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=default_world),
        SetEnvironmentVariable(name='SDF_PATH', value=[FindPackageShare('{sim_pkg}'), '/models']),
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=[FindPackageShare('{sim_pkg}'), '/models']),
{gazebo_launch}
        IncludeLaunchDescription(PythonLaunchDescriptionSource(rsp_launch)),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen', parameters=[{{'config_file': bridge_config}}]),
        Node(package='controller_manager', executable='spawner', output='screen', arguments=[{controller_args}]),
    ])
"""


def _description_package_xml(name: str) -> str:
    return _package_xml(name, ("xacro", "robot_state_publisher"))


def _bringup_package_xml(name: str, desc_pkg: str, sim_pkg: str) -> str:
    return _package_xml(
        name,
        (
            desc_pkg,
            sim_pkg,
            "ros_gz_bridge",
            "ros_gz_sim",
            "controller_manager",
            "gz_ros2_control",
        ),
    )


def _simulation_package_xml(name: str, desc_pkg: str) -> str:
    return _package_xml(name, (desc_pkg, "ros_gz_sim"))


def _package_xml(name: str, exec_depends: tuple[str, ...]) -> str:
    deps = "\n".join(f"  <exec_depend>{dep}</exec_depend>" for dep in exec_depends)
    return f"""<?xml version="1.0"?>
<package format="3">
  <name>{name}</name>
  <version>0.0.1</version>
  <description>Generated Hephaestus Milestone 3.0 artifact package.</description>
  <maintainer email="maintainer@example.com">Hephaestus</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_python</buildtool_depend>
{deps}
  <export><build_type>ament_python</build_type></export>
</package>
"""


def _setup_py(name: str) -> str:
    return f"""from glob import glob
from pathlib import Path
from setuptools import setup

package_name = '{name}'
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
for pattern, destination in (
    ('urdf/*', 'urdf'),
    ('config/*', 'config'),
    ('launch/*', 'launch'),
    ('worlds/*', 'worlds'),
    ('meshes/visual/*', 'meshes/visual'),
    ('meshes/collision/*', 'meshes/collision'),
):
    matches = glob(pattern)
    if matches:
        data_files.append(('share/' + package_name + '/' + destination, matches))
models_root = Path('models')
if models_root.exists():
    for item in sorted(models_root.rglob('*')):
        if item.is_file():
            data_files.append(('share/' + package_name + '/' + str(item.parent), [str(item)]))

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hephaestus',
    maintainer_email='maintainer@example.com',
    description='Generated Hephaestus package.',
    license='Apache-2.0',
)
"""


def _setup_cfg(name: str) -> str:
    return f"""[develop]
script_dir=$base/lib/{name}
[install]
install_scripts=$base/lib/{name}
"""


def _model_config(robot_name: str) -> str:
    return f"""<?xml version="1.0"?>
<model>
  <name>{robot_name}</name>
  <version>0.0.1</version>
  <sdf version="1.10">model.sdf</sdf>
  <author><name>Hephaestus</name></author>
  <description>Generated differential-drive warehouse inspection robot.</description>
</model>
"""


def _validation_report(
    robot_name: str, desc_pkg: str, sim_pkg: str, values: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "hephaestus.ros_gazebo_report.v1",
        "status": "passed",
        "robot_name": robot_name,
        "description_package": desc_pkg,
        "simulation_package": sim_pkg,
        "frames": [
            "base_footprint",
            "base_link",
            "left_wheel_link",
            "right_wheel_link",
            "lidar_link",
            "imu_link",
        ],
        "joints": {"left_wheel_joint": [0, 1, 0], "right_wheel_joint": [0, 1, 0]},
        "bridged_topics": ["/clock", "/scan", "/imu"],
        "controllers": ["joint_state_broadcaster", "diff_drive_controller"],
        "wheel_radius_m": _value(values, "wheel_radius"),
        "wheel_separation_m": _value(values, "wheel_separation"),
    }


def _role_for_path(path: str) -> str:
    if path.endswith(".urdf.xacro"):
        return "xacro_robot_description"
    if path.endswith(".urdf"):
        return "urdf_robot_description"
    if path.endswith(".sdf") and "/worlds/" in path:
        return "gazebo_world_sdf"
    if path.endswith(".sdf"):
        return "gazebo_model_sdf"
    if path.endswith("controllers.yaml"):
        return "ros2_control_config"
    if path.endswith("bridge.yaml"):
        return "ros_gz_bridge_config"
    if "/launch/" in path:
        return "ros2_launch"
    return "ros2_workspace_file"


def _value(values: dict[str, Any], key: str) -> float:
    return float(values[key]["value"])


def _box_inertia(mass: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    return (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)


def _cylinder_y_inertia(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    return (
        mass * (3 * radius * radius + length * length) / 12,
        0.5 * mass * radius * radius,
        mass * (3 * radius * radius + length * length) / 12,
    )


def _sphere_inertia(mass: float, radius: float) -> tuple[float, float, float]:
    inertia = 0.4 * mass * radius * radius
    return (inertia, inertia, inertia)


def _validate_xml(path: Path) -> None:
    try:
        ET.parse(path)
    except ET.ParseError as exc:
        raise RosGazeboGenerationError(f"generated XML did not parse: {path.name}") from exc
