"""Non-user-facing model fixtures for tests."""

# ruff: noqa: E501

from __future__ import annotations

from hephaestus.model import (
    Actuator,
    ActuatorKind,
    Assumption,
    Body,
    BusKind,
    CommunicationBus,
    CommunicationInterface,
    ComputeDevice,
    ElectricalDevice,
    ElectricalPort,
    EngineeringDecision,
    EngineeringProject,
    Entity,
    EntityKind,
    FieldOfView,
    Frame,
    Geometry,
    GeometryBinding,
    GeometryKind,
    GeometryRole,
    Joint,
    JointKind,
    MassProperties,
    MassProvenance,
    Material,
    MeshFormat,
    NoiseModel,
    Pose,
    PowerConnection,
    ProjectMetadata,
    Provenance,
    ProvenanceSource,
    QuantityRange,
    Reference,
    Requirement,
    RequirementKind,
    Sensor,
    SensorModality,
    SimulationScenario,
    SoftwareComponent,
    Transform,
    Vector3,
    identity_pose,
    q,
    zero_vector,
)


def representative_project() -> EngineeringProject:
    user = Provenance(ProvenanceSource.USER, "test fixture")
    calc = Provenance(ProvenanceSource.CALCULATED, "fixture calculation")

    frames = (
        Frame("frame:world", "World"),
        Frame("frame:base", "Base frame", Reference("entity", "entity:base")),
        Frame("frame:mast", "Sensor mast", Reference("entity", "entity:mast")),
        Frame("frame:camera", "Camera frame", Reference("sensor", "sensor:camera")),
        Frame("frame:imu", "IMU frame", Reference("sensor", "sensor:imu")),
    )
    entities = (
        Entity(
            "entity:base",
            "Mobile platform base",
            EntityKind.ASSEMBLY,
            frame_ref=Reference("frame", "frame:base"),
        ),
        Entity(
            "entity:mast",
            "Sensor mast",
            EntityKind.BODY,
            parent_entity=Reference("entity", "entity:base"),
            frame_ref=Reference("frame", "frame:mast"),
        ),
    )

    material = Material(
        "material:aluminum", "Aluminum", density=q(2700.0, "kg/m^3", "density"), provenance=user
    )
    geometries = (
        Geometry.box(
            "geometry:base_box",
            q(0.6, "m", "length"),
            q(0.42, "m", "length"),
            q(0.18, "m", "length"),
            material_ref=Reference("component", "material:aluminum"),
        ),
        Geometry.mesh("geometry:cover_mesh", "assets/base_cover.glb", MeshFormat.GLB),
        Geometry(
            "geometry:mast_cylinder",
            GeometryKind.CYLINDER,
            radius=q(20.0, "mm", "length"),
            length=q(0.55, "m", "length"),
        ),
    )

    inertia = (
        (
            q(0.08, "kg*m^2", "inertia"),
            q(0.0, "kg*m^2", "inertia"),
            q(0.0, "kg*m^2", "inertia"),
        ),
        (
            q(0.0, "kg*m^2", "inertia"),
            q(0.12, "kg*m^2", "inertia"),
            q(0.0, "kg*m^2", "inertia"),
        ),
        (
            q(0.0, "kg*m^2", "inertia"),
            q(0.0, "kg*m^2", "inertia"),
            q(0.15, "kg*m^2", "inertia"),
        ),
    )
    mass_props = MassProperties(
        q(18.0, "kg", "mass"), zero_vector(), inertia, MassProvenance.EXPLICIT
    )
    bodies = (
        Body(
            "body:base",
            Reference("entity", "entity:base"),
            Reference("frame", "frame:base"),
            visual_geometry=(
                GeometryBinding(Reference("component", "geometry:base_box"), GeometryRole.VISUAL),
                GeometryBinding(Reference("component", "geometry:cover_mesh"), GeometryRole.VISUAL),
            ),
            collision_geometry=(
                GeometryBinding(
                    Reference("component", "geometry:base_box"), GeometryRole.COLLISION
                ),
            ),
            mass_properties=mass_props,
        ),
        Body(
            "body:mast",
            Reference("entity", "entity:mast"),
            Reference("frame", "frame:mast"),
            visual_geometry=(
                GeometryBinding(
                    Reference("component", "geometry:mast_cylinder"), GeometryRole.VISUAL
                ),
            ),
            collision_geometry=(
                GeometryBinding(
                    Reference("component", "geometry:mast_cylinder"), GeometryRole.COLLISION
                ),
            ),
        ),
    )

    transforms = (
        Transform(
            "transform:world_base",
            Reference("frame", "frame:world"),
            Reference("frame", "frame:base"),
            identity_pose("frame:world"),
            user,
        ),
        Transform(
            "transform:base_mast",
            Reference("frame", "frame:base"),
            Reference("frame", "frame:mast"),
            Pose(
                Reference("frame", "frame:base"),
                Vector3(q(0.1, "m", "length"), q(0.0, "m", "length"), q(0.35, "m", "length")),
            ),
            calc,
        ),
        Transform(
            "transform:mast_camera",
            Reference("frame", "frame:mast"),
            Reference("frame", "frame:camera"),
            Pose(
                Reference("frame", "frame:mast"),
                Vector3(q(0.0, "m", "length"), q(0.0, "m", "length"), q(0.18, "m", "length")),
            ),
            user,
        ),
        Transform(
            "transform:base_imu",
            Reference("frame", "frame:base"),
            Reference("frame", "frame:imu"),
            Pose(Reference("frame", "frame:base"), zero_vector()),
            user,
        ),
    )

    joint = Joint(
        "joint:mast_mount",
        "Mast mount",
        JointKind.FIXED,
        Reference("body", "body:base"),
        Reference("body", "body:mast"),
    )
    actuator = Actuator(
        "actuator:left_drive",
        "Left drive actuator",
        ActuatorKind.ROTARY,
        controlled_joint=Reference("joint", "joint:mast_mount"),
        limits={"speed": q(4.0, "m/s", "speed")},
    )

    electrical_devices = (
        ElectricalDevice(
            "electrical:battery",
            "Battery",
            "battery",
            power_source=True,
            voltage_nominal=q(24.0, "V", "voltage"),
            power_limit=q(600.0, "W", "power"),
        ),
        ElectricalDevice(
            "electrical:motor_driver",
            "Motor driver",
            "motor_driver",
            entity_ref=Reference("entity", "entity:base"),
        ),
        ElectricalDevice(
            "electrical:compute",
            "Compute board",
            "compute_board",
            entity_ref=Reference("entity", "entity:base"),
            firmware_reference="firmware/main",
        ),
    )
    ports = (
        ElectricalPort(
            "port:battery_out",
            Reference("electrical_device", "electrical:battery"),
            "out",
            voltage_limit=q(24.0, "V", "voltage"),
            current_limit=q(20.0, "A", "current"),
        ),
        ElectricalPort(
            "port:driver_power", Reference("electrical_device", "electrical:motor_driver"), "power"
        ),
        ElectricalPort(
            "port:compute_power", Reference("electrical_device", "electrical:compute"), "power"
        ),
    )
    power = (
        PowerConnection(
            "power:battery_driver",
            Reference("electrical_port", "port:battery_out"),
            Reference("electrical_port", "port:driver_power"),
            voltage=q(24.0, "V", "voltage"),
        ),
        PowerConnection(
            "power:battery_compute",
            Reference("electrical_port", "port:battery_out"),
            Reference("electrical_port", "port:compute_power"),
            voltage=q(24.0, "V", "voltage"),
        ),
    )

    bus = CommunicationBus("bus:can0", "CAN bus", BusKind.CAN)
    interfaces = (
        CommunicationInterface(
            "interface:driver_can",
            Reference("electrical_device", "electrical:motor_driver"),
            Reference("bus", "bus:can0"),
            "driver CAN",
            update_rate_hz=100.0,
        ),
        CommunicationInterface(
            "interface:compute_can",
            Reference("compute_device", "compute:main"),
            Reference("bus", "bus:can0"),
            "compute CAN",
            update_rate_hz=100.0,
        ),
    )
    compute = ComputeDevice(
        "compute:main",
        "Main compute",
        Reference("electrical_device", "electrical:compute"),
        processor="arm64",
        interface_refs=(Reference("component", "interface:compute_can"),),
        firmware_reference="firmware/main",
    )

    camera = Sensor(
        "sensor:camera",
        "Forward camera",
        SensorModality.CAMERA,
        Reference("frame", "frame:camera"),
        update_rate_hz=30.0,
        latency=q(0.02, "s", "time"),
        field_of_view=FieldOfView(q(90.0, "deg", "angle"), q(60.0, "deg", "angle")),
        range=QuantityRange(minimum=q(0.2, "m", "length"), maximum=q(12.0, "m", "length")),
        resolution=(1280, 720),
        vendor_reference="fixture-camera",
    )
    imu = Sensor(
        "sensor:imu",
        "IMU",
        SensorModality.IMU,
        Reference("frame", "frame:imu"),
        update_rate_hz=200.0,
        noise_model=NoiseModel("gaussian", {"angular_random_walk": q(0.01, "rad", "angle")}),
    )

    control = SoftwareComponent(
        "software:base_controller",
        "Base controller",
        "control",
        compute_ref=Reference("compute_device", "compute:main"),
        input_refs=(Reference("sensor", "sensor:imu"),),
        output_refs=(Reference("actuator", "actuator:left_drive"),),
        timing={"period": q(0.01, "s", "time")},
    )
    scenario = SimulationScenario(
        "scenario:flat_ground",
        "Flat ground smoke scenario",
        environment="flat indoor floor",
        duration=q(30.0, "s", "time"),
        initial_state_refs=(Reference("entity", "entity:base"),),
    )

    return EngineeringProject(
        id="project:fixture",
        metadata=ProjectMetadata(
            "Representative Fixture", "Broad engineering model fixture for tests."
        ),
        requirements=(
            Requirement(
                "requirement:payload",
                RequirementKind.PAYLOAD,
                "Carry a 5 kg payload.",
                value=QuantityRange(minimum=q(5.0, "kg", "mass")),
                provenance=user,
            ),
            Requirement(
                "requirement:speed",
                RequirementKind.SPEED,
                "Reach a walking-speed test velocity.",
                value=QuantityRange(target=q(1.5, "m/s", "speed")),
                provenance=user,
            ),
        ),
        assumptions=(
            Assumption("assumption:indoor", "Operation is evaluated on a dry indoor floor.", user),
        ),
        entities=entities,
        frames=frames,
        transforms=transforms,
        materials=(material,),
        geometries=geometries,
        bodies=bodies,
        joints=(joint,),
        actuators=(actuator,),
        sensors=(camera, imu),
        electrical_devices=electrical_devices,
        electrical_ports=ports,
        power_connections=power,
        communication_buses=(bus,),
        communication_interfaces=interfaces,
        compute_devices=(compute,),
        software_components=(control,),
        simulation_scenarios=(scenario,),
        engineering_decisions=(
            EngineeringDecision(
                "decision:canonical_model",
                "Use one backend-agnostic engineering project model.",
                "Tests need one source of truth for linked engineering views.",
                user,
            ),
        ),
    )


def blocky_yellow_warehouse_utility_robot_spec():
    """Return a deterministic non-infringing blocky yellow warehouse robot concept."""

    from hephaestus.concept_robot import (
        ConceptBodySpec,
        ConceptColor,
        ConceptJointLimit,
        ConceptJointSpec,
        ConceptJointType,
        ConceptPose,
        ConceptRobotSpec,
        ConceptSensorSpec,
        ConceptSensorType,
        ConceptWorld,
        FunctionalRole,
        LocomotionMode,
        PrimitiveType,
        WheelRole,
    )

    yellow = ConceptColor(r=0.98, g=0.78, b=0.08, a=1.0)
    dark = ConceptColor(r=0.04, g=0.045, b=0.05, a=1.0)
    blue = ConceptColor(r=0.12, g=0.28, b=0.38, a=1.0)
    bodies = (
        ConceptBodySpec(
            semantic_key="base",
            display_name="Boxy lower chassis",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.72,
            size_y_m=0.46,
            size_z_m=0.20,
            pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.16),
            mass_kg=18.0,
            visual_color=yellow,
            functional_role=FunctionalRole.CHASSIS,
        ),
        ConceptBodySpec(
            semantic_key="left_wheel",
            display_name="Left practical drive wheel",
            primitive_type=PrimitiveType.CYLINDER,
            radius_m=0.09,
            length_m=0.05,
            pose=ConceptPose(x_m=-0.16, y_m=0.285, z_m=-0.07),
            mass_kg=1.2,
            visual_color=dark,
            functional_role=FunctionalRole.DRIVE_WHEEL,
        ),
        ConceptBodySpec(
            semantic_key="right_wheel",
            display_name="Right practical drive wheel",
            primitive_type=PrimitiveType.CYLINDER,
            radius_m=0.09,
            length_m=0.05,
            pose=ConceptPose(x_m=-0.16, y_m=-0.285, z_m=-0.07),
            mass_kg=1.2,
            visual_color=dark,
            functional_role=FunctionalRole.DRIVE_WHEEL,
        ),
        ConceptBodySpec(
            semantic_key="rear_caster",
            display_name="Rear caster ball",
            primitive_type=PrimitiveType.SPHERE,
            radius_m=0.04,
            pose=ConceptPose(x_m=0.24, y_m=0.0, z_m=-0.12),
            mass_kg=0.4,
            visual_color=dark,
            functional_role=FunctionalRole.PASSIVE_WHEEL,
        ),
        ConceptBodySpec(
            semantic_key="left_side_panel",
            display_name="Left non-contact side panel",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.58,
            size_y_m=0.035,
            size_z_m=0.12,
            pose=ConceptPose(x_m=-0.03, y_m=0.335, z_m=0.0),
            mass_kg=0.6,
            visual_color=dark,
            functional_role=FunctionalRole.DECORATIVE,
        ),
        ConceptBodySpec(
            semantic_key="right_side_panel",
            display_name="Right non-contact side panel",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.58,
            size_y_m=0.035,
            size_z_m=0.12,
            pose=ConceptPose(x_m=-0.03, y_m=-0.335, z_m=0.0),
            mass_kg=0.6,
            visual_color=dark,
            functional_role=FunctionalRole.DECORATIVE,
        ),
        ConceptBodySpec(
            semantic_key="torso",
            display_name="Raised utility torso",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.34,
            size_y_m=0.30,
            size_z_m=0.28,
            pose=ConceptPose(x_m=0.08, y_m=0.0, z_m=0.24),
            mass_kg=4.2,
            visual_color=yellow,
            functional_role=FunctionalRole.PAYLOAD,
        ),
        ConceptBodySpec(
            semantic_key="left_arm",
            display_name="Left fixed side arm",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.20,
            size_y_m=0.04,
            size_z_m=0.05,
            pose=ConceptPose(x_m=-0.02, y_m=0.205, z_m=0.0),
            mass_kg=0.35,
            visual_color=yellow,
            functional_role=FunctionalRole.ARM,
        ),
        ConceptBodySpec(
            semantic_key="right_arm",
            display_name="Right fixed side arm",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.20,
            size_y_m=0.04,
            size_z_m=0.05,
            pose=ConceptPose(x_m=-0.02, y_m=-0.205, z_m=0.0),
            mass_kg=0.35,
            visual_color=yellow,
            functional_role=FunctionalRole.ARM,
        ),
        ConceptBodySpec(
            semantic_key="sensor_pillar",
            display_name="Blocky sensor mast",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.10,
            size_y_m=0.08,
            size_z_m=0.26,
            pose=ConceptPose(x_m=0.08, y_m=0.0, z_m=0.24),
            mass_kg=1.0,
            visual_color=yellow,
            functional_role=FunctionalRole.SENSOR_MOUNT,
        ),
        ConceptBodySpec(
            semantic_key="left_eye_housing",
            display_name="Left sensor housing",
            primitive_type=PrimitiveType.CYLINDER,
            radius_m=0.045,
            length_m=0.055,
            pose=ConceptPose(x_m=0.03, y_m=0.065, z_m=0.16),
            mass_kg=0.25,
            visual_color=blue,
            functional_role=FunctionalRole.SENSOR_HOUSING,
        ),
        ConceptBodySpec(
            semantic_key="right_eye_housing",
            display_name="Right sensor housing",
            primitive_type=PrimitiveType.CYLINDER,
            radius_m=0.045,
            length_m=0.055,
            pose=ConceptPose(x_m=0.03, y_m=-0.065, z_m=0.16),
            mass_kg=0.25,
            visual_color=blue,
            functional_role=FunctionalRole.SENSOR_HOUSING,
        ),
        ConceptBodySpec(
            semantic_key="lidar_puck",
            display_name="Top lidar puck",
            primitive_type=PrimitiveType.CYLINDER,
            radius_m=0.05,
            length_m=0.04,
            pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.27),
            mass_kg=0.5,
            visual_color=dark,
            functional_role=FunctionalRole.SENSOR_HOUSING,
        ),
        ConceptBodySpec(
            semantic_key="imu_block",
            display_name="Internal IMU block",
            primitive_type=PrimitiveType.BOX,
            size_x_m=0.05,
            size_y_m=0.04,
            size_z_m=0.02,
            pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.04),
            mass_kg=0.08,
            visual_color=dark,
            functional_role=FunctionalRole.SENSOR_MOUNT,
        ),
    )
    no_limits = ConceptJointLimit()
    joints = [
        ConceptJointSpec(
            semantic_key="left_wheel_joint",
            parent_body_key="base",
            child_body_key="left_wheel",
            joint_type=ConceptJointType.CONTINUOUS,
            origin=ConceptPose(x_m=-0.16, y_m=0.285, z_m=-0.07),
            axis_y=1.0,
            axis_z=0.0,
            limits=no_limits,
            wheel_role=WheelRole.LEFT_DRIVE,
        ),
        ConceptJointSpec(
            semantic_key="right_wheel_joint",
            parent_body_key="base",
            child_body_key="right_wheel",
            joint_type=ConceptJointType.CONTINUOUS,
            origin=ConceptPose(x_m=-0.16, y_m=-0.285, z_m=-0.07),
            axis_y=1.0,
            axis_z=0.0,
            limits=no_limits,
            wheel_role=WheelRole.RIGHT_DRIVE,
        ),
        ConceptJointSpec(
            semantic_key="rear_caster_joint",
            parent_body_key="base",
            child_body_key="rear_caster",
            joint_type=ConceptJointType.REVOLUTE,
            origin=ConceptPose(x_m=0.24, y_m=0.0, z_m=-0.12),
            axis_z=1.0,
            limits=no_limits,
            wheel_role=WheelRole.PASSIVE,
        ),
    ]
    fixed_parent = {
        "left_side_panel": "base",
        "right_side_panel": "base",
        "torso": "base",
        "left_arm": "torso",
        "right_arm": "torso",
        "sensor_pillar": "torso",
        "left_eye_housing": "sensor_pillar",
        "right_eye_housing": "sensor_pillar",
        "lidar_puck": "sensor_pillar",
        "imu_block": "base",
    }
    by_key = {body.semantic_key: body for body in bodies}
    joints.extend(
        ConceptJointSpec(
            semantic_key=f"{key}_mount",
            parent_body_key=parent,
            child_body_key=key,
            joint_type=ConceptJointType.FIXED,
            origin=by_key[key].pose,
            limits=no_limits,
        )
        for key, parent in fixed_parent.items()
    )
    sensors = (
        ConceptSensorSpec(
            semantic_key="front_lidar",
            sensor_type=ConceptSensorType.LIDAR_2D,
            parent_body_key="lidar_puck",
            pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.02),
            update_rate_hz=10.0,
            topic="/scan",
        ),
        ConceptSensorSpec(
            semantic_key="body_imu",
            sensor_type=ConceptSensorType.IMU,
            parent_body_key="imu_block",
            pose=ConceptPose(x_m=0.0, y_m=0.0, z_m=0.0),
            update_rate_hz=50.0,
            topic="/imu",
        ),
    )
    return ConceptRobotSpec(
        spec_version="hephaestus.concept_robot.v1",
        robot_name="Blocky Yellow Warehouse Utility Robot",
        design_summary="Move inspection payloads through dry indoor warehouse aisles.",
        locomotion_mode=LocomotionMode.DIFFERENTIAL_DRIVE,
        bodies=bodies,
        joints=tuple(joints),
        sensors=sensors,
        world_template=ConceptWorld.WAREHOUSE,
        assumptions=(
            "Decorative side panels are fixed non-contact panels, not tracked drivetrains.",
            "All parts are original primitive geometry with no character likeness.",
        ),
        engineering_warnings=(
            "Tall torso and mast may increase tip risk during aggressive maneuvers.",
        ),
    )


def edited_blocky_yellow_warehouse_utility_robot_spec():
    """Return a deterministic edit of the concept robot fixture."""

    from hephaestus.concept_robot import ConceptBodySpec

    spec = blocky_yellow_warehouse_utility_robot_spec()
    bodies = []
    for body in spec.bodies:
        data = body.model_dump()
        if body.semantic_key == "torso":
            data.update({"size_z_m": 0.36, "mass_kg": 4.8})
            data["pose"] = {**body.pose.model_dump(), "z_m": 0.28}
        if body.semantic_key == "sensor_pillar":
            data["pose"] = {**body.pose.model_dump(), "z_m": 0.30}
        if body.semantic_key == "left_eye_housing":
            data["pose"] = {**body.pose.model_dump(), "z_m": 0.20}
        if body.semantic_key == "right_eye_housing":
            data["pose"] = {**body.pose.model_dump(), "z_m": 0.20}
        if body.semantic_key == "lidar_puck":
            data["pose"] = {**body.pose.model_dump(), "z_m": 0.38}
        bodies.append(ConceptBodySpec.model_validate(data))
    return spec.model_copy(
        update={
            "design_summary": (
                "Move inspection payloads with a taller torso and raised LiDAR mast."
            ),
            "bodies": tuple(bodies),
        }
    )


def blocky_yellow_warehouse_utility_robot_project() -> EngineeringProject:
    """Compile the deterministic generic concept robot fixture into an EngineeringProject."""

    from hephaestus.concept_robot import compile_concept_robot_design

    return compile_concept_robot_design(
        EngineeringProject.empty(name="Blocky Yellow Warehouse Utility Robot"),
        blocky_yellow_warehouse_utility_robot_spec(),
    ).project
