# Safety Notice

## Experimental engineering software

Hephaestus AI is experimental robotics engineering and simulation software. It
can generate incomplete, incorrect, unstable, physically implausible, or unsafe
models, code, parameters, commands, explanations, and recommendations.

Outputs must be independently reviewed and validated by a qualified person
before they are relied upon.

## Simulation-only movement in this release

The official release described by this notice permits local movement commands
only for a Gazebo simulation owned by the active Hephaestus session and isolated
from ordinary external ROS graphs.

Hephaestus movement commands are not authorization to control a physical robot,
vehicle, actuator, industrial machine, medical device, weapon, or other
real-world system.

The simulation-only checks are defense-in-depth software controls. They are not
a certified safety boundary and must not be treated as an emergency stop,
safety PLC, interlock, collision-avoidance system, or functional-safety
mechanism.

## Generated physics and engineering data

Generated mass, inertia, friction, damping, collision geometry, joint limits,
controller gains, actuator assumptions, sensor placement, battery estimates,
loads, dimensions, tolerances, and material properties may be wrong.

A Gazebo simulation is an approximation. Successful simulated behavior does not
establish that a design is safe, stable, controllable, manufacturable, lawful,
or suitable for real-world operation.

Disabling or simplifying physics can make simulation results less
representative of real behavior. Hephaestus should warn when a user requests
such simplification, but the user remains responsible for interpreting the
result.

## User responsibilities

Users are responsible for:

- inspecting generated source and artifacts;
- validating units, coordinate frames, geometry, inertia, collisions, limits,
  controllers, and environmental assumptions;
- maintaining independent backups and version control;
- using appropriate protective equipment and engineering review;
- complying with laws, standards, licenses, workplace rules, and export
  controls; and
- implementing independent real-world emergency stops and safety systems.

Do not use Hephaestus as the sole basis for a safety-critical decision.

## Movement safeguards

Local simulated movement commands use bounded speeds and durations and attempt
to publish a zero-velocity command after completion, interruption, or failure.
Software, middleware, operating-system, controller, network, or simulator
failures may prevent a stop command from being delivered.

The `/stop` command is a convenience for simulation. It is not a real-world
emergency stop.

## Prohibited harmful use

The official project is not intended to design, operate, optimize, or deploy
weapons or systems intended to injure people, damage property, evade lawful
safety controls, or facilitate illegal activity.

## Reporting safety issues

Report reproducible safety-control defects through the repository's private
security-reporting channel. Do not include credentials, confidential project
files, or instructions that unnecessarily increase the risk of harm.

## No warranty

Hephaestus is provided under the Apache License, Version 2.0. The license's
disclaimer of warranty and limitation of liability apply. Nothing in this
notice creates a warranty, certification, professional-client relationship, or
guarantee of fitness for a particular purpose.
