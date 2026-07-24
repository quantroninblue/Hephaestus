# Privacy Notice

Effective date: 24 July 2026

## Overview

Hephaestus AI is a local-first, open-source robotics engineering tool. The
Hephaestus project does not operate a hosted inference service, user-account
service, telemetry service, or project-storage service. Merely installing or
running Hephaestus does not send project data to the Hephaestus maintainers.

This notice describes the behavior of the open-source Hephaestus software. A
third party that redistributes, modifies, or hosts Hephaestus may have different
data practices and must provide its own disclosures.

## Local data

Project files, generated CAD, simulation assets, ROS workspaces, configuration,
logs, and runtime state are stored locally on the user's computer unless the
user deliberately copies or transmits them elsewhere.

The interactive conversation is held in process memory for the active session.
The `/clear` command clears the active conversation history. Exiting the
interactive session clears in-memory conversation state. Files deliberately
written to the workspace remain until the user deletes them.

Non-secret AI configuration may be stored under the user's operating-system
configuration directory. API credentials are read from environment variables.
Hephaestus is not intended to place API credentials inside project files,
generated artifacts, logs, command output, or `project_state.txt`.

## AI-provider processing

Natural-language AI features require a separately configured third-party AI
provider. The initial supported provider is Groq.

When a user invokes an AI feature, Hephaestus may send the provider:

- the user's prompt;
- Hephaestus system instructions;
- relevant conversation history;
- registered tool descriptions;
- selected workspace file contents requested through a read tool; and
- tool results needed to continue the engineering task.

Local slash commands such as `/cad`, `/gazebo`, `/ros`, `/cmdvel`, `/move`, and
`/stop` are handled locally and are not sent to the AI provider.

The user's relationship with an AI provider is governed by that provider's
terms and privacy documentation. Groq's current data documentation is available
at:

https://console.groq.com/docs/your-data

Users should review provider settings, retention controls, and account terms
before sending confidential, personal, proprietary, export-controlled, or
safety-sensitive information.

## Telemetry

The official Hephaestus open-source CLI does not enable product analytics,
behavioral telemetry, advertising trackers, or automatic project uploads.

Local capability checks and diagnostics inspect the user's machine but do not
send their results to the Hephaestus maintainers.

## API keys and secrets

Users are responsible for protecting their provider credentials. Credentials
should be supplied through documented environment variables or another
supported secret mechanism and must not be committed to source control.

Hephaestus attempts to redact secret-like values from ordinary output, but no
automated redaction system is perfect. Users should inspect logs and generated
files before sharing them.

## User control

Users may stop provider processing by not configuring an AI credential or by
removing it from the process environment. Provider-independent local commands
remain available without an AI credential.

Users may delete locally stored Hephaestus data by deleting the relevant
workspace, generated artifacts, logs, and non-secret configuration files.
Provider-side deletion and retention controls must be exercised through the
provider.

## Changes to this notice

Material changes to the official project's data behavior should be accompanied
by an update to this notice and its effective date.

## Contact

Privacy questions about the official open-source project should be submitted
through the repository's private maintainer or security contact channel. Do not
post API keys, personal data, confidential project files, or vulnerability
details in a public issue.
