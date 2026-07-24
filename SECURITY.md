# Security Policy

## Supported versions

Security fixes are provided on a best-effort basis for the most recent published
release and the current main development branch. Older versions may not receive
security updates.

## Reporting a vulnerability

Use the repository's private vulnerability-reporting feature when available.

If private reporting is unavailable, contact the maintainer through the private
contact method published on the repository profile. Do not publish exploit
details, credentials, private project files, or an unpatched vulnerability in a
public issue.

A useful report includes:

- the affected version and operating system;
- the relevant Hephaestus command or workflow;
- minimal reproduction steps;
- the expected and observed behavior;
- the potential impact;
- whether the issue crosses a workspace, credential, process, provider, ROS, or
  simulation boundary; and
- a proposed mitigation, if known.

Remove API keys and unrelated personal or proprietary information before
submitting a report.

## Security-sensitive areas

Reports are particularly important when they concern:

- workspace path traversal or symlink escape;
- arbitrary command or argument injection;
- execution outside the intended CadQuery worker boundary;
- credential disclosure or failed redaction;
- unauthorized file replacement;
- control of an unowned process;
- escape from the isolated simulation ROS graph;
- unintended physical-robot control;
- failure to issue a simulated zero-velocity command;
- unsafe package-installation or privilege-escalation behavior; or
- a provider response causing an unconfirmed local mutation.

## Secrets

Do not commit provider keys, tokens, passwords, private certificates, SSH keys,
cloud credentials, or confidential robot data.

Hephaestus credentials should be supplied using documented environment
variables or supported secret mechanisms. Secret-like values should not be
placed in prompts, project files, generated artifacts, screenshots, bug
reports, or public logs.

## Scope and limitations

The restricted CadQuery runner, workspace boundary, confirmations, process
ownership, and ROS simulation isolation are defense-in-depth controls. They are
not a complete operating-system sandbox or a certified safety boundary.

Provider availability, model accuracy, generated engineering quality, and
vulnerabilities in unsupported third-party installations are not themselves
security vulnerabilities in Hephaestus, although unsafe handling of those
conditions may be.

## Response

The project is maintained on a best-effort basis. Receipt of a report does not
create a support agreement, response-time commitment, bounty obligation, or
guarantee that a fix will be released.
