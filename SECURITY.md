# Security policy

## Supported versions

Security fixes are provided for the latest release of GKX. Older releases are
not patched; please upgrade to the latest release first.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Report it privately through GitHub: open the **Security** tab of
[uwplasma/GKX](https://github.com/uwplasma/GKX/security) and choose
**Report a vulnerability** (a private security advisory). Include the GKX
version, a description of the problem, and the smallest input or steps that
reproduce it.

## What to expect

- Acknowledgement within 7 days.
- An assessment and, if the report is confirmed, a fix timeline within 30 days.
- A fix in the next release, with the advisory published once a patched release
  is available. Reporters are credited unless they prefer otherwise.

## Scope

GKX is a numerical research code that reads input decks and geometry files and
writes result files. Reports about unsafe file handling, command or code
execution from crafted inputs, or compromised dependencies and CI workflows
are in scope. Numerical accuracy issues are ordinary bugs and belong in the
public issue tracker.
