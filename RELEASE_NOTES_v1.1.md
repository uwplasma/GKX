# SPECTRAX-GK v1.1

## Highlights

- Finalized the shipped benchmark/documentation surface for the `v1.1` release.
- Tightened the README and Sphinx docs around executable usage, runtime outputs,
  physics/model navigation, and parallelization terminology.
- Kept the public nonlinear benchmark atlas focused on the archived windows that
  are actually tracked and auditable in the repository.
- Clarified the autodiff validation story: single-mode inverse as a local
  sensitivity/identifiability demonstration, two-mode inverse as the release
  parameter-recovery validation.

## Validation

- Full `pytest -q` passed for the release commit.
- `python -m sphinx -W -b dummy docs docs/_build/dummy` passed.
- `python -m build` produced both the source distribution and the wheel for
  `spectrax-gk 1.1`.

## Packaging and metadata

- Bumped the package version to `1.1`.
- Updated package metadata to use a modern MIT license expression compatible
  with current `setuptools` release builds.

## Documentation and usability

- Added standard README badges for release, CI, MIT license, Python version,
  and coverage.
- Simplified installation instructions to `git clone`, `cd`, and `pip install -e .`.
- Standardized user-facing wording on `executable` and `parallelization`.
- Tightened the top-level documentation map so the physics, operators,
  numerics, geometry, and runtime input pages read as one coherent technical set.
