Validation And Coverage Strategy
================================

Two machine-readable files tie claims, tests, and artifacts together. The
evidence ledger lists every public number; the coverage manifest lists every
source module. Coverage is a guardrail: a test counts when it protects an
equation, a numerical method, a diagnostic convention, an artifact contract,
or an autodiff guarantee.

Evidence ledger
---------------

``tools/evidence_ledger.toml`` has one ``[[row]]`` per public number or claim.
Each row carries ``id``, ``claim``, ``observable``, ``artifact``,
``generator``, ``reference``, ``reference_rank``, ``tier``, and ``status``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Values
   * - ``reference_rank``
     - 1 analytic; 2 published numbers; 3 reference-code golden shipped in that
       code's tree; 4 open published dataset; 5 self-run comparator.
   * - ``tier``
     - 0 on push, under 60 s; 1 on push, bitwise on stored outputs; 2 nightly
       physics; 3 campaign or GPU.
   * - ``status``
     - ``open``, ``passing``, ``failing``, ``provisional``, ``superseded``.

A claim cites the highest-ranked reference available. A self-run reference is
rank 5 and is never described as published.

``tests/release/test_evidence_ledger.py`` enforces the contract:

- every row has the required keys and valid rank, tier, and status;
- every artifact and generator exists;
- a ``provisional`` row has ``notes`` saying what is unresolved;
- a rank-3 row pins its reference file with ``reference_sha256``;
- linked-boundary GX references record their end-damping clamp state;
- the README parity table and the ledger list the same cases, and each README
  percentage recomputes from its artifact as
  ``100 max|gkx - ref| / max|ref|``;
- every linear row declares its velocity-space regularization, and a
  collisionless row with no Laguerre-space sink reports its velocity limit;
- :doc:`verification_matrix` matches the ledger.

``scripts/validation_matrix.py`` writes the ledger section of
:doc:`verification_matrix`:

.. code-block:: bash

   python scripts/validation_matrix.py           # rewrite the generated block
   python scripts/validation_matrix.py --check   # exit 1 if it is stale

Claim scope
-----------

A passing test or a covered line does not promote a claim. The artifact must
also record the observable, reference, tolerance, and accepted claim level.
:doc:`release_scope` is the human-readable claim ledger and stays in step with
``benchmarks/references/gkx_1_7_release_contract.json``. An example that runs
is a release claim only when it is labeled release-gated and tied to its
artifact; otherwise it is a stress lane, pilot, or deferred lane.

Coverage manifest
-----------------

``tools/validation_coverage_manifest.toml`` gives each module a source path,
owning lane, reference anchors, physics and numerics contracts, fast tests,
artifacts, and next tests; :doc:`code_structure` gives the ownership rules.

.. code-block:: bash

   # structure and paths
   python tools/release/check_validation_coverage_manifest.py
   # JSON summary
   python tools/release/check_validation_coverage_manifest.py \
     --out-json docs/_static/validation_coverage_manifest_summary.json
   # with measured coverage, as the wide-coverage CI job runs it
   python tools/release/check_validation_coverage_manifest.py \
     --coverage-xml coverage-wide.xml \
     --enforce-package-coverage \
     --out-json docs/_static/validation_coverage_manifest_summary.json
   # which tracked gate reports pass
   python tools/release/check_validation_coverage_manifest.py gate-index

``--enforce-package-coverage`` fails below the 95% package target and records
per-module gaps; ``--enforce-module-coverage`` is a separate switch.
``gate-index`` writes ``docs/_static/validation_gate_index.json`` and skips
reports that set ``gate_index_include = false``. :doc:`testing` gives the
current gate-index result.
