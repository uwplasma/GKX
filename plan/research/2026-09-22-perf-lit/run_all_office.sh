#!/usr/bin/env bash
# Serialized office chain for the PERF-LIT remainder: GPU extra phase alone on
# a free GPU (host-launch-bound rows are sensitive to host load, so no CPU job
# of this lane runs beside it), then the CPU phase, then the pr3-cm re-run.
# Usage (repository root): run_all_office.sh <python>
set -u
PY=$1
bash plan/research/2026-09-22-perf-lit/wait_gpu_then_run.sh "$HOME/perflit_gpu2" "$PY"
PHASE=cpu bash plan/research/2026-09-22-perf-lit/run_profile.sh "$HOME/perflit_cpu" "$PY" cpu
bash plan/research/2026-09-22-perf-lit/run_pr3.sh "$HOME/perflit_pr3" "$PY"
