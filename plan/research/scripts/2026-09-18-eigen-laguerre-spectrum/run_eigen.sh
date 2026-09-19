#!/usr/bin/env bash
# Q16 supervisor: one fresh process per (nu, Nl) case, cheapest rung first.
#
# Usage: run_eigen.sh <run_dir> <key> [<key> ...]
#
# <run_dir> holds src_stage/ (the staged branch), results/, logs/, vectors/.
# Each key is "<nu_tag>-nl<Nl>"; NU_OF resolves the collisionality. The GPU
# guard refuses to start a key if the selected device has any compute process,
# so another session's run is never shared or preempted. The host process is
# pinned to cores 18-35; cores 2-17 are reserved for another lane.
set -u

RUN_DIR="$1"; shift
SRC="$RUN_DIR/src_stage"
DECKS="$SRC/plan/research/scripts/2026-09-18-eigen-laguerre-spectrum"
VENV=/home/rjorge/venvs/gkx-nl/bin
GPU="${GPU:-0}"
CORES="${CORES:-18-35}"
TIMEOUT="${TIMEOUT:-5400}"

mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/vectors"

nu_of() {
  case "$1" in
    nu0-*)    echo 0.0 ;;
    nu1e-3-*) echo 1e-3 ;;
    nu3e-3-*) echo 3e-3 ;;
    nu1e-2-*) echo 1e-2 ;;
    *) echo "unknown nu tag in key $1" >&2; exit 2 ;;
  esac
}

gpu_busy() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
         | awk -F', ' -v g="$GPU" '$1==g {print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    | grep -qF "$uuid"
}

for KEY in "$@"; do
  NU=$(nu_of "$KEY")
  NL="${KEY##*-nl}"
  if gpu_busy; then
    echo "STOP  $(date -Is)  GPU $GPU has a compute process; refusing $KEY" \
      | tee -a "$RUN_DIR/supervisor.txt"
    exit 4
  fi
  echo "START $(date -Is)  $KEY nu=$NU Nl=$NL" | tee -a "$RUN_DIR/supervisor.txt"
  ( cd "$SRC" && \
    env PYTHONPATH="$SRC/src:$SRC" \
        JAX_PLATFORMS=cuda \
        CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        JAX_ENABLE_X64=true GKX_X64=1 MPLBACKEND=Agg \
        GX_PARITY_REF_DIR=/home/rjorge/gkx-r0-rate-parity-20260905.GtHbRz/matched_refs \
    taskset -c "$CORES" \
    /usr/bin/time -v -o "$RUN_DIR/results/$KEY.time.txt" \
    timeout --signal=TERM --kill-after=10s "$TIMEOUT" \
    "$VENV/python" "$DECKS/eigen_spectrum.py" \
      --key "$KEY" --nu "$NU" --nl "$NL" --nm 96 \
      --repo "$SRC" --deck-dir "$DECKS" \
      --out "$RUN_DIR/vectors" --save-vector \
  ) > "$RUN_DIR/logs/$KEY.run.txt" 2>&1
  RC=$?
  grep -h '^RESULT ' "$RUN_DIR/logs/$KEY.run.txt" > "$RUN_DIR/results/$KEY.txt" 2>/dev/null
  echo "END   $(date -Is)  $KEY rc=$RC" | tee -a "$RUN_DIR/supervisor.txt"
  if [ "$RC" -ne 0 ]; then
    echo "ABORT $(date -Is)  $KEY exited $RC" | tee -a "$RUN_DIR/supervisor.txt"
    exit "$RC"
  fi
done
echo "ALL DONE $(date -Is)" | tee -a "$RUN_DIR/supervisor.txt"
