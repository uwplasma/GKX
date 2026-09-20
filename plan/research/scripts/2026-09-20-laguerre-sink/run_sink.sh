#!/usr/bin/env bash
# Q31 supervisor: one fresh process per (nu_hyper_l, Nl) case, cheapest first.
#
# Usage: run_sink.sh <run_dir> <key> [<key> ...]
#
# <run_dir> holds src_stage/ (the staged branch), results/, logs/, vectors/.
# Each key is "hl<tag>-nl<Nl>"; HL_OF resolves the sink strength. The GPU guard
# refuses to start a key if the selected device has any compute process, so a
# sibling lane's run is never shared or preempted.
set -u

RUN_DIR="$1"; shift
SRC="$RUN_DIR/src_stage"
DECKS="$SRC/plan/research/scripts/2026-09-20-laguerre-sink"
VENV=/home/rjorge/venvs/gkx-nl/bin
GPU="${GPU:-0}"
CORES="${CORES:-0-15}"
TIMEOUT="${TIMEOUT:-5400}"

mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/vectors"

hl_of() {
  case "$1" in
    hl0-*)    echo 0.0 ;;
    hl1e-3-*) echo 0.001 ;;
    hl1e-2-*) echo 0.01 ;;
    hl3e-2-*) echo 0.03 ;;
    hl1e-1-*) echo 0.1 ;;
    hl5e-1-*) echo 0.5 ;;
    *) echo "unknown sink tag in key $1" >&2; exit 2 ;;
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
  HL=$(hl_of "$KEY")
  NL="${KEY##*-nl}"
  if gpu_busy; then
    echo "STOP  $(date -Is)  GPU $GPU has a compute process; refusing $KEY" \
      | tee -a "$RUN_DIR/supervisor.txt"
    exit 4
  fi
  echo "START $(date -Is)  $KEY nu_hyper_l=$HL Nl=$NL" \
    | tee -a "$RUN_DIR/supervisor.txt"
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
    "$VENV/python" "$DECKS/sink_spectrum.py" \
      --key "$KEY" --nu 0.0 --nu-hyper-l "$HL" --p-hyper-l 6.0 \
      --nu-hyper-m-const 0.0 --nl "$NL" --nm 96 \
      --repo "$SRC" --deck-dir "$DECKS" \
      --out "$RUN_DIR/vectors" \
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
