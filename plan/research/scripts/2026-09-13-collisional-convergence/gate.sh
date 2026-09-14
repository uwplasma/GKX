#!/usr/bin/env bash
# GPU occupancy gate for Q8 (office). Written after GPU1 was taken by another
# session's process at 19:05 CDT and run_convergence.sh stopped (exit 4).
# Usage: gate.sh RUN_DIR WAIT_CAP_SECONDS
# Policy: never share or preempt a GPU. Poll both GPUs every 300 s; a GPU is
# eligible when it has no compute process and utilization < 5% on two
# consecutive polls (GPU1 preferred). On eligibility run, in order (lead's
# priority at the 2026-09-14 resume): the GKX base keys without a result in
# BASE order (Nl48, then Nl64, then Nl16), then the GX keys without a result,
# then the T=300 rerun of any base key whose settled flag is false. A supervisor that
# stops because the GPU became busy (exit 4) re-arms the gate with fresh
# polls; any other nonzero exit aborts. Cumulative sleep is capped.
set -u
RUN_DIR=$1
WAIT_CAP=$2
D=plan/research/scripts/2026-09-13-collisional-convergence
S="$RUN_DIR/src_stage/$D"
PY=/home/rjorge/venvs/gkx-nl/bin/python
LOG="$RUN_DIR/logs/gate.txt"
BASE="nu3e-3-nl24 nu3e-3-nl32 nu1e-3-nl48 nu3e-3-nl48 nu1e-2-nl48 nu0-nl64 nu1e-3-nl16 nu3e-3-nl16 nu1e-2-nl16"
GX="gx-nu1e-2-nl24 gx-nu1e-2-nl32 gx-nu1e-2-nl32-nohyper"

settled() {
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['cases'][0]['rows'][0]['converged'] else 1)" "$1"
}

remaining_gkx() {
  local k keys=""
  for k in $BASE; do
    [ -f "$RUN_DIR/results/$k.json" ] || keys="$keys $k"
  done
  echo $keys
}

remaining_t300() {
  local k keys=""
  for k in $BASE; do
    if [ -f "$RUN_DIR/results/$k.json" ] && ! settled "$RUN_DIR/results/$k.json" \
      && [ ! -f "$RUN_DIR/results/$k-t300.json" ]; then
      keys="$keys $k-t300"
    fi
  done
  echo $keys
}

remaining_gx() {
  local k keys=""
  for k in $GX; do
    [ -f "$RUN_DIR/gx/$k.json" ] || keys="$keys $k"
  done
  echo $keys
}

idle() {
  local apps util
  apps=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | tr -d ' \n')
  util=$(nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  [ -z "$apps" ] && [ "$util" -lt 5 ]
}

echo "gate pid $$ host $(hostname) start $(date -Is) wait cap ${WAIT_CAP}s" >> "$LOG"
waited=0
previous=""
while true; do
  keys_gkx=$(remaining_gkx)
  keys_gx=$(remaining_gx)
  keys_t300=""
  if [ -z "$keys_gkx" ] && [ -z "$keys_gx" ]; then
    keys_t300=$(remaining_t300)
  fi
  if [ -z "$keys_gkx" ] && [ -z "$keys_gx" ] && [ -z "$keys_t300" ]; then
    echo "ALL DONE $(date -Is)" >> "$LOG"
    exit 0
  fi
  current=""
  for g in 1 0; do
    if idle "$g"; then current="$current $g"; fi
  done
  state=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
  echo "poll $(date -Is) waited ${waited}s idle [${current# }] previous [${previous# }] gpus $state apps $apps" >> "$LOG"
  chosen=""
  for g in $current; do
    case " $previous " in *" $g "*) chosen=$g; break ;; esac
  done
  if [ -n "$chosen" ]; then
    if [ -n "$keys_gkx" ]; then
      echo "launch gkx gpu $chosen keys $keys_gkx $(date -Is)" >> "$LOG"
      bash "$S/run_convergence.sh" "$RUN_DIR" "$chosen" $keys_gkx
    elif [ -n "$keys_gx" ]; then
      echo "launch gx gpu $chosen keys $keys_gx $(date -Is)" >> "$LOG"
      bash "$S/run_gx.sh" "$RUN_DIR" "$chosen" $keys_gx
    else
      echo "launch gkx t300 gpu $chosen keys $keys_t300 $(date -Is)" >> "$LOG"
      bash "$S/run_convergence.sh" "$RUN_DIR" "$chosen" $keys_t300
    fi
    rc=$?
    echo "supervisor exit $rc $(date -Is)" >> "$LOG"
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 4 ]; then
      echo "ABORT supervisor exit $rc" >> "$LOG"
      exit "$rc"
    fi
    previous=""
    continue
  fi
  previous=$current
  if [ "$waited" -ge "$WAIT_CAP" ]; then
    echo "STOP wait cap reached $(date -Is)" >> "$LOG"
    exit 6
  fi
  sleep 300
  waited=$((waited + 300))
done
