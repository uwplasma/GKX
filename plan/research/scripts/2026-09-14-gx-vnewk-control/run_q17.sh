#!/bin/bash
# Q17 sequential GX supervisor (office). NOT YET RUN (Q17 paused 2026-09-14).
# Usage: run_q17.sh KEY [KEY ...]   (each KEY.in in the run directory R)
# Modeled on gx-toolchain-rebuild-20260914/gpu_run_long.sh. For every key:
# binary SHA-256 and ldd checked; a GPU is chosen only if it has no compute
# process and <5% utilization on two polls 60 s apart, and has no compute
# process again immediately before launch (never share or preempt); GPU wait
# capped by GPU_WAIT_S (default 7200 s, then abort); per-run cap RUN_CAP_S
# (default 2700 s); the log is scanned for nan/inf tokens every 30 s; one GX
# job at a time; restart/big files are deleted after the fit; stop on the first
# nonzero exit, nonfinite token or failed fit.
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
R=/home/rjorge/gkx-q17-gx-vnewk-20260914
BIN=/home/rjorge/gkx-nl24-discriminator-20260912.vvmgDD/gx
BINSHA=96a53403a803e40fe3f9f6d1734779158d8be84d22e13155eb952a9035d70536
PY=/home/rjorge/local/micromamba/envs/gk-fortran/bin/python
CAP=${RUN_CAP_S:-2700}
WAIT=${GPU_WAIT_S:-7200}
cd "$R" || exit 2
log(){ echo "=== $(date -Is) $*"; }
log "supervisor pid $$ host $(hostname) keys: $*"
[ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$BINSHA" ] || { log "ABORT binary sha mismatch"; exit 5; }
if ldd "$BIN" | grep -q "not found"; then log "ABORT ldd not found"; exit 90; fi
declare -A UUID2IDX
while IFS=", " read -r idx uuid; do UUID2IDX[$uuid]=$idx; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
poll(){
  local busy="" out=""
  while IFS=", " read -r uuid pid; do [ -n "$uuid" ] && busy="$busy ${UUID2IDX[$uuid]}"; done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
  while IFS=", " read -r idx util; do util=${util% %}; if [ "$util" -lt 5 ] && [[ " $busy " != *" $idx "* ]]; then out="$out $idx"; fi; done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader)
  echo $out
}
pick_gpu(){
  local deadline=$(( $(date +%s) + WAIT )) a b g
  GPU=""
  while [ "$(date +%s)" -lt "$deadline" ]; do
    a=$(poll); log "poll1 idle=[$a]"; sleep 60; b=$(poll); log "poll2 idle=[$b]"
    for g in $a; do if [[ " $b " == *" $g "* ]]; then GPU=$g; break; fi; done
    if [ -n "$GPU" ] && [ -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)" ]; then return 0; fi
    GPU=""; sleep 60
  done
  return 1
}
for key in "$@"; do
  pick_gpu || { log "ABORT no idle GPU within ${WAIT}s before $key"; exit 91; }
  log "$key start gpu=$GPU input sha256 $(sha256sum "$key.in" | cut -d' ' -f1)"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU setsid timeout --signal=TERM --kill-after=10s "${CAP}s" \
    /usr/bin/time -v -o "$key.time.txt" "$BIN" "$key.in" > "$key.run.log" 2>&1 &
  child=$!
  sleep 5
  log "$key timeout pid $child gx pids $(pgrep -f -- "$BIN $key.in" | tr '\n' ' ')"
  reason=completed
  : > "$key.gpumem.txt"
  while kill -0 "$child" 2> /dev/null; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> "$key.gpumem.txt"
    if grep -Eiq '(^|[^a-z])(nan|inf|infinity)([^a-z]|$)' "$key.run.log"; then
      reason="nonfinite log token"; kill -TERM -- "-$child" 2> /dev/null || kill -TERM "$child"
      sleep 10; kill -KILL -- "-$child" 2> /dev/null; break
    fi
    sleep 30
  done
  wait "$child"; rc=$?
  log "$key exit rc=$rc reason=$reason wall_s=$(( $(date +%s) - T0 )) gpu=$GPU"
  rm -f "$key.restart.nc" "$key.big.nc"
  if [ "$rc" -ne 0 ] || [ "$reason" != completed ]; then log "STOP at $key"; exit 1; fi
  "$PY" gx_fit.py "$key.out.nc" "$key" "$key.json" > "$key.fit.txt" 2>&1 || { log "STOP fit failed at $key"; exit 3; }
  "$PY" -c "import json,math,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if math.isfinite(r['gamma_phi2_late']) and math.isfinite(r['omega_second_half_mean']) and r['nonfinite_phi2']==0 else 3)" "$key.json" || { log "STOP nonfinite fit at $key"; exit 3; }
  log "$key out.nc sha256 $(sha256sum "$key.out.nc" | cut -d' ' -f1)"
done
log "DONE"
