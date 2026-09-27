#!/bin/bash
# usage: gpu_run.sh RUNDIR BINARY INPUT
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
RUNDIR=$1; BIN=$2; INP=$3
cd "$RUNDIR"
log(){ echo "=== $(date -Is) $*"; }
log "binary $BIN sha256 $(sha256sum "$BIN" | cut -d" " -f1)"
if ldd "$BIN" | grep -q "not found"; then log "ABORT ldd not found"; ldd "$BIN" | grep "not found"; exit 90; fi
declare -A UUID2IDX
while IFS=", " read -r idx uuid; do UUID2IDX[$uuid]=$idx; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
poll(){ # prints space-separated idle gpu indices
  local busy="" out=""
  while IFS=", " read -r uuid pid mem; do case "$(ps -o args= -p $pid 2>/dev/null)" in *gunicorn*) continue;; esac; [ -n "$uuid" ] && busy="$busy ${UUID2IDX[$uuid]}"; done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
  while IFS=", " read -r idx util; do util=${util% %}; if [ "$util" -lt 5 ] && [[ " $busy " != *" $idx "* ]]; then out="$out $idx"; fi; done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader)
  echo $out
}
DEADLINE=$(( $(date +%s) + 28800 )); GPU=""
while [ $(date +%s) -lt $DEADLINE ]; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr "\n" ";"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr "\n" ";"; echo
  A=$(poll); log "poll1 idle=[$A]"; sleep 60
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr "\n" ";"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr "\n" ";"; echo
  Bp=$(poll); log "poll2 idle=[$Bp]"
  for g in $A; do if [[ " $Bp " == *" $g "* ]]; then GPU=$g; break; fi; done
  [ -n "$GPU" ] && break
  sleep 60
done
[ -z "$GPU" ] && { log "ABORT no idle GPU within 2h"; exit 91; }
log "selected CUDA_VISIBLE_DEVICES=$GPU"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU "$BIN" "$INP" > run.log 2>&1
RC=$?
log "exit rc=$RC wall_s=$(( $(date +%s) - T0 )) gpu=$GPU"
exit $RC
