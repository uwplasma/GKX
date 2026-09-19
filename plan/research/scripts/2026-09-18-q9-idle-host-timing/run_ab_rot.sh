#!/bin/bash
# Q9 idle-host A/B/A/B timing with the arm order rotated per block.
# usage: run_ab_rot.sh DIR CORES BLOCKS "GRID ARGS" FLAGSET TAG
#   DIR      staged dir holding base/ and p2t/ trees and bench_q9.py
#   CORES    taskset list, e.g. 2-17
#   BLOCKS   number of alternating blocks
#   FLAGSET  pool (default XLA flags) or nopool (--xla_cpu_multi_thread_eigen=false)
#   TAG      output prefix, e.g. pool64
# Odd blocks run base then p2t, even blocks run p2t then base, so the arm
# order is rotated and a drifting machine load no longer favours one arm.
# Before each arm the CHECK_CORES must average below BUSYMEAN busy over 3 s,
# with no single core above BUSYWORST, and the 1-minute load must be below
# LOADMAX; otherwise the arm waits (each wait is logged).
# SIB_CORES (the hyperthread siblings of CORES) are only logged, not gated.
set -u
DIR=$1; CORES=$2; BLOCKS=$3; GRID=$4; FLAGSET=$5; TAG=$6
LOADMAX=${LOADMAX:-8}
PY=${PY:-python}
OUT=$DIR/out; mkdir -p "$OUT"
expand() { python3 -c "
import sys
out=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); out+=range(int(a),int(b or a)+1)
print(' '.join(map(str,out)))" "$1"; }
CORE_LIST=$(expand "${CHECK_CORES:-$CORES}")
SIB_LIST=$(expand "${SIB_CORES:-$CORES}")
busy() { python3 - "$@" <<'PY'
import sys, time
cores = [int(c) for c in sys.argv[1:]]
def snap():
    rows = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3].isdigit():
            f = line.split(); rows[int(f[0][3:])] = list(map(int, f[1:]))
    return rows
a = snap(); time.sleep(3); b = snap()
worst = 0.0
fracs = []
for c in cores:
    d = [y - x for x, y in zip(a[c], b[c])]
    total = sum(d); idle = d[3] + d[4]
    frac = 1.0 - idle / total if total else 0.0
    fracs.append(frac)
    worst = max(worst, frac)
print(f"{sum(fracs) / len(fracs):.3f} {worst:.3f}")
PY
}
if [ "$FLAGSET" = nopool ]; then
  FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
else
  FLAGS=""
fi
for block in $(seq 1 "$BLOCKS"); do
  if [ $((block % 2)) -eq 1 ]; then ARMS="base p2t"; else ARMS="p2t base"; fi
  for arm in $ARMS; do
    done_json=$OUT/${TAG}_${arm}_b${block}.json
    if [ -f "$done_json" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if len(d['kernels']) >= int(sys.argv[2]) and 'load_after' in d else 1)" "$done_json" "${EXPECT_KERNELS:-4}"; then
      echo "$(date +%T) block=$block arm=$arm tag=$TAG kept existing complete result" >> "$OUT/runs_${TAG}.log"
      continue
    fi
    while true; do
      load=$(awk '{print $1}' /proc/loadavg)
      read -r mean worst <<< "$(busy $CORE_LIST)"
      if awk -v l="$load" -v a="$mean" -v w="$worst" -v m="$LOADMAX" -v am="${BUSYMEAN:-0.10}" -v wm="${BUSYWORST:-0.50}" 'BEGIN{exit !(l < m && a < am && w < wm)}'; then break; fi
      echo "$(date +%T) wait load=$load mean_core_busy=$mean worst_core_busy=$worst" >> "$OUT/wait_${TAG}.log"
      sleep 60
    done
    read -r sibmean sibworst <<< "$(busy $SIB_LIST)"
    echo "$(date +%T) block=$block arm=$arm tag=$TAG order=$ARMS load=$load mean_core_busy=$mean worst_core_busy=$worst sib_mean=$sibmean sib_worst=$sibworst" >> "$OUT/runs_${TAG}.log"
    (cd "$DIR/$arm" && XLA_FLAGS="$FLAGS" PYTHONPATH=$DIR/$arm/src:$DIR/$arm JAX_PLATFORMS=cpu \
      taskset -c "$CORES" $PY "$DIR/bench_q9.py" "$OUT/${TAG}_${arm}_b${block}.json" $GRID \
      > "$OUT/${TAG}_${arm}_b${block}.log" 2>&1)
    rc=$?
    echo "$(date +%T) rc=$rc load_after=$(awk '{print $1}' /proc/loadavg)" >> "$OUT/runs_${TAG}.log"
  done
done
echo ALLDONE >> "$OUT/runs_${TAG}.log"
