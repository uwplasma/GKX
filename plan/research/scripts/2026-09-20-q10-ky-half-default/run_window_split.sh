#!/bin/bash
# Q10: compile/execution split of the eager window gradient, A/B on one tree.
#
# usage: run_window_split.sh DIR TREE CORES CHECK_CORES BLOCKS
#
# The gate, the pinning and the arm rotation are run_timing_ab.sh's, unchanged:
# before every arm the pinned cores and their hyperthread siblings must be
# under BUSYMEAN mean / BUSYWORST worst busy over 3 s and the 1-minute load
# under LOADMAX, and the arm order alternates per block.  The kernel is
# window_compile_split.py, which reports per call how many backend
# compilations ran and how long they took, so the wall time can be split.
set -u
DIR=$1; TREE=$2; CORES=$3; CHECK=$4; BLOCKS=$5
LOADMAX=${LOADMAX:-6}
PY=${PY:-python}
OUT=$DIR/out; mkdir -p "$OUT"
expand() { python3 -c "
import sys
out=[]
for part in sys.argv[1].split(','):
    a,_,b=part.partition('-'); out+=range(int(a),int(b or a)+1)
print(' '.join(map(str,out)))" "$1"; }
CORE_LIST=$(expand "$CHECK")
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
fracs = []
for c in cores:
    d = [y - x for x, y in zip(a[c], b[c])]
    total = sum(d); idle = d[3] + d[4]
    fracs.append(1.0 - idle / total if total else 0.0)
print(f"{sum(fracs) / len(fracs):.3f} {max(fracs):.3f}")
PY
}
for block in $(seq 1 "$BLOCKS"); do
  if [ $((block % 2)) -eq 1 ]; then ARMS="full half"; else ARMS="half full"; fi
  for arm in $ARMS; do
    while true; do
      load=$(awk '{print $1}' /proc/loadavg)
      read -r mean worst <<< "$(busy $CORE_LIST)"
      if awk -v l="$load" -v a="$mean" -v w="$worst" -v m="$LOADMAX" \
             -v am="${BUSYMEAN:-0.05}" -v wm="${BUSYWORST:-0.20}" \
             'BEGIN{exit !(l < m && a < am && w < wm)}'; then break; fi
      echo "$(date +%T) wait load=$load mean_core_busy=$mean worst_core_busy=$worst" >> "$OUT/wait_window_split.log"
      sleep 60
    done
    echo "$(date +%T) block=$block arm=$arm load=$load mean_core_busy=$mean worst_core_busy=$worst" >> "$OUT/runs_window_split.log"
    (cd "$TREE" && XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu \
      taskset -c "$CORES" $PY "$DIR/window_compile_split.py" "$OUT/window_split_${arm}_b${block}.json" \
      --ky-layout "$arm" > "$OUT/window_split_${arm}_b${block}.log" 2>&1)
    echo "$(date +%T) rc=$? load_after=$(awk '{print $1}' /proc/loadavg)" >> "$OUT/runs_window_split.log"
  done
done
echo ALLDONE >> "$OUT/runs_window_split.log"
