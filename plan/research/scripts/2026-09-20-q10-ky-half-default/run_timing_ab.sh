#!/bin/bash
# Q10 wall-clock A/B/A/B of the two ky layouts on one pinned tree.
#
# usage: run_timing_ab.sh DIR TREE CORES CHECK_CORES BLOCKS "GRID ARGS" FLAGSET
#   DIR          staging dir that holds bench_ky_layout.py and receives out/
#   TREE         the pinned checkout both arms import (one tree, two layouts)
#   CORES        taskset list for the run, e.g. 12-17
#   CHECK_CORES  cores that must be idle -- CORES plus their HT siblings
#   BLOCKS       number of alternating blocks
#   FLAGSET      label for this campaign; it also selects the XLA flags.  A
#                label starting with "nopool" runs with
#                --xla_cpu_multi_thread_eigen=false (the float32 FFT pin that
#                #248 showed is needed for a reproducible reduction order),
#                anything else with the default flags.  The rest of the label
#                is free, so two grids can share one output directory
#                (nopool_g32, nopool_g64).
#
# Two rules this driver keeps that a naive A/B does not:
#
#   * The **arm order rotates** per block -- full,half then half,full -- so a
#     monotone drift in machine state (thermal, page cache, a co-tenant
#     arriving) cannot be read as an arm difference.  Q9's driver rotated for
#     the same reason.
#   * Before every arm the pinned cores *and their hyperthread siblings* must
#     be below BUSYMEAN mean / BUSYWORST worst busy over 3 s and the 1-minute
#     load below LOADMAX, or the driver waits and logs the wait.  A sibling
#     thread running someone else's job halves the core, and /proc/loadavg
#     alone does not see that.
#
# Both arms are the same tree and the same binary; only --ky-layout differs.
set -u
DIR=$1; TREE=$2; CORES=$3; CHECK=$4; BLOCKS=$5; GRID=$6; FLAGSET=$7
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
case "$FLAGSET" in
  nopool*) FLAGS="--xla_cpu_multi_thread_eigen=false" ;;
  *) FLAGS="" ;;
esac
for block in $(seq 1 "$BLOCKS"); do
  if [ $((block % 2)) -eq 1 ]; then ARMS="full half"; else ARMS="half full"; fi
  for arm in $ARMS; do
    done_json=$OUT/${FLAGSET}_${arm}_b${block}.json
    if [ -f "$done_json" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if len(d['kernels']) >= int(sys.argv[2]) and 'load_after' in d else 1)" "$done_json" "${EXPECT_KERNELS:-4}"; then
      echo "$(date +%T) block=$block arm=$arm flags=$FLAGSET kept existing complete result" >> "$OUT/runs_${FLAGSET}.log"
      continue
    fi
    while true; do
      load=$(awk '{print $1}' /proc/loadavg)
      read -r mean worst <<< "$(busy $CORE_LIST)"
      if awk -v l="$load" -v a="$mean" -v w="$worst" -v m="$LOADMAX" \
             -v am="${BUSYMEAN:-0.05}" -v wm="${BUSYWORST:-0.20}" \
             'BEGIN{exit !(l < m && a < am && w < wm)}'; then break; fi
      echo "$(date +%T) wait load=$load mean_core_busy=$mean worst_core_busy=$worst" >> "$OUT/wait_${FLAGSET}.log"
      sleep 60
    done
    echo "$(date +%T) block=$block arm=$arm flags=$FLAGSET load=$load mean_core_busy=$mean worst_core_busy=$worst" >> "$OUT/runs_${FLAGSET}.log"
    (cd "$TREE" && XLA_FLAGS="$FLAGS" PYTHONPATH=$TREE/src:$TREE JAX_PLATFORMS=cpu \
      taskset -c "$CORES" $PY "$DIR/bench_ky_layout.py" "$done_json" --ky-layout "$arm" $GRID \
      > "$OUT/${FLAGSET}_${arm}_b${block}.log" 2>&1)
    echo "$(date +%T) rc=$? load_after=$(awk '{print $1}' /proc/loadavg)" >> "$OUT/runs_${FLAGSET}.log"
  done
done
echo ALLDONE >> "$OUT/runs_${FLAGSET}.log"
