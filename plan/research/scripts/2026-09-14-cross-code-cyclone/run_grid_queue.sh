#!/usr/bin/env bash
# Q20 grid-code lane (office CPU). Usage: run_grid_queue.sh <queue-file> <lane> <ranks> <cpu-list>
# Queue lines: "<code> <case> <timeout_s>". Serial; one mpirun per case pinned to <cpu-list>,
# nice 10, OMP_NUM_THREADS=1, per-case timeout; a case with DONE is skipped. stella v1.0 exits 2
# after writing its outputs (format-string bug in init_stella.f90:1084), so rc=2 is not a failure by itself.
set -u
Q=$1; LANE=$2; NP=$3; CPUS=$4
ROOT=$(cd "$(dirname "$0")" && pwd)
export MAMBA_ROOT_PREFIX=/home/rjorge/local/micromamba
eval "$(/home/rjorge/local/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate gk-fortran
export OMP_NUM_THREADS=1
SUP=$ROOT/logs/supervisor_$LANE.txt
mkdir -p "$ROOT/logs"
echo "lane=$LANE pid=$$ start=$(date -Is) np=$NP cpus=$CPUS queue=$Q" >> "$SUP"
while read -r code case tcap; do
  [ -z "${code:-}" ] && continue
  case "$code" in \#*) continue ;; esac
  d=$ROOT/$code/$case
  [ -f "$d/DONE" ] && continue
  cd "$d" || { echo "$code $case missing dir" >> "$SUP"; continue; }
  if [ "$code" = gs2 ]; then exe=/home/rjorge/gk-codes/gs2/bin/gs2; else exe=/home/rjorge/gk-codes/stella/stella; fi
  t0=$(date +%s.%N)
  echo "$code $case start=$(date -Is)" >> "$SUP"
  timeout --signal=TERM --kill-after=20s "${tcap}s" nice -n 10 /usr/bin/time -v \
    mpirun -np "$NP" --bind-to none taskset -c "$CPUS" "$exe" "$case.in" > run.stdout.txt 2> run.time.txt < /dev/null
  rc=$?
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
  echo "$code $case rc=$rc wall=$wall end=$(date -Is)" >> "$SUP"
  echo "rc=$rc wall=$wall np=$NP cpus=$CPUS host=$(hostname)" > DONE
done < "$Q"
echo "lane=$LANE ALL DONE $(date -Is)" >> "$SUP"
