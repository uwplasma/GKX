#!/usr/bin/env bash
# Q2 ladder supervisor (office CPU, shared host). Serial fresh processes, each in a
# systemd user scope with MemoryMax=20G, pinned to logical CPUs 0-11 with 12 BLAS/OMP
# threads, nice 10. A 5-s poll records process-group RSS and the scope's cgroup
# memory; it sends TERM then KILL above 20 GiB group RSS or 30 min wall (2 h total
# cap). Each rung runs the sparse arm first and seeds the next rung's shift with its
# certified eigenvalue, then the runtime-default arm; the propagator arms run last
# while budget remains. A memory or wall breach is a recorded result, never retried
# with a larger cap, and stops the ladder after that rung's remaining arm.
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
PY=/home/rjorge/venvs/gkx-nl/bin/python
OUT=$DIR/out
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=true GKX_X64=1
export PYTHONPATH=$DIR/src:$DIR
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 XLA_FLAGS=--xla_cpu_multi_thread_eigen=true
CPUS=0-11
MEM_MAX=20G
START=$(date +%s)
RUNG_CAP=1800
TOTAL_CAP=7200
RSS_CAP_KB=$((20 * 1024 * 1024))
SUP=$OUT/supervisor.txt
SHIFT="${1:-0.09302951-0.28199404j}"
echo "supervisor pid=$$ start=$(date -Is) shift0=$SHIFT cpus=$CPUS MemoryMax=$MEM_MAX OMP/OPENBLAS=12 XLA_FLAGS=$XLA_FLAGS" >> "$SUP"
BREACH=0

run_one() {
  local rung=$1 arm=$2 shift=$3
  local now remaining cap out tv tpid pypid pgid rss peak_rss cg mem peak_mem oom elapsed rc reason unit
  now=$(date +%s)
  remaining=$((TOTAL_CAP - (now - START)))
  if [ "$remaining" -lt 120 ]; then
    echo "skip rung=$rung arm=$arm reason=total-budget remaining=${remaining}s" >> "$SUP"
    return 2
  fi
  cap=$RUNG_CAP
  [ "$remaining" -lt "$cap" ] && cap=$remaining
  out=$OUT/r${rung}_${arm}.txt
  tv=$OUT/r${rung}_${arm}.time.txt
  unit=gkx-q2-r${rung}-${arm}-$$
  echo "launch rung=$rung arm=$arm shift=$shift cap=${cap}s unit=$unit $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$SUP"
  nohup nice -n 10 systemd-run --user --scope --quiet -p MemoryMax=$MEM_MAX --unit="$unit" \
    taskset -c "$CPUS" timeout --signal=TERM --kill-after=10s 7200s /usr/bin/time -v -o "$tv" \
    "$PY" "$DIR/exact_ladder.py" --repo "$DIR" --rung "$rung" --arm "$arm" --shift "$shift" \
    > "$out" 2>&1 < /dev/null &
  tpid=$!
  sleep 3
  pypid=$(pgrep -f "^$PY $DIR/exact_ladder.py --repo $DIR --rung $rung --arm $arm " | head -1)
  pgid=$(ps -o pgid= -p "${pypid:-$tpid}" 2> /dev/null | tr -d ' ')
  cg=""
  [ -n "${pypid:-}" ] && cg=/sys/fs/cgroup$(cut -d: -f3 "/proc/$pypid/cgroup" 2> /dev/null)
  echo "  pids launcher=$tpid python=${pypid:-?} pgid=${pgid:-?} affinity=$(taskset -p -c "${pypid:-$tpid}" 2> /dev/null | awk -F': ' '{print $2}') cgroup=$cg memory.max=$(cat "$cg/memory.max" 2> /dev/null)" >> "$SUP"
  reason=""
  peak_rss=0
  peak_mem=0
  oom=0
  while kill -0 "$tpid" 2> /dev/null; do
    elapsed=$(($(date +%s) - now))
    rss=0
    [ -n "${pgid:-}" ] && rss=$(ps -o rss= -g "$pgid" 2> /dev/null | awk '{s += $1} END {print s + 0}')
    [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
    mem=$(cat "$cg/memory.current" 2> /dev/null || echo 0)
    [ "$mem" -gt "$peak_mem" ] && peak_mem=$mem
    oom=$(awk '/^oom_kill /{print $2}' "$cg/memory.events" 2> /dev/null || echo "$oom")
    if [ "$elapsed" -gt "$cap" ]; then reason="wall>${cap}s"; fi
    if [ "$rss" -gt "$RSS_CAP_KB" ]; then reason="group_rss=${rss}kB>20GiB"; fi
    if [ -n "$reason" ]; then
      echo "  GUARD rung=$rung arm=$arm $reason elapsed=${elapsed}s peak_group_rss_kb=$peak_rss peak_cgroup_bytes=$peak_mem last=$(tail -1 "$out" | cut -c1-200)" >> "$SUP"
      [ -n "${pgid:-}" ] && kill -TERM -- "-$pgid" 2> /dev/null
      kill -TERM "$tpid" 2> /dev/null
      sleep 12
      [ -n "${pgid:-}" ] && kill -KILL -- "-$pgid" 2> /dev/null
      BREACH=1
      break
    fi
    sleep 5
  done
  wait "$tpid"
  rc=$?
  if [ -z "$reason" ] && [ "$rc" -ne 0 ] && ! grep -q '^RESULT ' "$out"; then
    echo "  BREACH rung=$rung arm=$arm rc=$rc no RESULT (cgroup oom_kill=$oom; MemoryMax $MEM_MAX) peak_group_rss_kb=$peak_rss peak_cgroup_bytes=$peak_mem" >> "$SUP"
    BREACH=1
  fi
  echo "  done rung=$rung arm=$arm rc=$rc elapsed=$(($(date +%s) - now))s maxrss_kb=$(awk -F': ' '/Maximum resident/{print $2}' "$tv" 2> /dev/null) peak_group_rss_kb=$peak_rss peak_cgroup_bytes=$peak_mem oom_kill=$oom $(date -Is)" >> "$SUP"
  return 0
}

for rung in 1 2 3 4; do
  run_one "$rung" sparse "$SHIFT"
  next=$("$PY" - "$OUT/r${rung}_sparse.txt" << 'EOF'
import json, sys
rec = None
for line in open(sys.argv[1]):
    if line.startswith("RESULT "):
        rec = json.loads(line[7:])
if rec and rec.get("certified"):
    print(f"{rec['gamma']:.10g}{-rec['omega']:+.10g}j")
EOF
  )
  if [ -n "$next" ]; then
    echo "  shift for rung $((rung + 1)) = $next (rung $rung certified sparse eigenvalue)" >> "$SUP"
    SHIFT=$next
  else
    echo "  rung $rung sparse not certified; next shift unchanged $SHIFT" >> "$SUP"
  fi
  run_one "$rung" default "$SHIFT"
  last_rung=$rung
  if [ "$BREACH" -ne 0 ]; then
    echo "ladder stopped after rung $rung (breach)" >> "$SUP"
    break
  fi
done
for rung in $(seq 1 "${last_rung:-0}"); do
  run_one "$rung" propagator "0.09302951-0.28199404j" || true
done
"$PY" "$DIR/exact_ladder.py" --summarize "$OUT" > "$OUT/summary.txt" 2>&1
echo "supervisor end=$(date -Is) total=$(($(date +%s) - START))s breach=$BREACH" >> "$SUP"
