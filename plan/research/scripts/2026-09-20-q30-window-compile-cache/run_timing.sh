#!/bin/zsh
# Q30: compiles per call and wall per call, two trees, both ky layouts.
#
#   ./run_timing.sh <main-tree> <branch-tree> <out-dir>
#
# Arms are interleaved within each repetition so a drifting host load cannot
# be mistaken for a tree difference.
set -e
MAIN=$1; BRANCH=$2; OUT=$3
PY=${PY:-python}
here=${0:a:h}
mkdir -p $OUT
echo "load before: $(uptime | sed 's/.*averages*://')"
for rep in 1 2; do
  for arm in full half; do
    for tree in $MAIN $BRANCH; do
      label=$([[ $tree == $MAIN ]] && echo main || echo branch)
      echo "--- rep$rep $label ky=$arm"
      (cd $tree && JAX_PLATFORMS=cpu PYTHONPATH=$tree/src:$tree nice -n 10 \
        $PY $here/bench_q30.py $OUT/time_${label}_${arm}_r${rep}.json --ky-mode $arm --reps 3 | tail -4)
    done
  done
done
echo "load after: $(uptime | sed 's/.*averages*://')"
