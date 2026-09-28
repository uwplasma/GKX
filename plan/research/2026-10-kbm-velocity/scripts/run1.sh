#!/bin/bash
# usage: run1.sh <cpus> <tag> <args to kbm_eig.py without out>
cd $HOME/lanes/kbm-vel; CPUS=$1; TAG=$2; shift 2
set -- "$@"; G=$1; KY=$2; NL=$3; NM=$4; V=$5; shift 5
out=out/${TAG}.npz; log=out/${TAG}.txt
[ -f $log ] && grep -q RESULT $log && exit 0
KBM_TIME=$KBM_TIME KBM_SPARSE=$KBM_SPARSE KBM_SHIFT=$KBM_SHIFT OMP_NUM_THREADS=4 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=$HOME/lanes/kbm-vel/src \
  timeout ${TMO:-7200} nice -n 10 taskset -c $CPUS $HOME/lanes/xcode/venv/bin/python kbm_eig.py $G $KY $NL $NM $V $out "$@" > $log 2>&1
