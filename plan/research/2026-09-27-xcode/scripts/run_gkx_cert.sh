#!/bin/bash
# usage: run_gkx_cert.sh <cpus> "<geom ky>" ...
cd ${BENCH:-$HOME/lanes/xcode/bench}; CPUS=$1; shift
for spec in "$@"; do set -- $spec
  out=gkx/cert_$1_ky$2_nl16_nm48.txt; [ -f $out ] && grep -q RESULT $out && continue
  OMP_NUM_THREADS=4 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= GKX_X64=1 \
    nice -n 10 taskset -c $CPUS ${PY:-python} gkx_cert.py $1 $2 16 48 > $out 2>&1
done
