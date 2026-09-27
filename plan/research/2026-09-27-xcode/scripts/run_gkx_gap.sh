#!/bin/bash
cd ${BENCH:-$HOME/lanes/xcode/bench}
for spec in "base 8 24" "nodrift 8 24" "nomirror 8 24" "nodrift_nomirror 8 24" "nodrift 16 48" "nomirror 16 48"; do
  set -- $spec
  OMP_NUM_THREADS=4 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
   GKX_X64=1 nice -n 10 taskset -c 24-27 ~/lanes/xcode/venv/bin/python gkx_gap.py $1 $2 $3 > gkx/gap_$1_nl$2_nm$3.txt 2>&1
done
echo ALLDONE >> gkx/done.txt
