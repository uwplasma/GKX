#!/bin/bash
# Q10 timing campaign launcher, run on the office host.
#
# Cores: the 2026-09-20 window had exactly two busy CPUs, 11 and 21, whose
# physical cores are 11 (siblings 11,29) and 3 (siblings 3,21).  This pins to
# CPUs 12-17 -- six physical cores whose hyperthread siblings 30-35 are also
# idle -- and gates on all twelve.  Re-derive the pin before reusing this:
# `python3 corebusy.py 8` prints per-core busy, and
# /sys/devices/system/cpu/cpuN/topology/thread_siblings_list prints the pairs.
cd "$(dirname "$0")"
DIR=$PWD
TREE=${TREE:?set TREE to the pinned checkout}
export PY=${PY:-python}
export LOADMAX=${LOADMAX:-6} BUSYMEAN=${BUSYMEAN:-0.05} BUSYWORST=${BUSYWORST:-0.20}
CORES=${CORES:-12-17}
CHECK=${CHECK:-12-17,30-35}
BLOCKS=${BLOCKS:-4}

EXPECT_KERNELS=3 ./run_timing_ab.sh "$DIR" "$TREE" "$CORES" "$CHECK" "$BLOCKS" \
  "--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 7 --scan-steps 5 --kernels rhs,rhs_vjp,scan_rk3" \
  nopool_g64
EXPECT_KERNELS=4 ./run_timing_ab.sh "$DIR" "$TREE" "$CORES" "$CHECK" "$BLOCKS" \
  "--Nx 32 --Ny 32 --Nz 24 --Nl 4 --Nm 8 --reps 7 --scan-steps 5 --window-reps 2 --window-steps 6 --kernels rhs,rhs_vjp,scan_rk3,window_vjp" \
  nopool_g32
echo DRIVER_DONE
