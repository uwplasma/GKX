#!/bin/bash
# Q9 idle-host campaign driver (office, cores 2-17 only).
cd "$(dirname "$0")"
DIR=$PWD
export PY=${PY:-/home/rjorge/venvs/gkx-nl/bin/python}
# This lane owns cores 2-17. At 19:02 another lane's DKX bench took cores
# 8-11 (`taskset -pc` confirmed an affinity of 8-11) and held them, so the
# campaign runs on the free subset 2-7,12-17: 12 cores, still inside this
# lane's allocation, identical for both arms. The 4 cores lent out are
# excluded from the idleness gate as well, so the gate describes the cores
# the benchmark actually runs on.
CORES=${CORES:-2-7,12-17}
export CHECK_CORES=${CHECK_CORES:-$CORES} SIB_CORES=${SIB_CORES:-20-25,30-35}
# Office is otherwise quiet: 1-min load ~1.2, from one root process pinned
# to cpu18 (the hyperthread sibling of cpu0, outside cores 2-17).
# The idleness gate that decides whether an arm may start is the per-core
# one (cores 2-17 below 10% mean and 50% worst over a 3 s sample); it
# measures the machine as it is at that instant. The 1-minute load gate is
# left off (LOADMAX=1000) because after each arm the load average is
# dominated by this campaign's own threads decaying, which would stall
# the run without saying anything about contention. Load is still recorded
# at every arm.
export LOADMAX=${LOADMAX:-1000} BUSYMEAN=${BUSYMEAN:-0.10} BUSYWORST=${BUSYWORST:-0.50}
G64="--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 7 --window-reps 7"
G32="--Nx 32 --Ny 32 --Nz 24 --Nl 4 --Nm 8 --reps 7 --window-reps 7"
EXPECT_KERNELS=4 ./run_ab_rot.sh "$DIR" "$CORES" "${BLOCKS64:-4}" "$G64" pool pool64
EXPECT_KERNELS=3 ./run_ab_rot.sh "$DIR" "$CORES" "${BLOCKSNP:-4}" \
  "--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 7 --scan-steps 2 --kernels rhs,rhs_vjp,scan_rk3" nopool nopool64
EXPECT_KERNELS=4 ./run_ab_rot.sh "$DIR" "$CORES" "${BLOCKS32:-4}" "$G32" pool pool32
echo DRIVER_DONE
