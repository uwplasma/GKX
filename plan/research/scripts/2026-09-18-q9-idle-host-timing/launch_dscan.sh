#!/bin/bash
# Supplementary: denser rk3 scan blocks (the kernel closest to production
# throughput), same rotation and same registered gate rule.
cd "$(dirname "$0")"
DIR=$PWD
export PY=/home/rjorge/venvs/gkx-nl/bin/python
CORES=2-7,12-17
export CHECK_CORES=$CORES SIB_CORES=20-25,30-35
export LOADMAX=1000 BUSYMEAN=0.10 BUSYWORST=0.50
EXPECT_KERNELS=1 ./run_ab_rot.sh "$DIR" "$CORES" 6 "--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 15 --kernels scan_rk3" pool dscan64
EXPECT_KERNELS=1 ./run_ab_rot.sh "$DIR" "$CORES" 6 "--Nx 32 --Ny 32 --Nz 24 --Nl 4 --Nm 8 --reps 21 --kernels scan_rk3" pool dscan32
echo DSCAN_DONE
