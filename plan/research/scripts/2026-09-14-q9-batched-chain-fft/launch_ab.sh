#!/bin/bash
cd "$(dirname "$0")"
DIR=$PWD
export PY=${PY:-python} CHECK_CORES=2-17
# Office is contended by unpinned jobs (cores 2-17 at 46-70% mean busy at
# 22:02); the idleness gate is recorded per arm but no longer blocks.
export LOADMAX=1000 BUSYMEAN=1.01 BUSYWORST=1.01
EXPECT_KERNELS=4 ./run_ab.sh $DIR 2-17 3 "--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 7 --window-reps 3" pool
EXPECT_KERNELS=3 ./run_ab.sh $DIR 2-17 3 "--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 7 --scan-steps 2 --kernels rhs,rhs_vjp,scan_rk3" nopool
echo DRIVER_DONE
