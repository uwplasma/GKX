#!/bin/bash
# Supplementary: high-rep RHS / RHS-gradient A/B/A/B, same rotation and same
# gate rule, to resolve the bimodal per-rep distribution the 7-rep blocks show
# on the small RHS kernel. Registered before these runs: same pass/faster/
# slower rule as ab_table2.py, applied to the dense blocks.
cd "$(dirname "$0")"
DIR=$PWD
export PY=/home/rjorge/venvs/gkx-nl/bin/python
CORES=2-7,12-17
export CHECK_CORES=$CORES SIB_CORES=20-25,30-35
export LOADMAX=1000 BUSYMEAN=0.10 BUSYWORST=0.50
EXPECT_KERNELS=2 ./run_ab_rot.sh "$DIR" "$CORES" 6 "--Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 --reps 31 --kernels rhs,rhs_vjp" pool dense64
EXPECT_KERNELS=2 ./run_ab_rot.sh "$DIR" "$CORES" 6 "--Nx 32 --Ny 32 --Nz 24 --Nl 4 --Nm 8 --reps 61 --kernels rhs,rhs_vjp" pool dense32
echo DENSE_DONE
