#!/bin/bash
# usage: lane.sh <cpus> "<tag> <S|M> <ky> <Nl> <Nm> <variant>" ...
# Fixed-step imex2 (the deck method; rk4 agrees to 1e-6 at Nm 8 and costs 6x) time runs at CFLF (default 0.7) x the explicit CFL estimate (6.14e-4 at Nm = 8, scaled by sqrt(8/Nm)), t_max 50.
CPUS=$1; shift
for spec in "$@"; do
  set -- $spec
  dt=$(python3 -c "print(f'{${CFLF:-0.7}*6.14e-4*(8/$5)**0.5:.3e}')")
  KBM_TIME=$dt,50 TMO=43200 $HOME/lanes/kbm-vel/run1.sh $CPUS $1 $2 $3 $4 $5 $6
done
