#!/bin/bash
R=${BENCH:-$HOME/lanes/xcode/bench}/gxref; BIN=<gx-repaired>/gx
for c in salpha:itg_salpha_adiabatic_electrons miller:itg_miller_adiabatic_electrons kbm:kbm_miller miller_ke:itg_miller_kinetic_electrons; do
  d=$R/${c%%:*}; f=${c##*:}.in
  [ -f $d/DONE ] && continue
  $R/gpu_run.sh $d $BIN $f > $d/supervisor.log 2>&1; echo "rc=$?" > $d/DONE
done
echo ALLDONE > $R/ALLDONE
