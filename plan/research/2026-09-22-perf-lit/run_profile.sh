#!/usr/bin/env bash
# Run the PERF-LIT profile set serially, one fresh process per measurement.
# Usage: run_profile.sh <outdir> <python> [gpu|cpu]
# From the repository root of a clean checkout at the pinned SHA.
set -u
OUT=$1; PY=$2; DEV=${3:-cpu}
mkdir -p "$OUT"
S=plan/research/2026-09-22-perf-lit/profile_perf_lit.py
export PYTHONPATH=$PWD/src
if [ "$DEV" = gpu ]; then
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
else
  export JAX_PLATFORMS=cpu
fi
run() { # label, args...
  local label=$1; shift
  echo "start $label $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg)" >> "$OUT/supervisor.txt"
  "$PY" "$S" "$@" --label "$label" --out "$OUT/$label.json" > "$OUT/$label.txt" 2>&1
  echo "end $label rc=$? $(date -u +%FT%TZ)" >> "$OUT/supervisor.txt"
}
if [ "${PHASE:-base}" = base ]; then
for grid in "32 32 24" "64 64 24"; do
  set -- $grid
  for mom in "4 8" "8 16"; do
    m=($mom)
    L=fwd_${1}x${2}x${3}_l${m[0]}m${m[1]}
    run $L forward --nx $1 --ny $2 --nz $3 --nl ${m[0]} --nm ${m[1]} --repeats 7 --trace-dir "$OUT/trace_$L"
    run fft_${L#fwd_} fftfloor --forward-json "$OUT/$L.json" --repeats 7
  done
done
run win16_256 window --nx 16 --ny 16 --nz 16 --steps 256 --checkpoint block none --repeats 3
run win16_1024 window --nx 16 --ny 16 --nz 16 --steps 1024 --checkpoint block --repeats 3
run win32_256 window --nx 32 --ny 32 --nz 24 --steps 256 --checkpoint block --repeats 3
run eig_l2m3z24 eigen --nz 24 --nl 2 --nm 3 --repeats 5
run eig_l4m8z48 eigen --nz 48 --nl 4 --nm 8 --repeats 3
run eig_l4m8z96 eigen --nz 96 --nl 4 --nm 8 --repeats 3
run win16_1024_nockpt window --nx 16 --ny 16 --nz 16 --steps 1024 --checkpoint none --repeats 2
fi
if [ "${PHASE:-base}" = extra ]; then
# Window variants in one process each, so value and gradient are compared
# arm-to-arm: block (shipped), block_noinner (block checkpoint, no per-step
# remat inside the block: one forward recompute, not two), none (no remat).
run win16_256_ab window --nx 16 --ny 16 --nz 16 --steps 256 --checkpoint block block_noinner none --repeats 2 --interleave 7
run win16_1024_ab window --nx 16 --ny 16 --nz 16 --steps 1024 --checkpoint block block_noinner --repeats 2 --interleave 7
run win32_256_ab window --nx 32 --ny 32 --nz 24 --steps 256 --checkpoint block block_noinner --repeats 2 --interleave 7
run win32_1024_ab window --nx 32 --ny 32 --nz 24 --steps 1024 --checkpoint block block_noinner --repeats 2 --interleave 5
run win16_256_x64 window --nx 16 --ny 16 --nz 16 --steps 256 --checkpoint block --precision 64 --repeats 3
run fwd_32x32x24_l4m8_x64 forward --nx 32 --ny 32 --nz 24 --nl 4 --nm 8 --precision 64 --repeats 7
run fwd_32x32x24_l4m8_src forward --nx 32 --ny 32 --nz 24 --nl 4 --nm 8 --repeats 3 --trace-dir "$OUT/trace_src32"
fi
if [ "${PHASE:-base}" = src ]; then
# Kernel -> source-line mapping of the top device kernels (timings unused).
run fwd_32x32x24_l4m8_srcmap forward --nx 32 --ny 32 --nz 24 --nl 4 --nm 8 --repeats 1 --trace-dir "$OUT/trace_srcmap32"
fi
if [ "${PHASE:-base}" = cpu ]; then
# CPU rows (office host, all cores visible to XLA; host load is recorded).
run cpu_smoke window --nx 8 --ny 8 --nz 8 --steps 16 --checkpoint block block_noinner none --repeats 1 --interleave 2
run cpu_win16_256_ab window --nx 16 --ny 16 --nz 16 --steps 256 --checkpoint block block_noinner none --repeats 2 --interleave 7
run cpu_win16_1024_ab window --nx 16 --ny 16 --nz 16 --steps 1024 --checkpoint block block_noinner --repeats 2 --interleave 5
run cpu_fwd_32x32x24_l4m8 forward --nx 32 --ny 32 --nz 24 --nl 4 --nm 8 --repeats 5
run cpu_eig_l2m3z24 eigen --nz 24 --nl 2 --nm 3 --repeats 5
run cpu_eigadapt_l2m3z24_x64 eigen-adaptive --nz 24 --nl 2 --nm 3 --precision 64
fi
echo done >> "$OUT/supervisor.txt"
