#!/usr/bin/env bash
# Sequential A/B/A/B campaign for PERF-ADJ. Usage (repository root):
#   bash plan/research/scripts/2026-09-22-perf-adj/run_ab.sh OUTDIR PY_011 PY_010
# PY_011: python of a jax/jaxlib 0.11.2 environment; PY_010: jax/jaxlib 0.10.2.
set -u
out=$1; py11=$2; py10=$3
mkdir -p "$out"
export PYTHONPATH=$PWD/src JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
d=plan/research/scripts/2026-09-22-perf-adj
run() {  # tag python x64 args...
  local tag=$1 py=$2 x64=$3; shift 3
  JAX_ENABLE_X64=$x64 GKX_X64=$([ "$x64" = true ] && echo 1 || echo 0) \
    "$py" $d/bench_ab.py "$@" --out "$out/$tag.json" > "$out/$tag.log" 2>&1
  grep -E "vjp" "$out/$tag.log"
}
for ky in half full; do
  run cpu011_x64_16_256_$ky "$py11" true --steps 256 --ky $ky --rounds 3
  run cpu010_x64_16_256_$ky "$py10" true --steps 256 --ky $ky --rounds 3
done
for ky in half full; do
  run cpu011_x64_32_256_$ky "$py11" true --Nx 32 --Ny 32 --Nz 24 --steps 256 --ky $ky --rounds 2
done
for ky in half full; do
  run cpu011_f32_16_256_$ky "$py11" false --steps 256 --ky $ky --rounds 3
done
for ky in half full; do
  run cpu011_x64_16_1024_$ky "$py11" true --steps 1024 --ky $ky --rounds 2
done
echo DONE
