#!/usr/bin/env bash
# GPU A/B/A/B for PERF-ADJ on one free device. Usage (repository root):
#   CUDA_VISIBLE_DEVICES=<free> bash plan/research/scripts/2026-09-22-perf-adj/run_gpu.sh OUTDIR PY
# PY: python of a jax[cuda12]==0.10.2 environment. Complex64 (the production precision).
set -u
out=$1; py=$2
mkdir -p "$out"
export PYTHONPATH=$PWD/src JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
d=plan/research/scripts/2026-09-22-perf-adj
v=main,stage,barrier,block,mainblock
run() { local tag=$1; shift
  JAX_ENABLE_X64=false GKX_X64=0 "$py" $d/bench_ab.py "$@" --variants $v --out "$out/$tag.json" > "$out/$tag.log" 2>&1
  grep -E "vjp" "$out/$tag.log"; }
for ky in half full; do run gpu_f32_16_256_$ky --steps 256 --ky $ky --rounds 3; done
for ky in half full; do run gpu_f32_32_256_$ky --Nx 32 --Ny 32 --Nz 24 --steps 256 --ky $ky --rounds 3; done
for ky in half full; do run gpu_f32_16_1024_$ky --steps 1024 --ky $ky --rounds 3; done
echo DONE
