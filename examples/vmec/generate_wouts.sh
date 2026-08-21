#!/usr/bin/env bash
set -euo pipefail

if ! command -v vmex >/dev/null 2>&1; then
  echo "vmex is required. Install it with: pip install vmec-jax" >&2
  exit 1
fi

inputs=(
  input.circular_tokamak
  input.NuhrenbergZille_1988_QHS
  input.nfp3_QI_fixed_resolution_final
  input.LandremanPaul2021_QA_lowres
  input.LandremanPaul2021_QH_reactorScale_lowres
  input.QI_stel_seed_3127
  input.li383_low_res
)

for input in "${inputs[@]}"; do
  echo "Generating ${input/input./wout_}.nc from $input"
  vmex "$input"
done
