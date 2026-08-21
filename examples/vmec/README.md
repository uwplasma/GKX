# VMEC-JAX Geometry Inputs

This directory contains small VMEC input decks copied from the `vmex`
example set so GKX examples can be run from a normal clone without
shipping large `wout_*.nc` files.

Generate the needed equilibrium files with:

```bash
pip install vmec-jax
cd examples/vmec
vmex input.circular_tokamak
vmex input.NuhrenbergZille_1988_QHS
vmex input.nfp3_QI_fixed_resolution_final
vmex input.LandremanPaul2021_QA_lowres
vmex input.LandremanPaul2021_QH_reactorScale_lowres
vmex input.QI_stel_seed_3127
vmex input.li383_low_res
```

The command writes `wout_<input-name-without-input.>.nc` next to the input
deck. The checked-in GKX TOMLs use relative paths to those expected
outputs, for example:

- `wout_circular_tokamak.nc`
- `wout_NuhrenbergZille_1988_QHS.nc`
- `wout_nfp3_QI_fixed_resolution_final.nc`
- `wout_LandremanPaul2021_QA_lowres.nc`
- `wout_LandremanPaul2021_QH_reactorScale_lowres.nc`
- `wout_QI_stel_seed_3127.nc`
- `wout_li383_low_res.nc`

The bundled QHS/QI/QA decks are self-contained demonstration equilibria. For
paper-specific HSX or W7-X validation, use the same TOMLs but override
`geometry.vmec_file` or pass `--vmec-file` with the exact validation
equilibrium.
