# Outputs & Diagnostics

The main entry point returns a Python dict `out` containing parameters, metadata, and saved time series.

## Time

- `out["time"]`: saved times (Diffrax `SaveAt` times)

## Scalar diagnostics (typical)

- `phi_rms`: RMS of `|phi_k|` over k-space
- `max_abs_phi`: max `|phi_k|`
- `max_abs_G`: max `|Gk|`
- `W_free` (or `W_total`): free-energy-like quadratic quantity
- `W_field`: field/polarization energy component
- `W_s` / `W_g_s`: per-species contributions

## Spectral diagnostics

- `E_m`: Hermite spectrum (shape `(Ns, Nh)`), computed from `|G|^2` summed over ℓ and k.

## Probes

If configured, the diagnostics function also saves:
- `phi_probe_re`, `phi_probe_im`
- `Gm0_probe_re_s`, `Gm0_probe_im_s`

and for 1D line setups (`Ny==Nx==1`):
- `phi_k_line_re`, `phi_k_line_im`
- `n_s_k_line_re`, `n_s_k_line_im` (per species)

## What is *not* saved

To keep Diffrax output trees real-valued and small:
- complex arrays are split into real/imag,
- large complex states are not saved unless `save="final"` and you explicitly store them.

## Reproducibility

The returned dict also includes the full `params` (including k-grids, masks, gyroaverage arrays),
which can be used to postprocess, re-run, or validate.