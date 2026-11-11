"""Two kinetic species with different velocity resolution on several devices.

``case.toml`` gives the ions ``(Nl, Nm) = (2, 8)`` and the electrons the state's
``(4, 16)``. With ``[parallel] strategy = "shard_map"`` and ``axis =
"species_hermite"`` GKX does not put one species on each device, which would load
the electron device with 64 entries to the ion device's 16. It cuts the Hermite
axis of *each* species group into one block per device, so a device holds
``ns * nl * nm / devices`` entries of every group: 40 each on two devices, whatever
the counts. The blocks exchange a width-2 Hermite halo and one field reduction per
right-hand side. The script runs the case on one device and then sharded, prints
where the work went, and checks that the final states agree. Two host CPU devices
stand in for two GPUs; the flag below creates them and is not needed on a machine
with two or more GPUs. A few seconds on a laptop CPU.
"""

import json
import os
from pathlib import Path

# Two logical CPU devices. Must be set before JAX starts; delete on real GPUs.
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()

import jax
import matplotlib.pyplot as plt
import numpy as np

import gkx
from gkx.config import RuntimeParallelConfig

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/14_multi_device_species")
DEVICES = 2  # Hermite counts (8 and 16) must divide by this; 8 / 4 = 2 is the limit

case = gkx.load(CASE)
run = case.run
counts = {  # (Nl, Nm) each species keeps; an unset count keeps [run]'s
    s.name: (s.Nl or run.Nl, s.Nm or run.Nm) for s in case.species if s.kinetic
}
print("species moments (Nl, Nm):", counts)

# Work per device. Entries of the packed state a device advances per right-hand side.
total = sum(nl * nm for nl, nm in counts.values())
padded = len(counts) * run.Nl * run.Nm
print(
    f"{DEVICES} devices, each cut of every species' Hermite axis in {DEVICES}: "
    f"{total / DEVICES:g} Nl*Nm entries per device "
    f"(a padded state would carry {padded / DEVICES:g}; "
    f"one species per device would carry {max(nl * nm for nl, nm in counts.values())}"
    " on the busiest)"
)

options = dict(
    ky_target=run.ky, Nl=run.Nl, Nm=run.Nm, diagnostics=False, return_state=True
)
serial = gkx.solve(case, **options)  # the deck as shipped is serial

if len(jax.devices()) < DEVICES:
    raise SystemExit(f"needs {DEVICES} JAX devices, found {len(jax.devices())}")
# The same case with a [parallel] block; the equivalent TOML is
#   [parallel]
#   strategy = "shard_map"        one SPMD program over a device mesh
#   axis = "species_hermite"      mesh over species and Hermite moments
#   num_devices = 2
#   strict_identity = false       skip the built-in serial re-run; checked below
case = case.replace(
    parallel=RuntimeParallelConfig(
        strategy="shard_map",
        axis="species_hermite",
        num_devices=DEVICES,
        strict_identity=False,
    )
)
messages = []
sharded = gkx.solve(case, status_callback=messages.append, **options)
for line in messages:
    if line.startswith(("routing nonlinear", "using named", "using whole")):
        print(line)

state, reference = np.asarray(sharded.state), np.asarray(serial.state)
difference = float(np.max(np.abs(state - reference)))
print(f"largest difference from the serial state: {difference:.2e}")
for i, (name, (nl, nm)) in enumerate(counts.items()):
    kept = np.abs(state[i, :nl, :nm]).max()
    above = max(
        np.abs(state[i, nl:]).max(initial=0), np.abs(state[i, :, nm:]).max(initial=0)
    )
    print(f"{name}: largest kept moment {kept:.2e}, above its counts {above:.1e}")

OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {
    "moments": {k: list(v) for k, v in counts.items()},
    "devices": DEVICES,
    "entries_per_device": total / DEVICES,
    "entries_per_device_padded": padded / DEVICES,
    "max_abs_difference_from_serial": difference,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, ax = plt.subplots(figsize=(6.0, 3.4))
hermite = np.abs(state).sum(axis=(1, 3, 4, 5))  # (species, Hermite order)
for i, name in enumerate(counts):
    ax.semilogy(np.arange(hermite.shape[1]), hermite[i] + 1e-30, "o-", label=name)
ax.set(xlabel="Hermite order m", ylabel=r"$\sum |G|$", title="Hermite spectrum")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "multi_device_species.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/multi_device_species.png")
