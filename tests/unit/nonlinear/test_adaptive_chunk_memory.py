"""The adaptive chunk loop must not accumulate what it is about to discard.

Moving each chunk's diagnostics to host memory bounds *device* residency, but it
moved the growth onto the host instead: the chunk list stays live until the final
concatenation, so host RAM grew with ``t_max`` and long production runs reached
hundreds of gigabytes.

The stride was the reason. Runtime configs sample sparsely -- the shipped Cyclone
nonlinear case uses ``diagnostics_stride = 20`` -- but the stride used to be
applied *after* concatenation, so nineteen of every twenty samples were
accumulated at full size and then thrown away. Striding each chunk on arrival
keeps the same samples for a twentieth of the peak.

That only holds if the stride phase carries across chunks, which is what the
first test pins: a chunk starting at global sample ``g`` must begin its stride at
``(-g) % stride``, or the kept set silently differs from what the old code
produced. The chunk length here is deliberately coprime with one stride and not
the other.

The second test pins the disk spill, for runs where even the strided series
outgrows RAM: it must be a pure storage choice, returning bit-identical
diagnostics.

Keeping the right samples is not enough if the kept chunk is a *view* onto the
chunk it was strided from, because then nothing is released and only the shapes
shrink. The third test pins that, at strides 7 and 20 where chunks fall to one
and zero samples.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pytest

from gkx.diagnostics.metadata import ResolvedDiagnostics, SimulationDiagnostics
from gkx.workflows.runtime import chunks as runtime_chunks
from gkx.workflows.runtime.chunks import run_adaptive_runtime_chunk_loop

CHUNK_SAMPLES = 7
CHUNK_COUNT = 5
DT = 0.5
T_MAX = (CHUNK_SAMPLES - 1) * DT * CHUNK_COUNT


def _chunk(index: int) -> SimulationDiagnostics:
    """A chunk whose payload carries its own global sample id."""

    sample_id = np.arange(CHUNK_SAMPLES, dtype=float) + index * CHUNK_SAMPLES
    zeros = np.zeros(CHUNK_SAMPLES)
    return SimulationDiagnostics(
        t=np.arange(CHUNK_SAMPLES, dtype=float) * DT,
        dt_t=np.full(CHUNK_SAMPLES, DT),
        dt_mean=np.asarray(DT),
        gamma_t=sample_id,
        omega_t=zeros,
        Wg_t=zeros,
        Wphi_t=zeros,
        Wapar_t=zeros,
        heat_flux_t=sample_id,
        particle_flux_t=zeros,
        energy_t=zeros,
        resolved=ResolvedDiagnostics(Phi2_kxt=sample_id.reshape(-1, 1)),
    )


def _run(*, stride: int, spill_dir: Path | None = None):
    remaining = iter(range(CHUNK_COUNT))

    def integrate_chunk(_show_progress):
        return None, _chunk(next(remaining)), object(), object()

    return run_adaptive_runtime_chunk_loop(
        integrate_chunk=integrate_chunk,
        t_max=T_MAX,
        chunk_steps=CHUNK_SAMPLES,
        label="memory-test",
        diagnostics_stride=stride,
        spill_dir=spill_dir,
    )


def _chunk_arrays(diag: SimulationDiagnostics) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, array)`` for every array on the chunk and its resolved payload."""

    for field in dataclass_fields(SimulationDiagnostics):
        if field.name == "resolved":
            continue
        value = getattr(diag, field.name)
        if value is not None:
            yield field.name, np.asarray(value)
    if diag.resolved is None:
        return
    for field in dataclass_fields(ResolvedDiagnostics):
        value = getattr(diag.resolved, field.name)
        if value is not None:
            yield f"resolved.{field.name}", np.asarray(value)


@pytest.mark.parametrize("stride", [1, 3, 4])
def test_per_chunk_stride_keeps_the_post_concatenation_samples(stride: int) -> None:
    kept = np.asarray(_run(stride=stride).diagnostics.gamma_t)
    expected = np.arange(CHUNK_SAMPLES * CHUNK_COUNT, dtype=float)[::stride]

    assert np.array_equal(kept, expected), (
        f"stride {stride} kept global samples {kept[:12]} but striding after "
        f"concatenation would keep {expected[:12]}; the phase is not carrying "
        "across chunk boundaries"
    )


def test_disk_spill_is_a_storage_choice_not_a_different_answer(tmp_path) -> None:
    in_ram = _run(stride=3).diagnostics
    on_disk = _run(stride=3, spill_dir=tmp_path / "spill").diagnostics

    assert np.array_equal(
        np.asarray(in_ram.gamma_t), np.asarray(on_disk.gamma_t)
    ), "spilling chunks to disk changed the retained time series"
    assert in_ram.resolved is not None and on_disk.resolved is not None
    assert np.array_equal(
        np.asarray(in_ram.resolved.Phi2_kxt), np.asarray(on_disk.resolved.Phi2_kxt)
    ), "the resolved payload did not survive the spill round trip"


@pytest.mark.parametrize("stride", [3, 7, 20])
def test_strided_chunks_own_their_samples(
    monkeypatch: pytest.MonkeyPatch, stride: int
) -> None:
    captured: list[SimulationDiagnostics] = []
    stride_chunk = runtime_chunks.stride_runtime_diagnostics

    def _capture(diag: SimulationDiagnostics, **kwargs: int) -> SimulationDiagnostics:
        strided = stride_chunk(diag, **kwargs)
        captured.append(strided)
        return strided

    monkeypatch.setattr(runtime_chunks, "stride_runtime_diagnostics", _capture)
    _run(stride=stride)

    assert len(captured) == CHUNK_COUNT, (
        f"captured {len(captured)} strided chunks but the loop runs {CHUNK_COUNT}; "
        "the stride is no longer applied once per chunk, so this test is not "
        "looking at the arrays it claims to be asserting on"
    )

    aliased = [
        f"chunk {index} {name}"
        for index, chunk in enumerate(captured)
        for name, arr in _chunk_arrays(chunk)
        if arr.base is not None
    ]
    assert not aliased, (
        f"stride {stride}: {len(aliased)} array(s) in the strided chunks are views "
        "onto the unstrided chunk they came from, so every discarded sample stays "
        "alive in the chunk list until the final concatenation and the per-chunk "
        f"stride frees nothing: {aliased}"
    )
