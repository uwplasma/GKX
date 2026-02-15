import numpy as np
from spectraxgk_old.post import save_summary
from spectraxgk_old.types import Result


def test_save_summary_png(tmp_path):
    # Tiny fake result
    nt, Nn, Nm = 10, 3, 2
    t = np.linspace(0.0, 1.0, nt)
    C = (np.random.randn(nt, Nn, Nm) + 1j*np.random.randn(nt, Nn, Nm)) * 1e-3
    meta = {"grid": {"vth": 1.0, "kpar": 0.5}}
    res = Result(t=t, C=C, meta=meta)
    out = tmp_path / "summary.png"
    save_summary(res, str(out))
    assert out.exists() and out.stat().st_size > 0
