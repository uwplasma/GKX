import tomllib

from spectraxgk.io_config import _preprocess_toml_expressions, read_toml


def test_preprocess_arithmetic_literals():
    s = "n0 = 0.5*10000\nval = 2*pi\n"
    pre = _preprocess_toml_expressions(s)
    d = tomllib.loads(pre)
    assert isinstance(d["n0"], str) and d["n0"] == "0.5*10000"
    assert isinstance(d["val"], str) and d["val"] == "2*pi"


def test_read_toml_exprs(tmp_path):
    p = tmp_path / "in.toml"
    p.write_text(
        "[sim]\nmode='fourier'\nbackend='eig'\ntmax=1.0\nnt=4\n"
        "[grid]\nL_lambdaD=2*pi\nNx=8\ndebye_species='e'\n"
        "[hermite]\nN=8\n[bc]\nkind='periodic'\n[plot]\n"
        "[[species]]\nname='e'\nq=-1\nn0=1e6\nmass_base='electron'\nmass_multiple=1\n"
        "temperature_eV=1\ndrift_c=0.0\n"
    )
    cfg = read_toml(str(p))
    assert cfg.grid.L_lambdaD is not None
    assert cfg.species[0].m > 0
    assert cfg.species[0].vth > 0
