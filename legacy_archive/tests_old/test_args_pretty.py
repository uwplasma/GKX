from spectraxgk_old.args import resolve_input_path
from spectraxgk_old.pretty import init_pretty


def test_resolve_input_path_variants(tmp_path):
    # Create a toml file under examples and cwd
    ex = tmp_path / "examples"
    ex.mkdir()
    (ex / "case.toml").write_text("[sim]\nnt=2\n")
    # Without extension should search examples
    p = resolve_input_path(str(ex / "case"))
    assert p.endswith("case") or p.endswith("case.toml")


def test_init_pretty_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    init_pretty(prefer_rich=True)  # should disable rich quietly
    # no assertions; just ensure no exception
