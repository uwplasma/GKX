import os

import pytest


@pytest.fixture(autouse=True)
def _x64_for_jax():
    os.environ.setdefault("JAX_ENABLE_X64", "true")
