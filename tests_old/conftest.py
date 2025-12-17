# Global test config: headless matplotlib and deterministic JAX
import os
import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("JAX_ENABLE_X64", "true")
