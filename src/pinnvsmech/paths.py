"""Standard locations in the repository (works when the package is installed with `pip install -e .`)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
TRAINING_DIR = DATA_DIR / "training"
EPI_DIR = DATA_DIR / "epi"

PRECOMPUTED_DIR = REPO_ROOT / "precomputed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR = REPO_ROOT / "tables"
