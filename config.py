"""Central configuration: paths and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
FIGURES_DIR = ROOT / "docs" / "images"

TARGET = "diagnosis"  # 1 = malignant, 0 = benign
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Data-quality thresholds used by the validation stage
MIN_ROWS = 100
MAX_MISSING_FRACTION = 0.20
MIN_CLASS_SHARE = 0.10
