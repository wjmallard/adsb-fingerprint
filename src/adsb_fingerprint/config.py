"""Load config.yaml into module-level constants."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

_config_path = PROJECT_ROOT / "config.yaml"
if not _config_path.exists():
    raise FileNotFoundError(
        f"{_config_path} not found — copy config.yaml.example to config.yaml and edit it.",
    )

_config = yaml.safe_load(_config_path.read_text())

DBNAME = _config["database"]["dbname"]

CAPTURE_DIR = Path(_config["paths"]["captures"]).expanduser()

CENTER_FREQ_HZ = int(_config["radio"]["center_freq_hz"])
SAMPLE_RATE_HZ = int(_config["radio"]["sample_rate_hz"])
RADIO_GAIN = _config["radio"]["gain"]
FREQ_CORRECTION_PPM = int(_config["radio"]["freq_correction_ppm"])

COLLECT_MAX_PER_AIRCRAFT = int(_config["collect"]["max_per_aircraft"])
COLLECT_WINDOW_SECONDS = int(_config["collect"]["window_seconds"])
