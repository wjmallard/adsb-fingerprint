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
MODEL_DIR = Path(_config["paths"]["models"]).expanduser()

CENTER_FREQ_HZ = int(_config["radio"]["center_freq_hz"])
SAMPLE_RATE_HZ = int(_config["radio"]["sample_rate_hz"])
RADIO_GAIN = _config["radio"]["gain"]
FREQ_CORRECTION_PPM = int(_config["radio"]["freq_correction_ppm"])

COLLECT_MAX_PER_AIRCRAFT = int(_config["collect"]["max_per_aircraft"])
COLLECT_WINDOW_SECONDS = int(_config["collect"]["window_seconds"])

# Optional surveyed receiver position. Decoding never needs it; adsb-collect
# locates the station from the traffic and only trusts these coordinates for
# the session's receiver stamp while they agree with that estimate.
_receiver = _config.get("receiver") or {}
RECEIVER_LAT = float(_receiver["latitude"]) if "latitude" in _receiver else None
RECEIVER_LON = float(_receiver["longitude"]) if "longitude" in _receiver else None

_gps = _config.get("gps", {})
GPS_PORT = _gps.get("port", "/dev/cu.usbmodem*")
GPS_TRACK_DIR = Path(_gps.get("tracks", "~/radio-classifier-data/gps/tracks")).expanduser()

_server = _config.get("server", {})
SERVER_HOST = _server.get("host", "127.0.0.1")
SERVER_PORT = int(_server.get("port", 5050))
SERVER_DEBUG = bool(_server.get("debug", True))

_map = _config.get("map", {})
MAP_TILES_PATH = Path(_map.get("tiles_file", "~/Github/offline-maps/data/basemap.pmtiles")).expanduser()
MAP_TILES_FILE = MAP_TILES_PATH.name
MAP_DEFAULT_ZOOM = _map.get("default_zoom", 8)
MAP_ROSTER_MINUTES = int(_map.get("roster_minutes", 15))
MAP_RINGS_KM = _map.get("rings_km", [50, 100, 150])
