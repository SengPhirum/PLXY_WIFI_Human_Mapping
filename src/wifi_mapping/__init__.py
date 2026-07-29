"""Wi-Fi Human Mapping — privacy-preserving indoor human localization using CSI.

Core package layout:

- ``config``      : typed experiment configuration loaded from YAML
- ``simulate``    : geometry-based multipath CSI simulator (demo without hardware)
- ``collect``     : ESP32 serial CSI reader + session recording / ground truth
- ``preprocess``  : filtering, phase sanitization, normalization, windowing, features
- ``models``      : RSSI/CSI baselines (sklearn) and optional deep models (torch)
- ``tracking``    : Kalman smoothing of sequential position estimates
- ``eval``        : leakage-aware dataset splits and localization metrics
- ``server``      : FastAPI real-time inference service + mapping dashboard
"""

__version__ = "0.1.0"
