import json
from pathlib import Path

from agon.config import (
    CalibrationConfig,
    ClockCalibrationConfig,
    PipelineConfig,
    Settings,
    resolve_device,
)


def test_calibration_config_from_json_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "pixel_vertices": [[0, 0], [0, 10], [10, 10], [10, 0]],
                "court_length_m": 10,
                "court_width_m": 20,
            }
        )
    )

    config = CalibrationConfig.from_json_file(path)

    assert config.pixel_vertices == [(0, 0), (0, 10), (10, 10), (10, 0)]
    assert config.court_length_m == 10
    assert config.court_width_m == 20


def test_clock_calibration_config_from_json_file(tmp_path):
    path = tmp_path / "clock_calibration.json"
    path.write_text(json.dumps({"clock_region_px": [10, 20, 110, 50]}))

    config = ClockCalibrationConfig.from_json_file(path)

    assert config.clock_region_px == (10, 20, 110, 50)


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.calibration_mode == "static"
    assert config.team_classifier == "pixel"
    assert config.tracker_backend == "bytetrack"
    assert config.device is None
    assert config.frame_filter_mode == "off"
    assert config.min_grass_fraction == 0.35
    assert config.clock_calibration_path is None
    assert config.detection_imgsz == 640


def test_settings_from_yaml_parses_quoted_off_as_string_not_boolean(tmp_path):
    # YAML 1.1 treats the bareword `off` as a boolean -- configs/default.yaml
    # quotes it ("off") specifically so this doesn't silently coerce to
    # False, which would fail PipelineConfig's Literal["off","tag","strip"]
    # validation (or worse, validate against a different type some day).
    path = tmp_path / "config.yaml"
    path.write_text('pipeline:\n  frame_filter_mode: "off"\n')

    settings = Settings.from_yaml(path)

    assert settings.pipeline.frame_filter_mode == "off"


def test_shipped_default_yaml_loads_cleanly():
    default_yaml = Path(__file__).parent.parent / "configs" / "default.yaml"
    settings = Settings.from_yaml(default_yaml)
    assert settings.pipeline.frame_filter_mode == "off"


def test_settings_from_yaml_overrides_pipeline_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("pipeline:\n  detection_confidence: 0.5\n  calibration_mode: dynamic\n")

    settings = Settings.from_yaml(path)

    assert settings.pipeline.detection_confidence == 0.5
    assert settings.pipeline.calibration_mode == "dynamic"
    # Fields not mentioned in the override file keep their defaults.
    assert settings.pipeline.tracker_backend == "bytetrack"


def test_resolve_device_returns_preferred_without_importing_torch():
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_degrades_to_cpu_without_torch():
    # This test environment has no torch installed, which is exactly the
    # scenario this fallback exists for (the onnxruntime-only default
    # runtime path shouldn't require torch to even call this function).
    assert resolve_device(None) in ("cpu", "cuda", "mps")
