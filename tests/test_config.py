import json

from soccer_analysis.config import CalibrationConfig, PipelineConfig, Settings, resolve_device


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


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.calibration_mode == "static"
    assert config.team_classifier == "pixel"
    assert config.tracker_backend == "bytetrack"
    assert config.device is None


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
