"""Command-line entry point: ``soccer-analysis process ...``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from soccer_analysis.config import CalibrationConfig, Settings
from soccer_analysis.logging_utils import configure_logging
from soccer_analysis.pipeline import EXPORT_FORMATS, run_pipeline

app = typer.Typer(add_completion=False, help="Soccer match footage -> ML-ready tracking data.")
logger = logging.getLogger(__name__)


@app.command()
def process(
    input_video: Path = typer.Option(..., "--input", exists=True, help="Match footage to process."),
    model: Path = typer.Option(..., "--model", exists=True, help="YOLO detection checkpoint."),
    calibration: Path = typer.Option(
        ..., "--calibration", exists=True, help="Pitch calibration JSON for this camera angle."
    ),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Where outputs are written."),
    formats: list[str] = typer.Option(
        ["jsonl", "parquet", "summary"],
        "--format",
        help=f"Repeatable. One or more of: {', '.join(EXPORT_FORMATS)}. "
        "'video' renders an annotated video; the rest write the ML-ready data export.",
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="YAML config overriding configs/default.yaml."
    ),
    use_cache: bool = typer.Option(
        False, "--cache/--no-cache", help="Reuse cached intermediate tracking results if present."
    ),
    log_level: str = typer.Option("INFO", help="Logging verbosity."),
) -> None:
    """Run the full detect -> track -> analyze -> export/render pipeline on one video."""
    configure_logging(log_level)

    unknown = set(formats) - set(EXPORT_FORMATS)
    if unknown:
        raise typer.BadParameter(f"Unknown format(s) {sorted(unknown)}; choose from {EXPORT_FORMATS}")

    settings = Settings.from_yaml(config_path) if config_path else Settings()
    calibration_config = CalibrationConfig.from_json_file(calibration)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{input_video.stem}_annotated.mp4" if "video" in formats else None
    data_formats = [f for f in formats if f != "video"]

    result = run_pipeline(
        video_path=input_video,
        model_path=model,
        calibration=calibration_config,
        config=settings.pipeline,
        stub_dir=output_dir / "cache" if use_cache else None,
        read_from_stub=use_cache,
        output_video_path=output_video_path,
        export_dir=output_dir if data_formats else None,
        export_formats=data_formats,
    )

    logger.info("Processed %d frames.", len(result.tracks["players"]))
    if output_video_path is not None:
        logger.info("Annotated video written to %s", output_video_path)
    if data_formats:
        logger.info("Data export (%s) written to %s", ", ".join(data_formats), output_dir)


if __name__ == "__main__":
    app()
