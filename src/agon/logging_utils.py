"""Shared logging setup for the pipeline and CLI."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
