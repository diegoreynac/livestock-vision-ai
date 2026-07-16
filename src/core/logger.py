"""
Project Logger

Provides a unified logging interface for the Livestock Weight Estimation project.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path


class ProjectLogger:
    """
    Wrapper around Python's logging module.

    Responsible only for writing log messages.
    """

    def __init__(
        self,
        logs_directory: Path,
        experiment_name: str,
        level: int = logging.INFO,
    ) -> None:

        self.log_file = logs_directory / f"{experiment_name}.log"

        self._logger = logging.getLogger(experiment_name)
        self._logger.setLevel(level)

        # Remove previous handlers if they exist
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s"
        )

        file_handler = logging.FileHandler(
            self.log_file,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)

    # =====================================================
    # Basic logging
    # =====================================================

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def exception(self, exception: Exception) -> None:
        """
        Log an exception together with its traceback.
        """

        self._logger.error(str(exception))
        self._logger.error(traceback.format_exc())

    # =====================================================
    # Formatting helpers
    # =====================================================

    def section(self, title: str) -> None:

        self.info("")
        self.info("=" * 80)
        self.info(title.upper())
        self.info("=" * 80)

    def key_value(
        self,
        key: str,
        value,
    ) -> None:

        self.info(f"{key:<30}: {value}")

    # =====================================================
    # Project lifecycle
    # =====================================================

    def start(self) -> None:

        self.section("Project Started")

        self.key_value("Log File", self.log_file)

    def finish(self) -> None:

        self.section("Project Finished")

    # =====================================================
    # Cleanup
    # =====================================================

    def close(self) -> None:

        handlers = self._logger.handlers[:]

        for handler in handlers:

            handler.close()

            self._logger.removeHandler(handler)