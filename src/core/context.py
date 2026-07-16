"""
Project execution context.

Creates the execution environment used during a single run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import platform
import sys

from src.core.config import ProjectConfig
from src.core.logger import ProjectLogger


@dataclass(slots=True, frozen=True)
class ExecutionDirectories:
    """
    Stores all directories created for a single execution.
    """

    root: Path
    logs: Path
    csv: Path
    figures: Path
    reports: Path


class ProjectContext:
    """
    Represents one execution of the project.

    Every execution creates an isolated output directory.
    """

    def __init__(
        self,
        config: ProjectConfig,
        experiment_name: str,
    ) -> None:

        self.config = config
        self.experiment_name = experiment_name

        self.timestamp = datetime.now()

        execution_name = self.timestamp.strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        root = (
            self.config.output_root
            / experiment_name
            / execution_name
        )

        self.directories = ExecutionDirectories(

            root=root,

            logs=root / "logs",

            csv=root / "csv",

            figures=root / "figures",

            reports=root / "reports",

        )

        self._create_directories()

        self.logger = ProjectLogger(

            logs_directory=self.logs_dir,

            experiment_name=experiment_name

        )

        self.logger.start()

        self._write_metadata()

    # ======================================================
    # Properties
    # ======================================================

    @property
    def output_dir(self) -> Path:
        return self.directories.root

    @property
    def logs_dir(self) -> Path:
        return self.directories.logs

    @property
    def csv_dir(self) -> Path:
        return self.directories.csv

    @property
    def figures_dir(self) -> Path:
        return self.directories.figures

    @property
    def reports_dir(self) -> Path:
        return self.directories.reports

    # ======================================================
    # Private methods
    # ======================================================

    def _create_directories(self) -> None:

        for directory in (
            self.directories.root,
            self.directories.logs,
            self.directories.csv,
            self.directories.figures,
            self.directories.reports,
        ):

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    def _write_metadata(self) -> None:

        metadata = {

            "project": "LivestockWeight",

            "experiment": self.experiment_name,

            "execution_time": self.timestamp.isoformat(),

            "python_version": platform.python_version(),

            "platform": platform.platform(),

            "dataset_root": str(self.config.dataset_root),

            "output_directory": str(self.output_dir),

        }

        metadata_path = self.output_dir / "metadata.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

    # ======================================================
    # Public methods
    # ======================================================

    def close(self) -> None:

        self.logger.finish()

        self.logger.close()