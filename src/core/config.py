"""
Project configuration.

This module defines the global configuration used throughout the
Livestock Weight Estimation project.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ProjectConfig:
    """
    Global project configuration.

    Parameters
    ----------
    project_root : Path
        Root directory of the project.

    dataset_root : Path
        Root directory containing the livestock dataset.

    output_root : Path
        Root directory where experiment outputs are stored.
    """

    project_root: Path
    dataset_root: Path
    output_root: Path

    @classmethod
    def default(cls) -> "ProjectConfig":
        """
        Creates the default project configuration.
        """

        project_root = Path(__file__).resolve().parents[2]

        dataset_root = Path(
            r"C:\MSI\Dataset\cattle-weight"
        )

        output_root = project_root / "output"

        return cls(
            project_root=project_root,
            dataset_root=dataset_root,
            output_root=output_root
        )

    def validate(self) -> None:
        """
        Validate project directories.
        """

        if not self.project_root.exists():
            raise FileNotFoundError(
                f"Project directory not found:\n{self.project_root}"
            )

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.dataset_root}"
            )
    
    def create_output_directories(self) -> None:
        """
        Creates all output directories required by the project.
        """

        folders = [

            self.output_root,

            self.figures_folder,

            self.reports_folder,

            self.csv_folder,

            self.data_folder,

            self.models_folder

        ]

        for folder in folders:

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    @property
    def pixel_dataset(self) -> Path:
        """
        Pixel dataset root.
        """
        return self.dataset_root / "Pixel"

    @property
    def vector_dataset(self) -> Path:
        """
        Vector dataset root.
        """
        return self.dataset_root / "Vector"
    
    @property
    def figures_folder(self) -> Path:
        """
        Directory used to store generated figures.
        """

        return self.output_root / "figures"

    @property
    def reports_folder(self) -> Path:
        """
        Directory used to store generated reports.
        """

        return self.output_root / "reports"
    
    @property
    def csv_folder(self) -> Path:
        """
        Directory used to store CSV files.
        """

        return self.output_root / "csv"
    
    @property
    def data_folder(self) -> Path:
        """
        Directory used to store generated data files.
        """

        return self.output_root / "data"
    
    @property
    def models_folder(self) -> Path:

        return self.output_root / "models"