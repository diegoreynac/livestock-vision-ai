"""
COCO dataset loading statistics.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class COCOStatistics:
    """
    Statistics collected while loading
    COCO annotations.
    """

    folders_processed: int = 0

    images_indexed: int = 0

    annotations_loaded: int = 0

    missing_annotations: int = 0

    warnings: int = 0

    errors: int = 0

    @property
    def success_rate(self) -> float:
        """
        Percentage of images successfully annotated.
        """

        if self.images_indexed == 0:
            return 0.0

        return (
            self.annotations_loaded
            / self.images_indexed
            * 100.0
        )

    @property
    def has_errors(self) -> bool:
        """
        Returns True if errors occurred.
        """

        return self.errors > 0

    @property
    def has_warnings(self) -> bool:
        """
        Returns True if warnings occurred.
        """

        return self.warnings > 0
    
    def reset(self) -> None:
        """
        Reset all statistics.
        """

        self.folders_processed = 0
        self.images_indexed = 0
        self.annotations_loaded = 0
        self.missing_annotations = 0
        self.warnings = 0
        self.errors = 0

    def __str__(self) -> str:

        return (
            "\n"
            "COCO Statistics\n"
            f"Folders processed : {self.folders_processed}\n"
            f"Images indexed    : {self.images_indexed}\n"
            f"Annotations loaded: {self.annotations_loaded}\n"
            f"Missing           : {self.missing_annotations}\n"
            f"Warnings          : {self.warnings}\n"
            f"Errors            : {self.errors}\n"
            f"Success rate      : {self.success_rate:.2f}%"
        )
    
    def merge(
        self,
        other: "COCOStatistics",
    ) -> None:
        """
        Merge statistics from another instance.
        """

        self.folders_processed += other.folders_processed
        self.images_indexed += other.images_indexed
        self.annotations_loaded += other.annotations_loaded
        self.missing_annotations += other.missing_annotations
        self.warnings += other.warnings
        self.errors += other.errors