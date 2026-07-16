"""
Dataset report generation.

Exports dataset statistics to multiple formats.
"""

from __future__ import annotations

import json

import pandas as pd

from src.core.context import ProjectContext

from src.analysis.statistics import (
    DatasetStatisticsResult,
)


class DatasetReport:
    """
    Generates dataset reports.
    """

    # =====================================================
    # Constructor
    # =====================================================

    def __init__(
        self,
        context: ProjectContext,
        statistics: DatasetStatisticsResult,
    ) -> None:

        self.context = context

        self.statistics = statistics

        self.report_folder = (
            context.config.reports_folder
        )

        self.csv_folder = (
            context.config.csv_folder
        )

        self.data_folder = (
            context.config.data_folder
        )

    # =====================================================
    # Public API
    # =====================================================

    def generate_all(self) -> None:
        """
        Generate every report.
        """

        self.context.logger.section(
            "Generating Reports"
        )

        self._generate_txt()

        self._generate_csv()

        self._generate_json()

        self.context.logger.info(
            "Reports generated successfully."
        )

    # =====================================================
    # TXT Report
    # =====================================================

    def _generate_txt(self) -> None:
        """
        Generate a human-readable text report.
        """

        report_path = (
            self.report_folder /
            "dataset_report.txt"
        )

        with open(report_path, "w", encoding="utf-8") as file:

            file.write(
                "DATASET AUDIT REPORT\n"
            )

            file.write(
                "=" * 60 + "\n\n"
            )

            # -------------------------------------------------
            # General
            # -------------------------------------------------

            file.write("GENERAL\n")
            file.write("-" * 60 + "\n")

            file.write(
                f"Total Images : {self.statistics.image_count}\n"
            )

            file.write(
                f"Unique Animals : {self.statistics.animal_count}\n"
            )

            file.write(
                f"Folders : {self.statistics.folder_count}\n\n"
            )

            # -------------------------------------------------
            # Dataset Distribution
            # -------------------------------------------------

            file.write("DATASETS\n")
            file.write("-" * 60 + "\n")

            for dataset, count in self.statistics.dataset_distribution.items():

                file.write(
                    f"{dataset:<10}: {count}\n"
                )

            file.write("\n")

            # -------------------------------------------------
            # View Distribution
            # -------------------------------------------------

            file.write("VIEWS\n")
            file.write("-" * 60 + "\n")

            for view, count in self.statistics.view_distribution.items():

                file.write(
                    f"{view:<10}: {count}\n"
                )

            file.write("\n")

            # -------------------------------------------------
            # Sex Distribution
            # -------------------------------------------------

            file.write("SEX\n")
            file.write("-" * 60 + "\n")

            for sex, count in self.statistics.sex_distribution.items():

                file.write(
                    f"{sex:<10}: {count}\n"
                )

            file.write("\n")

            # -------------------------------------------------
            # Weight Statistics
            # -------------------------------------------------

            weight = self.statistics.weight

            file.write("WEIGHT STATISTICS\n")
            file.write("-" * 60 + "\n")

            file.write(
                f"Minimum Weight : {weight.minimum:.2f} kg\n"
            )

            file.write(
                f"Maximum Weight : {weight.maximum:.2f} kg\n"
            )

            file.write(
                f"Mean Weight    : {weight.mean:.2f} kg\n"
            )

            file.write(
                f"Median Weight  : {weight.median:.2f} kg\n"
            )

            file.write(
                f"Std. Deviation : {weight.standard_deviation:.2f} kg\n"
            )

            file.write(
                f"Variance       : {weight.variance:.2f}\n"
            )

            file.write(
                f"Q1             : {weight.q1:.2f}\n"
            )

            file.write(
                f"Q3             : {weight.q3:.2f}\n"
            )

            file.write(
                f"IQR            : {weight.iqr:.2f}\n"
            )

    # =====================================================
    # CSV Report
    # =====================================================

    def _generate_csv(self) -> None:
        """
        Export dataset summaries to CSV.
        """

        rows = []

        for group in self.statistics.datasets:

            rows.append({

                "Category": "Dataset",

                "Name": group.name,

                "Images": group.image_count,

                "Animals": group.animal_count,

                "Minimum Weight": group.minimum_weight,

                "Maximum Weight": group.maximum_weight,

                "Mean Weight": group.mean_weight,

                "Median Weight": group.median_weight,

                "Std. Deviation": group.standard_deviation

            })

        for group in self.statistics.views:

            rows.append({

                "Category": "View",

                "Name": group.name,

                "Images": group.image_count,

                "Animals": group.animal_count,

                "Minimum Weight": group.minimum_weight,

                "Maximum Weight": group.maximum_weight,

                "Mean Weight": group.mean_weight,

                "Median Weight": group.median_weight,

                "Std. Deviation": group.standard_deviation

            })

        for group in self.statistics.sexes:

            rows.append({

                "Category": "Sex",

                "Name": group.name,

                "Images": group.image_count,

                "Animals": group.animal_count,

                "Minimum Weight": group.minimum_weight,

                "Maximum Weight": group.maximum_weight,

                "Mean Weight": group.mean_weight,

                "Median Weight": group.median_weight,

                "Std. Deviation": group.standard_deviation

            })

        dataframe = pd.DataFrame(rows)

        dataframe.to_csv(

            self.csv_folder /
            "dataset_summary.csv",

            index=False

        )


    # =====================================================
    # JSON Report
    # =====================================================

    def _generate_json(self) -> None:
        """
        Export statistics to JSON.
        """

        json_path = (
            self.data_folder /
            "dataset_statistics.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(

                self.statistics.to_dict(),

                file,

                indent=4

            )

    