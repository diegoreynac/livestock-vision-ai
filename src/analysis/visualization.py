"""
Dataset visualization.

Generates publication-quality figures for the livestock dataset.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

from src.core.context import ProjectContext

from src.analysis.statistics import (
    DatasetStatisticsResult,
)

class DatasetVisualizer:
    """
    Generates all figures used in the dataset audit.
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

        self.output_folder = (
            context.config.figures_folder
        )

    # =====================================================
    # Public API
    # =====================================================

    def generate_all(self) -> None:
        """
        Generates every figure.
        """

        self.context.logger.section(
            "Generating Figures"
        )

        self._plot_dataset_distribution()

        self._plot_view_distribution()

        self._plot_sex_distribution()

        self._plot_weight_histogram()

        self._plot_weight_boxplot()

        self._plot_weight_by_dataset()

        self._plot_weight_by_sex()

        self.context.logger.info(
            "Figures generated successfully."
        )

        self.context.logger.info(
            f"Figures Folder : {self.output_folder}"
        )

    # =====================================================
    # Generic Plot Helpers
    # =====================================================

    def _save_current_figure(
        self,
        filename: str
    ) -> None:
        """
        Save the current matplotlib figure.
        """

        plt.tight_layout()

        plt.savefig(
            self.output_folder / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    def _plot_bar_chart(
        self,
        distribution: Counter,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str
    ) -> None:
        """
        Draw a generic bar chart from a Counter object.
        """

        labels = list(distribution.keys())
        values = list(distribution.values())

        plt.figure(figsize=(8, 6))

        plt.bar(
            labels,
            values
        )

        plt.title(title)

        plt.xlabel(xlabel)

        plt.ylabel(ylabel)

        plt.grid(
            axis="y",
            alpha=0.30
        )

        self._save_current_figure(
            filename
        )

    def _plot_boxplot(
        self,
        data: list[float],
        title: str,
        ylabel: str,
        filename: str
    ) -> None:
        """
        Draw a generic boxplot.
        """

        plt.figure(figsize=(6, 6))

        plt.boxplot(
            data,
            vert=True
        )

        plt.title(title)

        plt.ylabel(ylabel)

        plt.grid(
            axis="y",
            alpha=0.30
        )

        self._save_current_figure(
            filename
        )

    # =====================================================
    # Distribution Plots
    # =====================================================

    def _plot_dataset_distribution(self) -> None:
        """
        Plot the number of images per dataset.
        """

        self._plot_bar_chart(

            distribution=self.statistics.dataset_distribution,

            title="Dataset Distribution",

            xlabel="Dataset",

            ylabel="Number of Images",

            filename="01_dataset_distribution.png"

        )

    def _plot_view_distribution(self) -> None:
        """
        Plot the number of images per camera view.
        """

        self._plot_bar_chart(

            distribution=self.statistics.view_distribution,

            title="View Distribution",

            xlabel="View",

            ylabel="Number of Images",

            filename="02_view_distribution.png"

        )

    def _plot_sex_distribution(self) -> None:
        """
        Plot the number of animals by sex.
        """

        self._plot_bar_chart(

            distribution=self.statistics.sex_distribution,

            title="Sex Distribution",

            xlabel="Sex",

            ylabel="Number of Animals",

            filename="03_sex_distribution.png"

        )

    def _plot_weight_histogram(self) -> None:
        plt.figure(figsize=(8, 6))

        plt.hist(
            self.statistics.weights,
            bins=30,
            edgecolor="black",
        )

        plt.title("Weight Distribution")
        plt.xlabel("Weight (kg)")
        plt.ylabel("Number of Images")

        self._save_current_figure(
            "04_weight_histogram.png"
        )

    def _plot_weight_boxplot(self) -> None:
        self._plot_boxplot(
            data=self.statistics.weights,
            title="Weight Distribution",
            ylabel="Weight (kg)",
            filename="05_weight_boxplot.png",
        )

    def _plot_weight_by_dataset(self) -> None:
        distribution = Counter({
            group.name: group.mean_weight
            for group in self.statistics.datasets
        })

        self._plot_bar_chart(
            distribution=distribution,
            title="Mean Weight by Dataset",
            xlabel="Dataset",
            ylabel="Mean Weight (kg)",
            filename="06_weight_by_dataset.png",
        )

    def _plot_weight_by_sex(self) -> None:
        distribution = Counter({
            group.name: group.mean_weight
            for group in self.statistics.sexes
        })

        self._plot_bar_chart(
            distribution=distribution,
            title="Mean Weight by Sex",
            xlabel="Sex",
            ylabel="Mean Weight (kg)",
            filename="07_weight_by_sex.png",
        )
        