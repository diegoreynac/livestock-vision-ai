"""
Main entry point.

Livestock Weight Estimation Dataset Audit
"""

from src.core.context import ProjectContext
from src.core.config import ProjectConfig

from src.dataset.reader import DatasetReader

from src.analysis.statistics import DatasetStatistics
from src.analysis.visualization import DatasetVisualizer
from src.analysis.report import DatasetReport
from time import perf_counter




def main() -> None:
    """
    Execute the complete dataset audit pipeline.
    """
    
    start_time = perf_counter()
    
    # =====================================================
    # Project Context
    # =====================================================

    config = ProjectConfig.default()
    context = ProjectContext(
    config=config,
    experiment_name="dataset_audit",
)

    context.config.validate()

    context.config.create_output_directories()

    context.logger.section(
        "Livestock Weight Estimation"
    )

    # =====================================================
    # Read Dataset
    # =====================================================

    reader = DatasetReader(context)

    dataset = reader.load()

    # =====================================================
    # Compute Statistics
    # =====================================================

    statistics = DatasetStatistics(dataset)

    results = statistics.compute()

    # =====================================================
    # Generate Figures
    # =====================================================

    visualizer = DatasetVisualizer(

        context,

        results

    )

    visualizer.generate_all()

    # =====================================================
    # Generate Reports
    # =====================================================

    report = DatasetReport(

        context,

        results

    )

    report.generate_all()

    # =====================================================
    # Finish
    # =====================================================

    context.logger.section(
        "Dataset Audit Completed"
    )

    elapsed = perf_counter() - start_time

    context.logger.info(
        f"Execution Time : {elapsed:.2f} seconds"
    )

    context.logger.section(
        "Summary"
    )

    context.logger.info(
        f"Images              : {results.image_count}"
    )

    context.logger.info(
        f"Animals             : {results.animal_count}"
    )

    context.logger.info(
        f"Figures Generated   : 7"
    )

    context.logger.info(
        f"Reports Generated   : 3"
    )


if __name__ == "__main__":

    main()