import unittest
from unittest.mock import MagicMock, patch

from src.core.context import ProjectContext
from src.training.data_loader import load_training_samples
from src.training.samples import TrainingSample


class TestLoadTrainingSamples(unittest.TestCase):
    @patch("src.training.data_loader.DatasetBuilder")
    @patch("src.training.data_loader.COCOReader")
    @patch("src.training.data_loader.DatasetReader")
    def test_load_training_samples_runs_pipeline(
        self,
        mock_dataset_reader_class,
        mock_coco_reader_class,
        mock_builder_class,
    ):
        context = MagicMock(spec=ProjectContext)

        dataset = MagicMock()
        expected_samples = [
            MagicMock(spec=TrainingSample),
            MagicMock(spec=TrainingSample),
        ]

        mock_dataset_reader = mock_dataset_reader_class.return_value
        mock_dataset_reader.load.return_value = dataset

        mock_coco_reader = mock_coco_reader_class.return_value

        mock_builder = mock_builder_class.return_value
        mock_builder.build.return_value = expected_samples

        result = load_training_samples(context)

        mock_dataset_reader_class.assert_called_once_with(context)
        mock_dataset_reader.load.assert_called_once_with()

        mock_coco_reader_class.assert_called_once_with(context)
        mock_coco_reader.enrich.assert_called_once_with(dataset)

        mock_builder_class.assert_called_once_with(
            require_both_views=True,
        )
        mock_builder.build.assert_called_once_with(
            dataset.records,
        )

        self.assertIs(result, expected_samples)


if __name__ == "__main__":
    unittest.main()