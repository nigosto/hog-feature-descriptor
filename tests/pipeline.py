import unittest
from src.models import Image, ImageMode
from src.pipeline import Pipeline
from src.pipeline.stages import Grayscale, SobelGradients, Histograms, Normalization


class TestPipeline(unittest.TestCase):
    def test_5x5_grayscale_image(self):
        img = Image(
            ImageMode.GRAY,
            5,
            5,
            [
                10,
                10,
                10,
                10,
                10,
                10,
                20,
                30,
                20,
                10,
                10,
                40,
                50,
                40,
                10,
                10,
                20,
                30,
                20,
                10,
                10,
                10,
                10,
                10,
                10,
            ],
        )
        pipeline = Pipeline(
            [
                Grayscale(),
                SobelGradients(),
                Histograms(cell_size=2, bins_per_cell=9),
                Normalization(block_size=2),
            ]
        )

        blocks = pipeline.run(img)
        vector = [
            feature for descriptor in blocks.descriptors for feature in descriptor
        ]
        expected = [
            0.0,
            0.0,
            0.35036,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.38302,
            0.0,
            0.35036,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.35036,
            0.0,
            0.32830,
            0.32830,
            0.0,
            0.35036,
            0.0,
            0.38302,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for actual, exp in zip(vector, expected):
            self.assertAlmostEqual(actual, exp, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
