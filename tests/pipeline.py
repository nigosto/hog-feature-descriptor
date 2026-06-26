import unittest
import numpy as np
from src.models import Image, ImageMode, NumpyImage
from src.pipeline import Pipeline
from src.pipeline.stages import (
    Grayscale,
    VectorizedGrayscale,
    NumbaGrayscale,
    SobelGradients,
    VectorizedSobelGradients,
    NumbaSobelGradients,
    Histograms,
    VectorizedHistograms,
    NumbaHistograms,
    Normalization,
    VectorizedNormalization,
    NumbaNormalization,
)

pixels = [
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
    0.32831,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.35036,
    0.0,
    0.0,
    0.32831,
    0.0,
    0.35036,
    0.0,
    0.38302,
    0.0,
    0.0,
    0.0,
    0.0,
]


class TestPipeline(unittest.TestCase):
    def test_5x5_grayscale_image(self):
        img = Image(ImageMode.GRAY, 5, 5, pixels)
        pipeline = Pipeline(
            [
                Grayscale(),
                SobelGradients(),
                Histograms(cell_size=2, bins_per_cell=9),
                Normalization(block_size=2),
            ]
        )

        vector = pipeline.run(img)

        for actual, exp in zip(vector, expected):
            self.assertAlmostEqual(actual, exp, delta=1e-5)

    def test_vectorized_5x5_grayscale_image(self):
        img = NumpyImage(ImageMode.GRAY, 5, 5, np.array(pixels).reshape(5, 5))
        pipeline = Pipeline(
            [
                VectorizedGrayscale(),
                VectorizedSobelGradients(),
                VectorizedHistograms(cell_size=2, bins_per_cell=9),
                VectorizedNormalization(block_size=2),
            ]
        )

        vector = pipeline.run(img)

        for actual, exp in zip(vector, expected):
            self.assertAlmostEqual(actual, exp, delta=1e-5)

    def test_numba_5x5_grayscale_image(self):
        img = NumpyImage(ImageMode.GRAY, 5, 5, np.array(pixels).reshape(5, 5))
        pipeline = Pipeline(
            [
                NumbaGrayscale(),
                NumbaSobelGradients(),
                NumbaHistograms(cell_size=2, bins_per_cell=9),
                NumbaNormalization(block_size=2),
            ]
        )

        vector = pipeline.run(img)

        for actual, exp in zip(vector, expected):
            self.assertAlmostEqual(actual, exp, delta=1e-5)

if __name__ == "__main__":
    unittest.main()
