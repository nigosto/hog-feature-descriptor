import sys
from src.models import NumpyImage
from src.pipeline.stages import (
    NumbaGrayscale,
    NumbaSobelGradients,
    NumbaHistograms,
    NumbaNormalization
)
from src.pipeline import Pipeline

if __name__ == "__main__":
    img = NumpyImage.from_file(sys.argv[1])
    pipeline = Pipeline(
        [
            NumbaGrayscale(),
            NumbaSobelGradients(),
            NumbaHistograms(cell_size=8, bins_per_cell=9),
            NumbaNormalization(block_size=2),
        ]
    )
    pipeline.run(img)
