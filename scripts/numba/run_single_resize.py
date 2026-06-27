import sys
from src.models import NumpyImage
from src.pipeline import Pipeline
from src.pipeline.stages import (
    NumbaResizing,
    NumbaGrayscale,
    NumbaSobelGradients,
    NumbaHistograms,
    NumbaNormalization,
)

if __name__ == "__main__":
    img = NumpyImage.from_file(sys.argv[1])
    pipeline = Pipeline(
        [
            NumbaResizing(64, 128),
            NumbaGrayscale(),
            NumbaSobelGradients(),
            NumbaHistograms(cell_size=8, bins_per_cell=9),
            NumbaNormalization(block_size=2),
        ]
    )
    pipeline.run(img)
