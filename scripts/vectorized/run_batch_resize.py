from pathlib import Path
from src.models import NumpyImage
from src.pipeline import Pipeline
from src.pipeline.stages import (
    VectorizedResizing,
    VectorizedGrayscale,
    VectorizedSobelGradients,
    VectorizedHistograms,
    VectorizedNormalization,
)

if __name__ == "__main__":
    directory = Path("./data/caltech-101")

    pipeline = Pipeline(
        [
            VectorizedResizing(64, 128),
            VectorizedGrayscale(),
            VectorizedSobelGradients(),
            VectorizedHistograms(cell_size=8, bins_per_cell=9),
            VectorizedNormalization(block_size=2),
        ]
    )

    for file in directory.rglob("*"):
        if file.is_file():
            img = NumpyImage.from_file(file)
            pipeline.run(img)
