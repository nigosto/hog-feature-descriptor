import sys
from pathlib import Path
from src.models import NumpyImage
from src.pipeline import Pipeline
from src.pipeline.stages import (
    VectorizedGrayscale,
    VectorizedSobelGradients,
    VectorizedHistograms,
    VectorizedNormalization,
)

if __name__ == "__main__":
    image_count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    directory = Path("./data/caltech-101")

    pipeline = Pipeline(
        [
            VectorizedGrayscale(),
            VectorizedSobelGradients(),
            VectorizedHistograms(cell_size=8, bins_per_cell=9),
            VectorizedNormalization(block_size=2),
        ]
    )

    files = [file for file in directory.rglob("*") if file.is_file()][:image_count]

    for file in files:
        img = NumpyImage.from_file(file)
        pipeline.run(img)
