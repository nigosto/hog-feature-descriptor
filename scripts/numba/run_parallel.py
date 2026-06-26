from numba import set_num_threads
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from src.models import NumpyImage
from src.pipeline import Pipeline
from src.pipeline.stages import (
    NumbaGrayscale,
    NumbaSobelGradients,
    NumbaHistograms,
    NumbaNormalization,
)

pipeline = None


def init_pipeline():
    set_num_threads(2)

    global pipeline
    pipeline = Pipeline(
        [
            NumbaGrayscale(),
            NumbaSobelGradients(),
            NumbaHistograms(cell_size=8, bins_per_cell=9),
            NumbaNormalization(block_size=2),
        ]
    )


def process_image(file: str):
    img = NumpyImage.from_file(file)
    pipeline.run(img)


if __name__ == "__main__":
    image_count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    directory = Path("./data/caltech-101")

    files = [file for file in directory.rglob("*") if file.is_file()][:image_count]
    with ProcessPoolExecutor(max_workers=4, initializer=init_pipeline) as executor:
        executor.map(process_image, files)
