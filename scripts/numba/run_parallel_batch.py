from numba import set_num_threads
import sys
from itertools import batched
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from src.pipeline import ParallelPipeline
from src.pipeline.stages import (
    VectorizedLoadImage,
    NumbaGrayscale,
    NumbaSobelGradients,
    NumbaHistograms,
    NumbaNormalization,
)

pipeline = None


def init_pipeline():
    set_num_threads(2)

    global pipeline
    pipeline = ParallelPipeline(
        [
            VectorizedLoadImage(),
            NumbaGrayscale(),
            NumbaSobelGradients(),
            NumbaHistograms(cell_size=8, bins_per_cell=9),
            NumbaNormalization(block_size=2),
        ]
    )


def process_images(files: list[str]):
    wait([pipeline.submit(file) for file in files])


if __name__ == "__main__":
    image_count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    directory = Path("./data/caltech-101")

    files = [file for file in directory.rglob("*") if file.is_file()][:image_count]
    with ProcessPoolExecutor(max_workers=4, initializer=init_pipeline) as executor:
        executor.map(process_images, batched(files, 100))
