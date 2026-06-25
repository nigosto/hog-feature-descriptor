import sys
from mpi4py.futures import MPIPoolExecutor
from itertools import batched
from concurrent.futures import wait
from pathlib import Path
from src.pipeline import ParallelPipeline
from src.pipeline.stages import (
    VectorizedLoadImage,
    VectorizedGrayscale,
    VectorizedSobelGradients,
    VectorizedHistograms,
    VectorizedNormalization,
)

pipeline = None


def init_pipeline():
    global pipeline
    pipeline = ParallelPipeline(
        [
            VectorizedLoadImage(),
            VectorizedGrayscale(),
            VectorizedSobelGradients(),
            VectorizedHistograms(cell_size=8, bins_per_cell=9),
            VectorizedNormalization(block_size=2),
        ]
    )


def process_images(files: list[str]):
    wait([pipeline.submit(file) for file in files])


if __name__ == "__main__":
    image_count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    directory = Path("./data/caltech-256")

    files = [file for file in directory.rglob("*") if file.is_file()][:image_count]
    with MPIPoolExecutor(initializer=init_pipeline) as executor:
        executor.map(process_images, batched(files, 50))
