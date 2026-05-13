from pathlib import Path
from src.models import Image
from src.pipeline import Pipeline
from src.pipeline.stages import Resizing, Grayscale, SobelGradients, Histograms, Normalization

if __name__ == "__main__":
    directory = Path("./data/caltech-101")

    pipeline = Pipeline(
        [
            Resizing(64, 128),
            Grayscale(),
            SobelGradients(),
            Histograms(cell_size=8, bins_per_cell=9),
            Normalization(block_size=2)   
        ]
    )

    for file in directory.rglob("*"):
        if file.is_file():
            img = Image.from_file(file)
            pipeline.run(img)