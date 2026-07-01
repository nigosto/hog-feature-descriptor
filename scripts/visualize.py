import sys
import math
import matplotlib.pyplot as plt
from src.models import Image
from src.pipeline import Pipeline
from src.pipeline.stages import Grayscale, SobelGradients, Histograms, Normalization

if __name__ == "__main__":
    img = Image.from_file(sys.argv[1])

    cell_size = 8
    bins_per_cell = 9

    pipeline = Pipeline(
        [
            Grayscale(),
            SobelGradients(),
            Histograms(cell_size, bins_per_cell),
            Normalization(block_size=2, flatten=False),
        ]
    )
    cells = pipeline.run(img)

    rows = img.height // cell_size
    columns = img.width // cell_size

    for y in range(rows):
        for x in range(columns):
            for b in range(bins_per_cell):
                strength = cells[y * columns + x][b]
                angle = math.radians(b * (180 / bins_per_cell))

                cy = y * cell_size + cell_size // 2
                cx = x * cell_size + cell_size // 2

                dx = math.cos(angle) * strength * cell_size / 2
                dy = math.sin(angle) * strength * cell_size / 2

                plt.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "r")

    plt.gca().invert_yaxis()
    plt.title("HOG (Pure Serial, Sobel-based)")
    plt.axis("off")
    plt.show()
