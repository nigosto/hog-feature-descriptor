import sys
import math
import matplotlib.pyplot as plt
from src.models import NumpyImage
from src.pipeline import Pipeline
from src.pipeline.stages import VectorizedGrayscale, VectorizedSobelGradients, VectorizedHistograms, VectorizedNormalization

if __name__ == "__main__":
    img = NumpyImage.from_file(sys.argv[1])

    cell_size = 4
    bins_per_cell = 9

    pipeline = Pipeline(
        [
            VectorizedGrayscale(),
            VectorizedSobelGradients(),
            VectorizedHistograms(cell_size, bins_per_cell),
            VectorizedNormalization(block_size=1, flatten=False)   
        ]
    )
    cells = pipeline.run(img)

    blocks_per_column, blocks_per_row, _ = cells.shape

    for y in range(blocks_per_column):
        for x in range(blocks_per_row):
            for b in range(bins_per_cell):
                strength = cells[y, x, b]
                
                angle = math.radians(b * (180 / bins_per_cell))

                cy = y * cell_size + cell_size // 2
                cx = x * cell_size + cell_size // 2

                dx = math.cos(angle) * strength * cell_size / 2
                dy = math.sin(angle) * strength * cell_size / 2

                plt.plot([cx - dx, cx + dx],
                         [cy - dy, cy + dy],
                         'r')

    plt.gca().invert_yaxis()
    plt.title("HOG (Numpy vectorized, Sobel-based)")
    plt.axis('off')
    plt.show()
