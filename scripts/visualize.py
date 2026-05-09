import sys
import math
import matplotlib.pyplot as plt
from src.models import Image
from src.pipeline import Pipeline
from src.pipeline.stages import Grayscale, SobelGradients, Histograms

if __name__ == "__main__":
    img = Image.from_file(sys.argv[1])

    pipeline = Pipeline(
        [
            Grayscale(),
            SobelGradients(),
            Histograms(cell_size=8, bins_per_cell=9),
        ]
    )
    cells = pipeline.run(img)

    image_height = cells.rows * cells.size
    image_width = cells.columns * cells.size

    fig, ax = plt.subplots()

    ax.imshow(
        [[0.0 for _ in range(image_width)] for _ in range(image_height)],
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    bins_per_cell = len(cells.histograms[0])
    bin_width = 180 / bins_per_cell

    for cell_row in range(cells.rows):
        for cell_column in range(cells.columns):
            histogram = cells.histograms[cell_row * cells.columns + cell_column]

            center_x = cell_column * cells.size + cells.size / 2
            center_y = cell_row * cells.size + cells.size / 2

            for bin_index in range(bins_per_cell):
                magnitude = histogram[bin_index]

                if magnitude < 1e-5:
                    continue

                angle = bin_index * bin_width + bin_width / 2

                angle_rad = math.radians(angle)

                length = magnitude * 8

                dx = math.cos(angle_rad) * length
                dy = math.sin(angle_rad) * length

                x1 = center_x - dx
                x2 = center_x + dx

                y1 = center_y - dy
                y2 = center_y + dy

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="white",
                    linewidth=1,
                )

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_title("HOG Visualization")
    plt.show()
