from .. import Stage
from src.models import Gradients, Cells


class Histograms(Stage):
    def __init__(self, cell_size: int, bins_per_cell: int) -> Cells:
        self.cell_size = cell_size
        self.bins_per_cell = bins_per_cell
        self.bin_width = 180 / bins_per_cell

    def apply(self, gradients: Gradients):
        cells_per_column = gradients.height // self.cell_size
        cells_per_row = gradients.width // self.cell_size
        histograms = []

        for i in range(cells_per_column):
            for j in range(cells_per_row):
                histograms.append(self.compute_cell_histogram(gradients, i, j))

        return Cells(
            cells_per_column, cells_per_row, self.cell_size, histograms
        )

    def compute_cell_histogram(
        self, gradients: Gradients, cell_row: int, cell_column: int
    ) -> list[float]:
        histogram = [0.0] * self.bins_per_cell

        for i in range(self.cell_size):
            for j in range(self.cell_size):
                row = i + cell_row * self.cell_size
                column = j + cell_column * self.cell_size
                pixel_index = row * gradients.width + column
                angle = gradients.orientations[pixel_index]
                bin_index = min(int(angle / self.bin_width), self.bins_per_cell - 1)

                histogram[bin_index] += gradients.magnitudes[pixel_index]

        return histogram
