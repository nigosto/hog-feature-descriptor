import numpy as np
from .. import Stage
from src.models import Gradients, NumpyGradients, Cells, NumpyCells


class BaseHistograms(Stage):
    def __init__(self, cell_size: int, bins_per_cell: int):
        self.cell_size = cell_size
        self.bins_per_cell = bins_per_cell
        self.bin_width = 180 / bins_per_cell


class Histograms(BaseHistograms):
    def __init__(self, cell_size: int, bins_per_cell: int):
        super().__init__(cell_size, bins_per_cell)

    def apply(self, gradients: Gradients) -> Cells:
        cells_per_column = gradients.height // self.cell_size
        cells_per_row = gradients.width // self.cell_size
        histograms = []

        for i in range(cells_per_column):
            for j in range(cells_per_row):
                histograms.append(self.compute_cell_histogram(gradients, i, j))

        return Cells(cells_per_column, cells_per_row, self.cell_size, histograms)

    def compute_cell_histogram(
        self, gradients: Gradients, cell_row: int, cell_column: int
    ) -> list[float]:
        histogram = [0.0] * self.bins_per_cell

        for i in range(self.cell_size):
            for j in range(self.cell_size):
                row = i + cell_row * self.cell_size
                column = j + cell_column * self.cell_size
                pixel_index = row * gradients.width + column
                angle = gradients.orientations[pixel_index] % 180
                bin_index = int(angle / self.bin_width)

                histogram[bin_index] += gradients.magnitudes[pixel_index]

        return histogram


class VectorizedHistograms(BaseHistograms):
    def __init__(self, cell_size: int, bins_per_cell: int):
        super().__init__(cell_size, bins_per_cell)

    def apply(self, gradients: NumpyGradients) -> NumpyCells:
        cells_per_column = gradients.height // self.cell_size
        cells_per_row = gradients.width // self.cell_size
        num_cells = cells_per_column * cells_per_row

        rows = cells_per_column * self.cell_size
        columns = cells_per_row * self.cell_size
        magnitudes = (
            gradients.magnitudes[:rows, :columns]
            .reshape(cells_per_column, self.cell_size, cells_per_row, self.cell_size)
            .transpose(0, 2, 1, 3)
            .reshape(num_cells, -1)
        )
        orientations = (
            gradients.orientations[:rows, :columns]
            .reshape(cells_per_column, self.cell_size, cells_per_row, self.cell_size)
            .transpose(0, 2, 1, 3)
            .reshape(num_cells, -1)
        )

        orientations = np.minimum(
            (orientations % 180) // self.bin_width, self.bins_per_cell - 1
        ).astype(np.intp)

        orientations = np.arange(num_cells)[:, None] * self.bins_per_cell + orientations

        histograms = np.bincount(
            orientations.ravel(),
            weights=magnitudes.ravel(),
            minlength=(num_cells * self.bins_per_cell),
        ).reshape(cells_per_column, cells_per_row, self.bins_per_cell)

        return NumpyCells(cells_per_column, cells_per_row, self.cell_size, histograms)
