import math
from .. import Stage
from src.models import Cells


class Normalization(Stage):
    def __init__(self, block_size: int, flatten: bool = True):
        self.block_size = block_size
        self.flatten = flatten

    def apply(self, cells: Cells) -> list[float]:
        if self.flatten:
            return self.apply_with_flatten(cells)
        
        return self.apply_without_flatten(cells)

    def apply_with_flatten(self, cells: Cells) -> list[float]:
        blocks_per_column = cells.rows - self.block_size + 1
        blocks_per_row = cells.columns - self.block_size + 1
        vector = []

        for i in range(blocks_per_column):
            for j in range(blocks_per_row):
                    vector.extend(self.normalize_histograms_per_block(cells, i, j))

        return vector

    def apply_without_flatten(self, cells: Cells) -> list[float]:
        blocks_per_column = cells.rows - self.block_size + 1
        blocks_per_row = cells.columns - self.block_size + 1
        vector = []

        for i in range(blocks_per_column):
            for j in range(blocks_per_row):
                    vector.append(self.normalize_histograms_per_block(cells, i, j))

        return vector

    def normalize_histograms_per_block(
        self, cells: Cells, block_row: int, block_column: int
    ) -> list[float]:
        descriptor = []

        for i in range(self.block_size):
            for j in range(self.block_size):
                row = i + block_row
                column = j + block_column
                histogram_index = row * cells.columns + column

                descriptor.extend(cells.histograms[histogram_index])

        norm = 1e-6
        for i in range(len(descriptor)):
            norm += descriptor[i] ** 2
        norm = math.sqrt(norm)

        return [descriptor[i] / norm for i in range(len(descriptor))]
