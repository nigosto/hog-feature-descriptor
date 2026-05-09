import math
from .. import Stage
from src.models import Cells, Blocks


class Normalization(Stage):
    def __init__(self, block_size: int):
        self.block_size = block_size

    def apply(self, cells: Cells) -> Blocks:
        blocks_per_column = cells.rows - self.block_size + 1
        blocks_per_row = cells.columns - self.block_size + 1
        descriptor = []

        for i in range(blocks_per_column):
            for j in range(blocks_per_row):
                descriptor.append(self.normalize_histograms_per_block(cells, i, j))

        return Blocks(blocks_per_column, blocks_per_row, self.block_size, descriptor)

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
        for k in range(len(descriptor)):
            norm += descriptor[k] ** 2

        norm = math.sqrt(norm)

        for k in range(len(descriptor)):
            descriptor[k] /= norm

        return descriptor
