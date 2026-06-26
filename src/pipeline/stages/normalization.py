import math
import numpy as np
from numba import njit, prange
from .. import Stage
from src.models import Cells, NumpyCells


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


class VectorizedNormalization(Stage):
    def __init__(self, block_size: int, flatten: bool = True):
        self.block_size = block_size
        self.flatten = flatten

    def apply(self, cells: NumpyCells) -> np.ndarray:
        blocks_per_column = cells.rows - self.block_size + 1
        blocks_per_row = cells.columns - self.block_size + 1

        histogram_size = cells.histograms.shape[-1]
        block_features = self.block_size * self.block_size * histogram_size

        descriptors = (
            np.lib.stride_tricks.sliding_window_view(
                cells.histograms,
                window_shape=(self.block_size, self.block_size),
                axis=(0, 1),
            )
            .transpose(0, 1, 3, 4, 2)
            .reshape(blocks_per_column * blocks_per_row, block_features)
        )

        descriptors = descriptors / (
            np.linalg.norm(descriptors, axis=-1, keepdims=True) + 1e-6
        )

        if self.flatten:
            return descriptors.ravel()
        return descriptors.reshape(blocks_per_column, blocks_per_row, block_features)


class NumbaNormalization(Stage):
    def __init__(self, block_size: int, flatten: bool = True):
        self.block_size = block_size
        self.flatten = flatten

    def apply(self, cells: NumpyCells) -> np.ndarray:
        descriptors = _normalize(
            cells.histograms, cells.rows, cells.columns, self.block_size
        )
        return descriptors.ravel() if self.flatten else descriptors


@njit(cache=True, parallel=True)
def _normalize(histograms: np.ndarray, rows: int, columns: int, block_size: int):
    blocks_per_column = rows - block_size + 1
    blocks_per_row = columns - block_size + 1

    histogram_size = histograms.shape[-1]
    block_features = block_size * block_size * histogram_size

    descriptors = np.empty(
        (blocks_per_column, blocks_per_row, block_features), dtype=np.float32
    )

    for i in prange(blocks_per_column):
        for j in range(blocks_per_row):
            for bi in range(block_size):
                for bj in range(block_size):
                    row = i + bi
                    column = j + bj
                    descriptor_idx = (bi * block_size + bj) * histogram_size

                    for h in range(histogram_size):
                        flat_idx = descriptor_idx + h
                        descriptors[i, j, flat_idx] = histograms[row, column, h]

            descriptors[i, j] = descriptors[i, j] / (
                np.linalg.norm(descriptors[i, j]) + 1e-6
            )

    return descriptors
