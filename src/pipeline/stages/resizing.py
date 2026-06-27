import math
from numba import njit, prange
import numpy as np
from typing import Tuple
from .. import Stage
from src.models import Image, ImageMode, NumpyImage


class Resizing(Stage):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply(self, img: Image) -> Image:
        if img.mode == ImageMode.GRAY:
            resized_image_pixels = self._resize_grayscale(img)
        else:
            resized_image_pixels = self._resize_rgb(img)

        return Image(img.mode, self.height, self.width, resized_image_pixels)

    def _resize_grayscale(self, img: Image) -> list[float]:
        resized_image_pixels = []
        height_scale_ratio = img.height / self.height
        width_scale_ratio = img.width / self.width

        for i in range(self.height):
            for j in range(self.width):
                x_min = j * width_scale_ratio
                x_max = (j + 1) * width_scale_ratio
                y_min = i * height_scale_ratio
                y_max = (i + 1) * height_scale_ratio

                value, total_area = 0.0, 0.0
                for si in range(
                    max(0, math.floor(y_min)), min(img.height, math.ceil(y_max))
                ):
                    h = max(0, min(y_max, si + 1) - max(y_min, si))

                    for sj in range(
                        max(0, math.floor(x_min)), min(img.width, math.ceil(x_max))
                    ):
                        w = max(0, min(x_max, sj + 1) - max(x_min, sj))
                        area = w * h

                        value += img.pixels[si * img.width + sj] * area
                        total_area += area
                value /= total_area
                resized_image_pixels.append(value)

        return resized_image_pixels

    def _resize_rgb(self, img: Image) -> list[Tuple[float, float, float]]:
        resized_image_pixels = []
        height_scale_ratio = img.height / self.height
        width_scale_ratio = img.width / self.width

        for i in range(self.height):
            for j in range(self.width):
                x_min = j * width_scale_ratio
                x_max = (j + 1) * width_scale_ratio
                y_min = i * height_scale_ratio
                y_max = (i + 1) * height_scale_ratio

                value, total_area = [0.0, 0.0, 0.0], 0.0
                for si in range(
                    max(0, math.floor(y_min)), min(img.height, math.ceil(y_max))
                ):
                    h = max(0, min(y_max, si + 1) - max(y_min, si))

                    for sj in range(
                        max(0, math.floor(x_min)), min(img.width, math.ceil(x_max))
                    ):
                        w = max(0, min(x_max, sj + 1) - max(x_min, sj))
                        area = w * h

                        for c in range(3):
                            value[c] += img.pixels[si * img.width + sj][c] * area
                        total_area += area
                resized_image_pixels.append(tuple((p / total_area for p in value)))

        return resized_image_pixels


class VectorizedResizing(Stage):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply(self, img: NumpyImage) -> NumpyImage:
        if img.mode == ImageMode.GRAY:
            resized_image_pixels = self._resize_grayscale(img)
        else:
            resized_image_pixels = self._resize_rgb(img)

        return NumpyImage(img.mode, self.height, self.width, resized_image_pixels)

    def _resize_grayscale(self, img: NumpyImage) -> np.ndarray:
        height_scale_ratio = img.height / self.height
        width_scale_ratio = img.width / self.width

        area_widths = self._get_area_dimensions(
            self.width, img.width, width_scale_ratio
        )
        area_heights = self._get_area_dimensions(
            self.height, img.height, height_scale_ratio
        )

        values = area_heights @ img.pixels @ area_widths.T
        total_area = np.outer(area_heights.sum(axis=1), area_widths.sum(axis=1))

        return values / total_area

    def _resize_rgb(self, img: NumpyImage) -> np.ndarray:
        height_scale_ratio = img.height / self.height
        width_scale_ratio = img.width / self.width

        area_widths = self._get_area_dimensions(
            self.width, img.width, width_scale_ratio
        )
        area_heights = self._get_area_dimensions(
            self.height, img.height, height_scale_ratio
        )

        values = area_heights @ img.pixels.transpose(2, 0, 1) @ area_widths.T
        total_area = np.outer(area_heights.sum(axis=1), area_widths.sum(axis=1))

        return (values / total_area).transpose(1, 2, 0)

    def _get_area_dimensions(self, target_size, image_size, scale_ratio):
        area_indices = np.arange(target_size, dtype=np.float32)
        mins = area_indices * scale_ratio
        maxs = (area_indices + 1) * scale_ratio
        original_indices = np.arange(image_size, dtype=np.float32)

        low = np.maximum(mins[:, None], original_indices[None, :])
        high = np.minimum(maxs[:, None], original_indices[None, :] + 1)
        return np.clip(high - low, 0, None)


class NumbaResizing(Stage):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply(self, img: NumpyImage) -> NumpyImage:
        if img.mode == ImageMode.GRAY:
            resized_image_pixels = _resize_grayscale(
                img.pixels, img.height, img.width, self.height, self.width
            )
        else:
            resized_image_pixels = _resize_rgb(
                img.pixels, img.height, img.width, self.height, self.width
            )

        return NumpyImage(img.mode, self.height, self.width, resized_image_pixels)


@njit(cache=True, parallel=True)
def _resize_grayscale(
    pixels: np.ndarray, height: int, width: int, target_height: int, target_width: int
):
    height_scale_ratio = height / target_height
    width_scale_ratio = width / target_width
    resized_image_pixels = np.zeros((target_height, target_width), dtype=np.float32)

    for i in prange(target_height):
        for j in range(target_width):
            x_min = j * width_scale_ratio
            x_max = (j + 1) * width_scale_ratio
            y_min = i * height_scale_ratio
            y_max = (i + 1) * height_scale_ratio

            total_area = 0.0
            for si in range(
                max(0, math.floor(y_min)), min(height, math.ceil(y_max))
            ):
                h = max(0, min(y_max, si + 1) - max(y_min, si))

                for sj in range(
                    max(0, math.floor(x_min)), min(width, math.ceil(x_max))
                ):
                    w = max(0, min(x_max, sj + 1) - max(x_min, sj))
                    area = w * h

                    resized_image_pixels[i, j] += pixels[si, sj] * area
                    total_area += area
            resized_image_pixels[i, j] /= total_area

    return resized_image_pixels


@njit(cache=True, parallel=True)
def _resize_rgb(
    pixels: np.ndarray, height: int, width: int, target_height: int, target_width: int
):
    height_scale_ratio = height / target_height
    width_scale_ratio = width / target_width
    resized_image_pixels = np.zeros((target_height, target_width, 3), dtype=np.float32)

    for i in prange(target_height):
        for j in range(target_width):
            x_min = j * width_scale_ratio
            x_max = (j + 1) * width_scale_ratio
            y_min = i * height_scale_ratio
            y_max = (i + 1) * height_scale_ratio

            total_area = 0.0
            for si in range(
                max(0, math.floor(y_min)), min(height, math.ceil(y_max))
            ):
                h = max(0, min(y_max, si + 1) - max(y_min, si))

                for sj in range(
                    max(0, math.floor(x_min)), min(width, math.ceil(x_max))
                ):
                    w = max(0, min(x_max, sj + 1) - max(x_min, sj))
                    area = w * h

                    resized_image_pixels[i, j] += pixels[si, sj] * area
                    total_area += area
            resized_image_pixels[i, j] /= total_area

    return resized_image_pixels