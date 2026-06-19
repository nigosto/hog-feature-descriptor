import numpy as np
from typing import Tuple
from .. import Stage
from src.models import Image, ImageMode, NumpyImage


class Grayscale(Stage):
    W_R, W_G, W_B = 0.299, 0.587, 0.114

    def apply(self, img: Image) -> Image:
        if img.mode == ImageMode.GRAY:
            gray_pixels = [pixel / 255.0 for pixel in img.pixels]
        else:
            gray_pixels = [self._to_grayscale(pixel) for pixel in img.pixels]

        return Image(ImageMode.GRAY, img.height, img.width, gray_pixels)

    def _to_grayscale(self, pixel: Tuple[float, float, float]) -> float:
        return (self.W_R * pixel[0] + self.W_G * pixel[1] + self.W_B * pixel[2]) / 255.0


class VectorizedGrayscale(Stage):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def apply(self, img: NumpyImage) -> NumpyImage:
        if img.mode == ImageMode.GRAY:
            gray_pixels = img.pixels / 255.0
        else:
            gray_pixels = img.pixels @ self.weights / 255.0

        return Image(ImageMode.GRAY, img.height, img.width, gray_pixels)
