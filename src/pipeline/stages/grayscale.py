from numba import njit, prange
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

        return NumpyImage(ImageMode.GRAY, img.height, img.width, gray_pixels)

class NumbaGrayscale(Stage):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def apply(self, img: NumpyImage) -> NumpyImage:
        if img.mode == ImageMode.GRAY:
            gray_pixels = _normalize_grayscale(img.pixels, img.height, img.width)
        else:
            gray_pixels = _transform_grayscale(img.pixels, img.height, img.width, self.weights)

        return NumpyImage(ImageMode.GRAY, img.height, img.width, gray_pixels)

@njit(cache=True, parallel=True)
def _normalize_grayscale(pixels: np.ndarray, height: int, width: int):
    gray_pixels = np.empty((height, width), dtype=np.float32)
    
    for i in prange(height):
        for j in range(width):
            gray_pixels[i, j] = pixels[i, j] / 255.0
            
    return gray_pixels

@njit(cache=True, parallel=True)
def _transform_grayscale(pixels: np.ndarray, height: int, width: int, weights: np.ndarray):
    gray_pixels = np.empty((height, width), dtype=np.float32)
    
    wr, wg, wb = weights[0], weights[1], weights[2]
    
    for i in prange(height):
        for j in range(width):
            r = pixels[i, j, 0]
            g = pixels[i, j, 1]
            b = pixels[i, j, 2]
            gray_pixels[i, j] = (wr * r + wg * g + wb * b) / 255.0
            
    return gray_pixels