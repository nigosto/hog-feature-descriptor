import numpy as np
import math
from .. import Stage
from src.models import Image, NumpyImage, Gradients, NumpyGradients


class SobelGradients(Stage):
    gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

    def apply(self, img: Image) -> Gradients:
        size = img.width * img.height
        magnitudes, orientations = [0.0] * size, [0.0] * size

        for i in range(1, img.height - 1):
            for j in range(1, img.width - 1):
                horizontal_gradient, vertical_gradient = 0.0, 0.0

                for gi in range(3):
                    for gj in range(3):
                        pixel_index = (i + gi - 1) * img.width + j + gj - 1
                        kernel_index = gi * 3 + gj
                        pixel = img.pixels[pixel_index]

                        horizontal_gradient += pixel * self.gx[kernel_index]
                        vertical_gradient += pixel * self.gy[kernel_index]

                magnitudes[i * img.width + j] = math.hypot(
                    horizontal_gradient, vertical_gradient
                )
                orientations[i * img.width + j] = (
                    math.degrees(math.atan2(vertical_gradient, horizontal_gradient))
                    % 180
                )

        return Gradients(img.height, img.width, magnitudes, orientations)


class VectorizedSobelGradients(Stage):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def apply(self, img: NumpyImage) -> NumpyGradients:
        strided = np.lib.stride_tricks.sliding_window_view(img.pixels, (3, 3))
        horizontal_gradients = np.sum(strided * self.gx, axis=(2, 3))
        vertical_gradients = np.sum(strided * self.gy, axis=(2, 3))

        magnitudes = np.hypot(horizontal_gradients, vertical_gradients)
        orientations = (
            np.degrees(np.arctan2(vertical_gradients, horizontal_gradients)) % 180
        )

        magnitudes = np.pad(magnitudes, pad_width=1, mode='constant', constant_values=0.0)
        orientations = np.pad(orientations, pad_width=1, mode='constant', constant_values=0.0)

        return NumpyGradients(img.height, img.width, magnitudes, orientations)
