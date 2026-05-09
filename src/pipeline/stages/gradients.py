import math
from .. import Stage
from src.models import Image, Gradients


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
