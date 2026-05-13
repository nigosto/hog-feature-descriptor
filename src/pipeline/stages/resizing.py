import math
from .. import Stage
from src.models import Image, ImageMode

class Resizing(Stage):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply(self, img: Image) -> Image:
        if img.mode == ImageMode.GRAY:
            resized_image_pixels = self.resize_grayscale(img)
        else:
            resized_image_pixels = self.resize_rgb(img)
        
        return Image(img.mode, self.height, self.width, resized_image_pixels)

    def resize_grayscale(self, img: Image):
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
                    max(0, math.floor(y_min)),
                    min(img.height, math.ceil(y_max))
                ):
                    for sj in range(
                        max(0, math.floor(x_min)), 
                        min(img.width, math.ceil(x_max))
                    ):                        
                        w = max(0, min(x_max, sj + 1) - max(x_min, sj))
                        h = max(0, min(y_max, si + 1) - max(y_min, si))
                        area = w * h

                        value += img.pixels[si * img.width + sj] * area
                        total_area += area
                value /= total_area 
                resized_image_pixels.append(value)

        return resized_image_pixels

    def resize_rgb(self, img: Image):
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
                    max(0, math.floor(y_min)),
                    min(img.height, math.ceil(y_max))
                ):
                    for sj in range(
                        max(0, math.floor(x_min)), 
                        min(img.width, math.ceil(x_max))
                    ):
                        w = max(0, min(x_max, sj + 1) - max(x_min, sj))
                        h = max(0, min(y_max, si + 1) - max(y_min, si))
                        area = w * h

                        for c in range(3):
                            value[c] += img.pixels[si * img.width + sj][c] * area
                        total_area += area
                resized_image_pixels.append(tuple((p / total_area for p in value)))

        return resized_image_pixels