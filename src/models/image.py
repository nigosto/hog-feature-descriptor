import numpy as np
from dataclasses import dataclass
from typing import Self, Union, Tuple
from enum import StrEnum
from PIL import Image as PILImage


class ImageMode(StrEnum):
    GRAY = "L"
    RGB = "RGB"
    RGBA = "RGBA"


@dataclass
class Image:
    mode: ImageMode
    height: int
    width: int
    pixels: list[Union[float, Tuple[float, float, float]]]

    @classmethod
    def from_file(cls, path: str) -> Self:
        img = PILImage.open(path)
        width, height = img.size

        return cls(ImageMode(img.mode), height, width, list(img.get_flattened_data()))

@dataclass
class NumpyImage:
    mode: ImageMode
    height: int
    width: int
    pixels: np.ndarray

    @classmethod
    def from_file(cls, path: str) -> Self:
        img = PILImage.open(path)
        width, height = img.size
        mode = ImageMode(img.mode)

        if mode == ImageMode.GRAY:
            shape = (height, width)
        else:
            shape = (height, width, 3)

        pixels = np.array(list(img.get_flattened_data()), dtype=np.float32)
        pixels = np.reshape(pixels, shape)

        return cls(mode, height, width, pixels)