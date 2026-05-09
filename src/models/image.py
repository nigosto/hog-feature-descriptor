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
