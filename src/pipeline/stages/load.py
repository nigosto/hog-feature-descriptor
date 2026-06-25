from .. import Stage
from src.models import Image, NumpyImage


class LoadImage(Stage):
    def apply(self, path: str) -> Image:
        return Image.from_file(path)


class VectorizedLoadImage(Stage):
    def apply(self, path: str) -> NumpyImage:
        return NumpyImage.from_file(path)
