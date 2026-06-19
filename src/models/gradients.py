import numpy as np
from dataclasses import dataclass

@dataclass
class Gradients:
    height: int
    width: int
    magnitudes: list[float]
    orientations: list[float]

@dataclass
class NumpyGradients:
    height: int
    width: int
    magnitudes: np.ndarray
    orientations: np.ndarray