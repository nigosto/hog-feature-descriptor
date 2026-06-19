import numpy as np
from dataclasses import dataclass

@dataclass
class Cells:
    rows: int
    columns: int
    size: int
    histograms: list[list[float]]

@dataclass
class NumpyCells:
    rows: int
    columns: int
    size: int
    histograms: np.ndarray