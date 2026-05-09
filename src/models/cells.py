from dataclasses import dataclass

@dataclass
class Cells:
    rows: int
    columns: int
    size: int
    histograms: list[list[float]]