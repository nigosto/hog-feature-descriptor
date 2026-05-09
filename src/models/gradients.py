from dataclasses import dataclass

@dataclass
class Gradients:
    height: int
    width: int
    magnitudes: list[float]
    orientations: list[float]