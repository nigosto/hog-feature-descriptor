from dataclasses import dataclass

@dataclass
class Blocks:
    rows: int
    columns: int
    block_size: int
    descriptors: list[list[float]]