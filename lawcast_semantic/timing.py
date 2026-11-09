"""Internal search timing spans on the process monotonic clock."""

from dataclasses import dataclass


@dataclass
class DensePhase:
    started: float
    finished: float


@dataclass
class EmbeddingStages:
    tokenization: DensePhase
    model: DensePhase
    cpu_conversion: DensePhase
