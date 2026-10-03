"""Every setting of a training run in one place."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3          # the number to tune first
    batch_size: int = 64      # memory and speed trade-off
    epochs: int = 5           # watch validation loss, do not guess
    hidden: int = 64          # model size for the MLP
    weight_decay: float = 0.0
    seed: int = 0             # same seed, same run

    def to_dict(self):
        return asdict(self)
