from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TrainConfig:
    """Hyper‑parameters for a training run."""
    lr: float = 1e-3          # learning rate
    hidden: int = 16          # size of hidden layer
    epochs: int = 10          # number of training epochs
    batch_size: int = 32      # mini‑batch size
    seed: int = 42            # random seed for reproducibility

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of the config."""
        return asdict(self)