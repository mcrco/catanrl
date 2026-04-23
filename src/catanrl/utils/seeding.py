import hashlib
import random
import sys
from typing import Any

import numpy as np
import torch


MAX_SEED = sys.maxsize


def derive_seed(root_seed: int, *components: Any) -> int:
    """Derive a stable child seed from a root seed and labels."""
    payload = "|".join([str(root_seed), *(str(component) for component in components)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % MAX_SEED


def derive_map_and_game_seeds(episode_seed: int) -> tuple[int, int]:
    """Split an episode seed into independent map and game seeds."""
    return (
        derive_seed(episode_seed, "map"),
        derive_seed(episode_seed, "game"),
    )


def set_global_seeds(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch in one place."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
