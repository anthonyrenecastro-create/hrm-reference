"""Counter-derived random streams with no hidden mutable generator state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeterministicContext:
    seed: int
    step: int
    run_id: str

    def stream_seed(self, stream: str) -> int:
        payload = json.dumps([self.seed, self.step, self.run_id, stream], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little", signed=False)

    def generator(self, stream: str) -> np.random.Generator:
        return np.random.default_rng(self.stream_seed(stream))

    def identity(self, stream: str = "root") -> dict[str, int | str]:
        return {
            "scheme": "sha256-counter-v1",
            "seed": self.seed,
            "step": self.step,
            "stream": stream,
            "stream_seed": self.stream_seed(stream),
        }
