from typing import Dict, Optional

import math
import torch
import torch.nn as nn

from .base import BaseScheduler


class AnnealedScheduler(BaseScheduler):
    def __init__(
        self,
        noise_start: float = 1.0,
        noise_end: float = 0.3,
        aug_start: float = 0.0,
        aug_end: float = 0.8,
        reg_start: float = 0.0,
        reg_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.aug_start = aug_start
        self.aug_end = aug_end
        self.reg_start = reg_start
        self.reg_end = reg_end

    def _cosine_anneal(self, start: float, end: float, u: float) -> float:
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * u))

    def forward(
        self,
        step: int,
        total_steps: int,
        state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if total_steps <= 1:
            u = 1.0
        else:
            u = step / float(total_steps - 1)
        noise_sigma = self._cosine_anneal(self.noise_start, self.noise_end, u)
        aug_p = self._cosine_anneal(self.aug_start, self.aug_end, u)
        reg_lambda = self._cosine_anneal(self.reg_start, self.reg_end, u)
        return {
            "noise_sigma": float(noise_sigma),
            "augment_p": float(aug_p),
            "reg_lambda": float(reg_lambda),
            "u": float(u),
        }
