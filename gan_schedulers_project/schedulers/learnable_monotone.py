from typing import Dict, Optional

import math
import torch
import torch.nn as nn

from .base import BaseScheduler


class LearnableMonotoneScheduler(BaseScheduler):
    def __init__(
        self,
        k_bins: int = 16,
        min_noise: float = 0.3,
        max_noise: float = 1.0,
        min_aug: float = 0.0,
        max_aug: float = 0.8,
        min_reg: float = 0.0,
        max_reg: float = 1.0,
    ) -> None:
        super().__init__()
        self.k_bins = k_bins
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.min_aug = min_aug
        self.max_aug = max_aug
        self.min_reg = min_reg
        self.max_reg = max_reg

        self.noise_logits = nn.Parameter(torch.zeros(k_bins))
        self.aug_logits = nn.Parameter(torch.zeros(k_bins))
        self.reg_logits = nn.Parameter(torch.zeros(k_bins))

    def _monotone_curve(self, logits: torch.Tensor, u: float) -> torch.Tensor:
        probs = torch.softmax(logits, dim=0)
        cdf = torch.cumsum(probs, dim=0)
        cdf = torch.clamp(cdf, 0.0, 1.0)
        idx_f = u * (self.k_bins - 1)
        idx0 = int(math.floor(idx_f))
        idx1 = min(self.k_bins - 1, idx0 + 1)
        t = idx_f - idx0
        v0 = cdf[idx0]
        v1 = cdf[idx1]
        v = (1.0 - t) * v0 + t * v1
        return v

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
        with torch.no_grad():
            u_clamped = max(0.0, min(1.0, u))
        u = u_clamped

        v_noise = self._monotone_curve(self.noise_logits, u)
        v_aug = self._monotone_curve(self.aug_logits, u)
        v_reg = self._monotone_curve(self.reg_logits, u)

        noise_sigma = self.min_noise + (self.max_noise - self.min_noise) * float(v_noise)
        aug_p = self.min_aug + (self.max_aug - self.min_aug) * float(v_aug)
        reg_lambda = self.min_reg + (self.max_reg - self.min_reg) * float(v_reg)

        return {
            "noise_sigma": noise_sigma,
            "augment_p": aug_p,
            "reg_lambda": reg_lambda,
            "u": u,
        }
