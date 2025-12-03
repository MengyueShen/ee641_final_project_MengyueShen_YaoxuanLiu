from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseScheduler
from .annealed import AnnealedScheduler


class AdaptiveAnnealedScheduler(BaseScheduler):
    def __init__(
        self,
        base_scheduler: Optional[AnnealedScheduler] = None,
        hidden_dim: int = 32,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        ema_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_scheduler = base_scheduler or AnnealedScheduler()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ema_alpha = ema_alpha

        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

        self.register_buffer("ema_loss_g", torch.tensor(0.0))
        self.register_buffer("ema_loss_d", torch.tensor(0.0))
        self.register_buffer("ema_grad_g", torch.tensor(0.0))
        self.register_buffer("ema_grad_d", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _update_ema_value(self, current: float, ema_buf: torch.Tensor) -> None:
        value = torch.tensor(
            float(current),
            dtype=torch.float32,
            device=ema_buf.device,
        )
        if not bool(self.initialized):
            ema_buf.copy_(value)
        else:
            ema_buf.mul_(1.0 - self.ema_alpha).add_(self.ema_alpha * value)

    def forward(
        self,
        step: int,
        total_steps: int,
        state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        base = self.base_scheduler(step, total_steps, state)
        u = base["u"]

        if state is None:
            state = {}

        loss_g = state.get("loss_g", 0.0)
        loss_d = state.get("loss_d", 0.0)
        grad_g = state.get("grad_norm_g", 0.0)
        grad_d = state.get("grad_norm_d", 0.0)

        self._update_ema_value(loss_g, self.ema_loss_g)
        self._update_ema_value(loss_d, self.ema_loss_d)
        self._update_ema_value(grad_g, self.ema_grad_g)
        self._update_ema_value(grad_d, self.ema_grad_d)
        if not bool(self.initialized):
            self.initialized.fill_(True)

        loss_g_ema = float(self.ema_loss_g.item())
        loss_d_ema = float(self.ema_loss_d.item())
        grad_g_ema = float(self.ema_grad_g.item())
        grad_d_ema = float(self.ema_grad_d.item())

        x = torch.tensor(
            [loss_g_ema, loss_d_ema, grad_g_ema, grad_d_ema, u],
            dtype=torch.float32,
            device=self.mlp[0].weight.device,
        )
        h = self.mlp(x)
        scale_raw = (h + 1.0) * 0.5
        scale = self.min_scale + (self.max_scale - self.min_scale) * scale_raw

        base_noise = torch.tensor(
            float(base["noise_sigma"]),
            dtype=torch.float32,
            device=self.mlp[0].weight.device,
        )
        base_aug = torch.tensor(
            float(base["augment_p"]),
            dtype=torch.float32,
            device=self.mlp[0].weight.device,
        )
        base_reg = torch.tensor(
            float(base["reg_lambda"]),
            dtype=torch.float32,
            device=self.mlp[0].weight.device,
        )

        noise_sigma = base_noise * scale[0]
        augment_p = base_aug * scale[1]
        augment_p = torch.clamp(augment_p, 0.0, 1.0)
        reg_lambda = base_reg * scale[2]

        return {
            "noise_sigma": noise_sigma,
            "augment_p": augment_p,
            "reg_lambda": reg_lambda,
            "u": u,
        }
