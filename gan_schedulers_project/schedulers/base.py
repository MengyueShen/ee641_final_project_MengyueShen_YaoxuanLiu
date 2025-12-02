from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        step: int,
        total_steps: int,
        state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError
