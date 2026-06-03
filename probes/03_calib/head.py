"""Stage 2a regressor head: item embedding -> (difficulty b, log discrimination).

Shared by train_stage2.py (offline) and model.py (runtime) so the architecture
can never drift between fit and inference. Outputs are standardized; invert with
the (mean, std) saved in stage2_config.json.
"""

from __future__ import annotations

import torch.nn as nn


class ParamHead(nn.Module):
    def __init__(self, in_dim: int, h1: int = 256, h2: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 2),  # (b_z, log_a_z), standardized
        )

    def forward(self, x):
        return self.net(x)
