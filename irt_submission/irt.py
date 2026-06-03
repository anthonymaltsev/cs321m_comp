"""Stage 1: 2PL IRT fit (offline).

Model:  P(correct | subject j, item i) = sigmoid(a_i * (theta_j - b_i))
  theta_j  subject ability      (one per subject)
  b_i      item difficulty      (one per item)
  a_i      item discrimination  (one per item, > 0)

Fit by full-batch Adam on the Bernoulli log-likelihood with Gaussian priors
(L2) on theta, b, and log(a). The priors pin the location/scale that a 2PL is
otherwise free to rescale, and a = exp(log_a) keeps discrimination positive,
which removes the reflection ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

A_MIN, A_MAX = 0.05, 4.0  # clamp discrimination at inference for stability


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class IRTFit:
    theta: np.ndarray   # [n_subjects]
    b: np.ndarray       # [n_items] difficulty
    a: np.ndarray       # [n_items] discrimination (>0)


def fit_2pl(
    subj: np.ndarray,
    item: np.ndarray,
    y: np.ndarray,
    n_subjects: int,
    n_items: int,
    *,
    steps: int = 800,
    lr: float = 0.05,
    reg_theta: float = 1e-2,
    reg_b: float = 1e-2,
    reg_loga: float = 1e-1,
    device: str | None = None,
    verbose: bool = False,
) -> IRTFit:
    device = device or pick_device()
    s = torch.as_tensor(subj, dtype=torch.long, device=device)
    it = torch.as_tensor(item, dtype=torch.long, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    theta = torch.zeros(n_subjects, device=device, requires_grad=True)
    b = torch.zeros(n_items, device=device, requires_grad=True)
    log_a = torch.zeros(n_items, device=device, requires_grad=True)  # a = exp(log_a), init 1
    opt = torch.optim.Adam([theta, b, log_a], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        logit = torch.exp(log_a)[it] * (theta[s] - b[it])
        nll = F.binary_cross_entropy_with_logits(logit, yt)
        reg = (reg_theta * theta.pow(2).mean()
               + reg_b * b.pow(2).mean()
               + reg_loga * log_a.pow(2).mean())
        loss = nll + reg
        loss.backward()
        opt.step()
        if verbose and (step % 100 == 0 or step == steps - 1):
            print(f"  step {step:4d}  nll={nll.item():.4f}")

    return IRTFit(
        theta=theta.detach().float().cpu().numpy(),
        b=b.detach().float().cpu().numpy(),
        a=np.exp(log_a.detach().float().cpu().numpy()).clip(A_MIN, A_MAX),
    )


def prob(theta_j: np.ndarray, a_i: np.ndarray, b_i: np.ndarray) -> np.ndarray:
    """Vectorized 2PL probability for aligned arrays."""
    z = np.clip(a_i, A_MIN, A_MAX) * (theta_j - b_i)
    return 1.0 / (1.0 + np.exp(-z))
