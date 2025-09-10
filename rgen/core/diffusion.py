from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

def betas_for_alpha_bar(num_steps: int) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).
    Returns betas shape [num_steps].
    """
    import math
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    return torch.tensor(betas, dtype=torch.float32)

@dataclass
class DiffusionConfig:
    image_size: int = 32
    timesteps: int = 4000
    learn_sigma: bool = True  # if model predicts mean+sigma channels
    predict_type: str = "eps" # "eps" | "v"

class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        betas = betas_for_alpha_bar(cfg.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x0
            + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )

    def training_loss(self, x0: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> dict:
        """
        x0 in [-1,1]. t is int64. y is labels with possible null id for CFG.
        Uses epsilon prediction MSE by default.
        """
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        model_out = self.model(xt, t, y)

        if self.cfg.learn_sigma:
            # predict [eps, sigma], but we keep standard eps-MSE objective;
            # sigma head is ignored in baseline (kept for parity with ADM).
            c = x0.shape[1]
            eps_pred, _sigma = torch.split(model_out, [c, c], dim=1)
        else:
            eps_pred = model_out

        loss = F.mse_loss(eps_pred, noise, reduction="mean")
        return {"loss": loss}
