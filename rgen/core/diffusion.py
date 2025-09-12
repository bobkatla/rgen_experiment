from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

def betas_for_alpha_bar(num_steps: int) -> torch.Tensor:
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
    learn_sigma: bool = True
    predict_type: str = "eps"  # "eps" | "v"

class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

        betas = betas_for_alpha_bar(cfg.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x0
            + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )

    def training_loss(self, x0: torch.Tensor, t: torch.Tensor, y: torch.Tensor, labels_true: torch.Tensor, urc_fn=None) -> dict:
        """
        x0 in [-1,1], t int64. If urc_fn is provided, compute x0_pred and add extra loss.
        urc_fn signature: (x0_pred, x0_real, labels, t) -> scalar Tensor (already weighted).
        """
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        model_out = self.model(xt, t, y)
        if self.cfg.learn_sigma:
            c = x0.shape[1]
            eps_pred, _sigma = torch.split(model_out, [c, c], dim=1)
        else:
            eps_pred = model_out

        loss_eps = F.mse_loss(eps_pred, noise, reduction="mean")
        out = {"loss": loss_eps}

        if urc_fn is not None:
            at   = self.sqrt_alphas_cumprod[t][:, None, None, None]
            omt  = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            x0_pred = (xt - omt * eps_pred) / (at + 1e-8)
            urc_loss = urc_fn(x0_pred.clamp(-1, 1), x0, labels_true, t)
            out["urc_loss"] = urc_loss
            out["loss"] = loss_eps + urc_loss

        return out
