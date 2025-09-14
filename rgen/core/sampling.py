from __future__ import annotations
import json
import math
import os
from typing import Optional
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from rgen.core.model import UNet32
from rgen.core.diffusion import DiffusionConfig, GaussianDiffusion

def _load_ema_unet_and_cfg(ckpt_path: str, num_classes: int = 10, cond_null_id: int = 10, device: torch.device = torch.device("cpu")):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Parse timesteps from saved config (stringified JSON) if present
    timesteps = 4000
    if "config" in ckpt:
        try:
            cfg = json.loads(ckpt["config"]) if isinstance(ckpt["config"], str) else ckpt["config"]
            if "timesteps" in cfg:
                timesteps = int(cfg["timesteps"])
        except Exception:
            pass

    model = UNet32(
        in_ch=3, base_ch=128, ch_mults=(1,2,2),
        num_res_blocks=3, attn_resolutions=(16,),
        num_classes=num_classes, cond_null_id=cond_null_id,
        dropout=0.0, predict_sigma=True
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().requires_grad_(False)

    df = GaussianDiffusion(model, DiffusionConfig(image_size=32, timesteps=timesteps, learn_sigma=True, predict_type="eps")).to(device)
    return model, df, timesteps

def _build_schedule(timesteps: int, steps: int) -> torch.Tensor:
    """Create a decreasing list of t indices for DDIM with `steps` evaluations."""
    if steps >= timesteps:
        return torch.arange(timesteps - 1, -1, -1, dtype=torch.long)
    stride = math.ceil(timesteps / steps)
    t = torch.arange(timesteps - 1, -1, -stride, dtype=torch.long)  # e.g., 3999, 3959, ...
    if t[-1] != 0:
        t = torch.cat([t, torch.zeros(1, dtype=torch.long)])
    # ensure strictly descending and unique
    t = torch.unique_consecutive(t)
    return t

@torch.no_grad()
def ddim_sample(
    ckpt_path: str,
    out_dir: str,
    n_samples: int = 50_000,
    batch_size: int = 100,
    steps: int = 250,
    guidance_weight: float = 0.0,
    use_guidance: bool = False,
    class_balance: bool = True,
    seed: int = 0,
    device: Optional[torch.device] = None,
    classes_json: Optional[str] = None,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(out_dir, exist_ok=True)

    # recover class info (for null id & balanced labels)
    if classes_json is None:
        # default: look next to ckpt path
        classes_json = os.path.join(os.path.dirname(ckpt_path), "classes.json")
    if os.path.exists(classes_json):
        with open(classes_json, "r") as f:
            cj = json.load(f)
        classes = cj.get("classes", [str(i) for i in range(10)])
        cond_null_id = int(cj.get("cond_null_id", 10))
        num_classes = len(classes)
    else:
        classes = [str(i) for i in range(10)]
        cond_null_id = 10
        num_classes = 10

    model, diffusion, timesteps = _load_ema_unet_and_cfg(ckpt_path, num_classes=num_classes, cond_null_id=cond_null_id, device=device)

    # labels to generate
    if class_balance:
        per = int(math.ceil(n_samples / num_classes))
        labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(per)[:n_samples]
    else:
        labels = torch.randint(0, num_classes, (n_samples,), dtype=torch.long)
    labels = labels.to(device)

    # DDIM schedule (indices descending)
    t_seq = _build_schedule(timesteps, steps).to(device)  # e.g., [3999, ..., 0]
    a_bar = diffusion.alphas_cumprod.to(device)

    # sampling
    g = torch.Generator(device=device).manual_seed(seed)
    saved = 0
    pbar = tqdm(range(0, n_samples, batch_size), desc=f"DDIM{' + CFG' if use_guidance else ''} sampling", ncols=88)
    while saved < n_samples:
        bs = min(batch_size, n_samples - saved)
        y = labels[saved:saved+bs]
        x = torch.randn(bs, 3, 32, 32, device=device, generator=g)

        for ti, t in enumerate(t_seq):
            t_int = int(t.item())
            at = a_bar[t_int]
            at_prev = a_bar[t_seq[ti + 1].item()] if ti + 1 < len(t_seq) else torch.tensor(1.0, device=device)

            # predict eps (with or without CFG)
            if use_guidance and guidance_weight != 0.0:
                y_null = torch.full_like(y, cond_null_id)
                eps_u = model(x, t.expand(bs), y_null)[:, :3]
                eps_c = model(x, t.expand(bs), y)[:, :3]
                eps   = eps_u + guidance_weight * (eps_c - eps_u)
            else:
                eps   = model(x, t.expand(bs), y)[:, :3]

            # DDIM update (eta=0)
            sqrt_at = at.sqrt()
            sqrt_one_minus_at = (1.0 - at).sqrt()
            x0_pred = (x - sqrt_one_minus_at * eps) / sqrt_at
            sqrt_at_prev = at_prev.sqrt()
            sqrt_one_minus_at_prev = (1.0 - at_prev).sqrt()
            x = sqrt_at_prev * x0_pred + sqrt_one_minus_at_prev * eps

            if t_int == 0:
                break

        # save to disk
        imgs = (x.clamp(-1, 1) + 1) * 0.5
        for i in range(bs):
            save_image(imgs[i], os.path.join(out_dir, f"{saved + i:06d}.png"))
        saved += bs
        pbar.update(1)

    # manifest
    with open(os.path.join(out_dir, "manifest_ddim.json"), "w") as f:
        json.dump({
            "n_samples": n_samples,
            "batch_size": batch_size,
            "steps": steps,
            "use_guidance": use_guidance,
            "guidance_weight": guidance_weight,
            "ckpt": ckpt_path,
            "classes_json": classes_json,
            "seed": seed
        }, f, indent=2)

@torch.no_grad()
def ddpm_sample(ckpt_path, out_dir, n_samples=10000, batch_size=100, steps=1000,
                class_balance=True, seed=0, device=None, classes_json=None, use_guidance=False, guidance_weight=0.0):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.makedirs(out_dir, exist_ok=True)

    # recover class info (for null id & balanced labels)
    if classes_json is None:
        # default: look next to ckpt path
        classes_json = os.path.join(os.path.dirname(ckpt_path), "classes.json")
    if os.path.exists(classes_json):
        with open(classes_json, "r") as f:
            cj = json.load(f)
        classes = cj.get("classes", [str(i) for i in range(10)])
        cond_null_id = int(cj.get("cond_null_id", 10))
        num_classes = len(classes)
    else:
        classes = [str(i) for i in range(10)]
        cond_null_id = 10
        num_classes = 10

    # load model + diffusion (same helper you already have)
    model, diffusion, timesteps = _load_ema_unet_and_cfg(ckpt_path, device=device)

    # labels
    num_classes = 10
    if class_balance:
        per = int(math.ceil(n_samples / num_classes))
        labels = torch.arange(num_classes).repeat_interleave(per)[:n_samples]
    else:
        labels = torch.randint(0, num_classes, (n_samples,))
    labels = labels.to(device)

    # choose evenly spaced DDPM indices
    t_seq = _build_schedule(timesteps, steps).to(device)  # descending indices
    a_bar = diffusion.alphas_cumprod.to(device)
    betas = diffusion.betas.to(device)

    g = torch.Generator(device=device).manual_seed(seed)
    saved = 0
    pbar = tqdm(range(0, n_samples, batch_size), desc="DDPM sampling", ncols=88)
    while saved < n_samples:
        bs = min(batch_size, n_samples - saved)
        y = labels[saved:saved+bs]
        x = torch.randn(bs, 3, 32, 32, device=device, generator=g)

        for ti, t in enumerate(t_seq):
            t_int = int(t.item())
            out = model(x, t.expand(bs), y)
            eps = out[:, :3]

            if use_guidance and guidance_weight != 0.0:
                y_null = torch.full_like(y, cond_null_id)
                eps_u = model(x, t.expand(bs), y_null)[:, :3]
                eps_c = model(x, t.expand(bs), y)[:, :3]
                eps   = eps_u + guidance_weight * (eps_c - eps_u)
            else:
                eps   = model(x, t.expand(bs), y)[:, :3]

            at = a_bar[t_int]
            sqrt_at = at.sqrt()
            sqrt_one_minus_at = (1.0 - at).sqrt()
            x0_pred = (x - sqrt_one_minus_at * eps) / sqrt_at

            # DDPM mean
            at_prev = a_bar[t_seq[ti + 1].item()] if ti + 1 < len(t_seq) else torch.tensor(1.0, device=device)
            coef1 = (betas[t_int].sqrt())
            coef2 = ((1 - at_prev - betas[t_int]).clamp(min=0).sqrt())
            mean = ( (at_prev.sqrt()) * x0_pred + (coef2) * eps )

            if t_int > 0:
                # Replace torch.randn_like with torch.randn for compatibility
                noise = torch.randn(x.size(), device=x.device)
                x = mean + coef1 * noise
            else:
                x = mean

        imgs = (x.clamp(-1,1) + 1) * 0.5
        for i in range(bs):
            save_image(imgs[i], os.path.join(out_dir, f"{saved+i:06d}.png"))
        saved += bs
        pbar.update(1)

    # manifest
    with open(os.path.join(out_dir, "manifest_ddpm.json"), "w") as f:
        json.dump({
            "n_samples": n_samples,
            "batch_size": batch_size,
            "steps": steps,
            "use_guidance": use_guidance,
            "guidance_weight": guidance_weight,
            "ckpt": ckpt_path,
            "classes_json": classes_json,
            "seed": seed
        }, f, indent=2)
