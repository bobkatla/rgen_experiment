from __future__ import annotations
import os
import json
import click
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as tvutils

from rgen.utils.seed import set_seed
from rgen.utils.log import JsonlLogger, save_text
from rgen.core.model import UNet32
from rgen.core.diffusion import DiffusionConfig, GaussianDiffusion
from rgen.core.ema import EMA
import torch.nn.functional as F
from rgen.models.y_encoder import YEncoder

# URC deps (guarded import so the baseline still runs without urc installed)
try:
    from urc.config import URCConfig, MDNConfig, QuantileConfig, LossConfig
    from urc.core import URCModule
    from urc.io import save_urc, load_urc
    _URC_AVAILABLE = True
except Exception:
    _URC_AVAILABLE = False

_NORM_TO_MINUS1_1 = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def _cycle(loader):
    while True:
        for batch in loader:
            yield batch

@click.command(help="Train CIFAR-10 class-conditional diffusion (baseline).")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default="data/cifar10/train")
@click.option("--out-dir", type=click.Path(file_okay=False), default="runs/cifar10_unet_eps")
@click.option("--batch-size", type=int, default=256)
@click.option("--lr", type=float, default=2e-4)
@click.option("--timesteps", type=int, default=4000, help="Diffusion steps for training schedule.")
@click.option("--iters", type=int, default=800_000, help="Total training iterations (optimizer steps).")
@click.option("--ema", type=float, default=0.9999)
@click.option("--amp/--no-amp", default=True)
@click.option("--seed", type=int, default=0)
@click.option("--save-interval", type=int, default=10_000)
@click.option("--label-drop-prob", type=float, default=0.0, help="Set >0.0 to enable classifier-free guidance training.")
@click.option("--num-workers", type=int, default=4)
@click.option("--use-urc/--no-use-urc", default=False, help="Enable URC region regularizer.")
@click.option("--urc-weight", type=float, default=0.0, help="URC loss weight (can be ramped).")
@click.option("--urc-alpha", type=float, default=0.05, help="URC quantile alpha (e.g., 0.05).")
@click.option("--urc-dy", type=int, default=128, help="Feature dim for y-encoder.")
@click.option("--urc-resume", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to URC checkpoint to resume.")
def train(
    data_dir: str,
    out_dir: str,
    batch_size: int,
    lr: float,
    timesteps: int,
    iters: int,
    ema: float,
    amp: bool,
    seed: int,
    save_interval: int,
    label_drop_prob: float,
    num_workers: int,
    use_urc: bool,
    urc_weight: float,
    urc_alpha: float,
    urc_dy: int,
    urc_resume: str,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (expects folder structure from your exporter)
    tfm = transforms.Compose([transforms.ToTensor(), _NORM_TO_MINUS1_1])
    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    num_classes = len(ds.classes)
    assert num_classes == 10, f"Expected 10 classes, found {num_classes}."
    cond_null_id = num_classes  # reserve null id == 10
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    loader_it = _cycle(loader)

    # Model + diffusion
    model = UNet32(
        in_ch=3, base_ch=128, ch_mults=(1,2,2),
        num_res_blocks=3, attn_resolutions=(16,),
        num_classes=num_classes, cond_null_id=cond_null_id,
        dropout=0.0, predict_sigma=True
    ).to(device)

    df_cfg = DiffusionConfig(image_size=32, timesteps=timesteps, learn_sigma=True, predict_type="eps")
    diffusion = GaussianDiffusion(model, df_cfg).to(device)

    # EMA and opt
    ema_helper = EMA(model, decay=ema)

    # Logging
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump({"classes": ds.classes, "cond_null_id": cond_null_id}, f, indent=2)
    cfg_txt = json.dumps({
        "data_dir": data_dir, "out_dir": out_dir, "batch_size": batch_size, "lr": lr,
        "timesteps": timesteps, "iters": iters, "ema": ema, "amp": amp,
        "seed": seed, "label_drop_prob": label_drop_prob
    }, indent=2)
    save_text(cfg_txt, os.path.join(out_dir, "config.json"))
    logger = JsonlLogger(os.path.join(out_dir, "train_metrics.jsonl"))

    # ---- optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # --- URC: build (optional)
    urc = None
    y_enc = None
    if use_urc:
        if not _URC_AVAILABLE:
            raise click.ClickException("URC not installed. pip/uv install the 'urc' package or disable --use-urc.")
        # y-encoder
        y_enc = YEncoder(d_y=urc_dy).to(device, dtype=torch.float32)

        # URC config
        urc_cfg = URCConfig(
            phi=None,  # we pass y directly
            mdn=MDNConfig(d_y=urc_dy, d_e=num_classes, n_comp=4, hidden=128, var_floor=1e-3, lr=5e-4),
            quant=QuantileConfig(num_classes=num_classes, window_size=2048, warmup_min=128, alpha=urc_alpha),
            loss=LossConfig(w_acc=1.0, w_sep=0.0, w_size=0.0, margin_acc=0.0, mode_acc="softplus"),
        )
        urc = URCModule(urc_cfg).to(device, dtype=torch.float32)

        if urc_resume is not None:
            urc = load_urc(urc_resume, map_location=device).to(device)
    
        # URC hook closure captures diagnostics into this dict
    urc_last: dict = {}
    def urc_fn(x0_pred: torch.Tensor, x_real: torch.Tensor,
            labels_true: torch.Tensor, _t_idx: torch.Tensor):
        if urc is None or y_enc is None or urc_weight == 0.0:
            # return a scalar with the same dtype as the outer loss
            return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)

        # (optional) compute features under AMP for speed
        with torch.amp.autocast("cuda", enabled=amp):
            with torch.no_grad():
                y_real_mixed = y_enc(x_real)     # no grad on real branch
            y_gen_mixed = y_enc(x0_pred)         # grad flows to UNet via x0_pred
            one_hot = F.one_hot(labels_true, num_classes=num_classes).to(x0_pred.device).float()

        # >>> URC must be FP32 (slogdet requires it) <<<
        with torch.amp.autocast("cuda", enabled=False):
            out = urc.step(
                y_real=y_real_mixed.float(),
                e_real=one_hot.float(),
                labels_real=labels_true,
                y_gen=y_gen_mixed.float(),
                e_gen=one_hot.float(),
                labels_gen=labels_true,
            )
            urc_loss_fp32 = urc_weight * out["loss"]  # FP32 scalar

        # capture diagnostics
        urc_last.clear()
        for k, v in out.items():
            try:
                urc_last[k] = float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
            except Exception:
                urc_last[k] = v

        # Return loss with the SAME dtype as diffusion loss (likely FP16 under AMP)
        return urc_loss_fp32.to(dtype=x0_pred.dtype)
    # ---- end URC hook

    # ---- train loop
    model.train()
    step = 0
    while step < iters:
        x, y = next(loader_it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        # keep the true labels for URC
        labels_true = y.clone()

        # build conditional labels for the diffusion model (may include null id)
        if label_drop_prob > 0.0:
            mask = (torch.rand_like(y.float()) < label_drop_prob)
            y_cond = torch.where(mask, torch.full_like(y, cond_null_id), y)
        else:
            y_cond = y

        t = torch.randint(0, timesteps, (x.size(0),), device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=amp):
            out = diffusion.training_loss(
                x, t, y_cond, labels_true,
                urc_fn if (urc is not None and urc_weight > 0.0) else None
            )
            loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()       # ✅ scale + backward
        scaler.step(optimizer)              # ✅ step the optimizer via scaler
        scaler.update()

        ema_helper.update(model)            # ✅ after optimizer step
        step += 1

        if step % 100 == 0:
            rec = {"step": step, "loss": float(loss.detach().cpu())}
            if "urc_loss" in out:
                rec["urc/loss_weighted"] = float(out["urc_loss"].detach().cpu())
            if urc_last:
                # common URC diagnostics
                for k in ("loss", "loss_acc", "hit_at_alpha", "tau_mean", "n_warm"):
                    if k in urc_last:
                        rec[f"urc/{k}"] = urc_last[k]
            logger.write(rec)


        if step % save_interval == 0 or step == iters:
            # save raw
            raw_ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg_txt,
            }
            torch.save(raw_ckpt, os.path.join(out_dir, f"ckpt_raw_{step:07d}.pt"))

            # save EMA snapshot
            ema_model = UNet32(
                in_ch=3, base_ch=128, ch_mults=(1,2,2),
                num_res_blocks=3, attn_resolutions=(16,),
                num_classes=num_classes, cond_null_id=cond_null_id,
                dropout=0.0, predict_sigma=True
            ).to(device)
            ema_helper.copy_to(ema_model)
            ema_ckpt = {"step": step, "model": ema_model.state_dict(), "config": cfg_txt}
            torch.save(ema_ckpt, os.path.join(out_dir, f"ckpt_ema_{step:07d}.pt"))

            # quick sanity image (training batch preview)
            ema_model.eval()
            with torch.no_grad():
                grid = tvutils.make_grid((x[:64].clamp(-1,1) + 1) * 0.5, nrow=8)
                tvutils.save_image(grid, os.path.join(out_dir, f"train_batch_{step:07d}.png"))
            
            # URC snapshot (optional)
            if urc is not None:
                try:
                    save_urc(os.path.join(out_dir, f"urc_{step:07d}.pt"), urc, include_optimizer=True)
                except Exception as e:
                    click.echo(f"[warn] URC save failed at step {step}: {e}")


    click.echo(f"Training finished at step {step}. Checkpoints in {out_dir}.")
