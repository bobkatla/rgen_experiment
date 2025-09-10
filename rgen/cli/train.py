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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

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

    # ---- train loop
    model.train()
    step = 0
    while step < iters:
        x, y = next(loader_it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        # classifier-free label dropout (keep at 0.0 for baseline)
        if label_drop_prob > 0.0:
            mask = (torch.rand_like(y.float()) < label_drop_prob)
            y = torch.where(mask, torch.full_like(y, cond_null_id), y)

        t = torch.randint(0, timesteps, (x.size(0),), device=device, dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=amp):
            out = diffusion.training_loss(x, t, y)
            loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()       # ✅ scale + backward
        scaler.step(optimizer)              # ✅ step the optimizer via scaler
        scaler.update()

        ema_helper.update(model)            # ✅ after optimizer step
        step += 1

        if step % 100 == 0:
            logger.write({"step": step, "loss": float(loss.detach().cpu())})

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

    click.echo(f"Training finished at step {step}. Checkpoints in {out_dir}.")
