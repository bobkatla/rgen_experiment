from __future__ import annotations
import click
import torch
from rgen.core.sampling import ddim_sample, ddpm_sample

@click.command(help="Sample images from an EMA diffusion checkpoint (DDIM/DDPM, optional CFG).")
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to ckpt_ema_*.pt")
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
@click.option("--n-samples", type=int, default=50_000)
@click.option("--batch-size", type=int, default=100)
@click.option("--steps", type=int, default=250, help="Number of DDIM steps (e.g., 250, 1000).")
@click.option("--guidance/--no-guidance", default=True, help="Enable classifier-free guidance.")
@click.option("--w", "guidance_weight", type=float, default=3.0, help="CFG scale.")
@click.option("--class-balance/--no-class-balance", default=True)
@click.option("--seed", type=int, default=0)
@click.option("--classes-json", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Optional path to classes.json (normally inferred from ckpt folder).")
@click.option("--sample-type", "-type", type=click.Choice(["ddim", "ddpm"]), default="ddim", help="Sampling type.")
def sample(ckpt, out_dir, n_samples, batch_size, steps, guidance, guidance_weight, class_balance, seed, classes_json, sample_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sample_type == "ddpm":
        ddpm_sample(
            ckpt_path=ckpt,
            out_dir=out_dir,
            n_samples=n_samples,
            batch_size=batch_size,
            steps=steps,
            class_balance=class_balance,
            seed=seed,
            device=device,
            classes_json=classes_json,
        )
    else:
        ddim_sample(
            ckpt_path=ckpt,
            out_dir=out_dir,
            n_samples=n_samples,
            batch_size=batch_size,
            steps=steps,
            guidance_weight=guidance_weight,
            use_guidance=guidance,
            class_balance=class_balance,
            seed=seed,
            device=device,
            classes_json=classes_json,
        )
    click.echo(f"Saved {n_samples} images to {out_dir}")
