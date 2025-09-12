from __future__ import annotations
import os, json, time
import click
import torch
from cleanfid import fid
from rgen.core.sampling import ddim_sample

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

@click.command(help="Sweep CFG weights (and optionally steps) → sample and compute FID. Writes metrics.jsonl.")
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to ema checkpoint.")
@click.option("--out-root", type=click.Path(file_okay=False), required=True, help="Root folder for sweep outputs.")
@click.option("--weights", type=str, default="1,2,3,4", help="Comma-separated CFG weights, e.g. 1,2,3,4")
@click.option("--steps", type=str, default="250", help="Comma-separated DDIM steps, e.g. 250,1000")
@click.option("--n-samples", type=int, default=50000, help="Samples per setting (use 10000 for quick sweeps).")
@click.option("--batch-size", type=int, default=100)
@click.option("--seed", type=int, default=0)
@click.option("--class-balance/--no-class-balance", default=True)
def sweep_cfg(ckpt, out_root, weights, steps, n_samples, batch_size, seed, class_balance):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(out_root)
    metrics_path = os.path.join(out_root, "metrics.jsonl")

    ws = [float(x) for x in weights.split(",") if x.strip()]
    ss = [int(x) for x in steps.split(",") if x.strip()]

    for s in ss:
        for w in ws:
            tag = f"steps{s}_w{w:g}"
            out_dir = os.path.join(out_root, tag)
            # sample
            ddim_sample(
                ckpt_path=ckpt,
                out_dir=out_dir,
                n_samples=n_samples,
                batch_size=batch_size,
                steps=s,
                guidance_weight=w,
                use_guidance=True,
                class_balance=class_balance,
                seed=seed,
                device=device,
                classes_json=None,
            )
            # fid
            score = fid.compute_fid(out_dir, dataset_name="cifar10", dataset_split="test")
            rec = {
                "time": time.time(),
                "ckpt": ckpt,
                "out_dir": out_dir,
                "steps": s,
                "w": w,
                "n_samples": n_samples,
                "fid": float(score),
            }
            print(f"[{tag}] FID={score:.4f}")
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    print(f"Done. Metrics at {metrics_path}")
