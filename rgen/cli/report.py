from __future__ import annotations
import os
import json
import click

def _read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

@click.command(help="Aggregate sweep metrics.jsonl → Markdown table. Prints to stdout or writes to a file.")
@click.option("--sweep-root", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out-md", type=click.Path(dir_okay=False), default=None, help="Optional path to write a markdown file.")
def report(sweep_root, out_md):
    metrics_path = os.path.join(sweep_root, "metrics.jsonl")
    rows = _read_jsonl(metrics_path)
    if not rows:
        click.echo("No metrics found.")
        return
    rows.sort(key=lambda r: (r.get("steps", 1<<30), r.get("fid", 1e9)))

    # build markdown
    lines = []
    lines.append("| steps | w | n_samples | FID | out_dir |")
    lines.append("|------:|---:|----------:|----:|--------|")
    for r in rows:
        lines.append(f"| {r['steps']} | {r['w']} | {r['n_samples']} | {r['fid']:.4f} | `{os.path.relpath(r['out_dir'], sweep_root)}` |")
    md = "\n".join(lines)

    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        click.echo(f"Wrote {out_md}")
    else:
        click.echo(md)
