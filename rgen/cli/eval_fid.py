from __future__ import annotations
import click
from cleanfid import fid

@click.command(help="Compute FID using clean-fid.")
@click.option("--gen-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Folder with generated PNGs")
@click.option("--reference", type=click.Choice(["cifar10_test", "folder"]), default="cifar10_test",
              help="Use built-in CIFAR-10 test stats or compare to a folder.")
@click.option("--ref-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="If reference=folder, path to real images folder.")
def eval_fid(gen_dir, reference, ref_dir):
    if reference == "cifar10_test":
        score = fid.compute_fid(gen_dir, dataset_name="cifar10", dataset_split="test")
    else:
        if ref_dir is None:
            raise click.ClickException("Provide --ref-dir when reference=folder")
        score = fid.compute_fid(gen_dir, ref_dir)
    click.echo(f"FID: {score:.4f}")
