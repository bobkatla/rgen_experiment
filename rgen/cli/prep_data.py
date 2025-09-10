from __future__ import annotations
import os
import click
from rgen.dataio.cifar_export import export_cifar10_as_images

@click.group()
def prep_data():
    """Data preparation commands."""
    pass

@prep_data.command(help="Download and export CIFAR-10 into class-folder PNGs.")
@click.option("--root", type=click.Path(file_okay=False), default="data",
              help="Root folder to create data/cifar10/{train,test}/... under.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing PNG files.")
def cifar10_export(root: str, overwrite: bool) -> None:
    os.makedirs(root, exist_ok=True)
    summary = export_cifar10_as_images(root, splits=("train", "test"), overwrite=overwrite)
    click.echo(f"Done. Summary: {summary}")

