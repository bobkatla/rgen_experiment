import click
from rgen import __version__
from rgen.cli.prep_data import prep_data
from rgen.cli.train import train

@click.group()
@click.version_option(version=__version__, prog_name="rgen")
def main():
    """RGEN - Running Generative AI Experiments"""
    pass


@main.command()
def info():
    """Display information about the rgen package."""
    click.echo(f"rgen version {__version__}")
    click.echo("Running generative AI experiments package.")

main.add_command(prep_data)
main.add_command(train)
