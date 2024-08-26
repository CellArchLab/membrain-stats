"""Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer.core import TyperGroup
from typing import List
from typer import Option



class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False
)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    MemBrain-pick's data conversion / mesh projection module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    membrain-pick --help

    -------

    Example:
    -------
    membrain-pick process-folder --mb-folder <path-to-your-folder> --tomo-path <path-to-tomo> 
        --output-folder <path-to-store-meshes>

    -------
    """



@cli.command(name="protein_concentration", no_args_is_help=True)
def protein_concentration(
    in_folder: str = Option(  # noqa: B008
        ..., help="Path to the directory containing either .h5 files or .obj and .star files", **PKWARGS
    ),
    out_folder: str = Option(  # noqa: B008
        "./stats/protein_centration", help="Path to the folder where computed stats should be stored."
    ),
    pixel_size_multiplier: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.",
    ),
    only_one_side: bool = Option(  # noqa: B008
        False, help="If True, only one side of the membrane will be considered for area calculation."
    ),
    exclude_edges: bool = Option(  # noqa: B008
        False, help="If True, the edges of the membrane will be excluded from the area calculation."
    ),
    edge_exclusion_width: float = Option(  # noqa: B008
        50., help="Width of the edge exclusion zone in Anstrom."
    ),
):
    """Compute the protein concentration in all membrane meshes in a folder.

    Example
    -------
    membrain_stats protein_concentration --in-folder <path-to-your-folder> --out-folder <path-to-store-meshes> --mesh-pixel-size 14.08 --only-one-side --exclude-edges --edge-exclusion-width 50
    """
    
    from membrain_stats.protein_concentration import (
        protein_concentration_folder,
    )
    protein_concentration_folder(
        in_folder=in_folder,
        out_folder=out_folder,
        pixel_size_multiplier=pixel_size_multiplier,
        only_one_side=only_one_side,
        exclude_edges=exclude_edges,
        edge_exclusion_width=edge_exclusion_width,
    )