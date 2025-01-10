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
    pretty_exceptions_show_locals=False,
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
        ...,
        help="Path to the directory containing either .h5 files or .obj and .star files",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./stats/protein_concentration",
        help="Path to the folder where computed stats should be stored.",
    ),
    pixel_size_multiplier: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.",
    ),
    pixel_size_multiplier_positions: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, loaded positions are multiplied by this value.",
    ),
    only_one_side: bool = Option(  # noqa: B008
        False,
        help="If True, only one side of the membrane will be considered for area calculation.",
    ),
    exclude_edges: bool = Option(  # noqa: B008
        False,
        help="If True, the edges of the membrane will be excluded from the area calculation.",
    ),
    edge_exclusion_width: float = Option(  # noqa: B008
        50.0, help="Width of the edge exclusion zone in Anstrom."
    ),
    edge_percentile: float = Option(  # noqa: B008
        95, help="Percentile to use for edge exclusion."
    ),
    plot: bool = Option(  # noqa: B008
        False,
        help="If True, the protein concentration will be plotted for each mesh.",
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
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
        only_one_side=only_one_side,
        exclude_edges=exclude_edges,
        edge_exclusion_width=edge_exclusion_width,
        edge_percentile=edge_percentile,
        plot=plot,
    )


@cli.command(name="protein_concentration_wrt", no_args_is_help=True)
def protein_concentration_wrt(
    in_folder: str = Option(  # noqa: B008
        ...,
        help="Path to the directory containing either .h5 files or .obj and .star files",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./stats/protein_concentration",
        help="Path to the folder where computed stats should be stored.",
    ),
    pixel_size_multiplier: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.",
    ),
    pixel_size_multiplier_positions: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, loaded positions are multiplied by this value.",
    ),
    exclude_edges: bool = Option(  # noqa: B008
        False,
        help="If True, the edges of the membrane will be excluded from the area calculation.",
    ),
    edge_exclusion_width: float = Option(  # noqa: B008
        50.0, help="Width of the edge exclusion zone in Anstrom."
    ),
    only_one_side: bool = Option(  # noqa: B008
        False,
        help="If True, only one side of the membrane will be considered for area calculation. Works only if exclude_edges is False.",
    ),
    num_bins: int = Option(  # noqa: B008
        25, help="Number of bins to use for the histogram."
    ),
    consider_classes: List[int] = Option(  # noqa: B008
        [-1],
        help="List of classes to consider for protein concentration calculation. If set to -1, all classes will be considered.",
    ),
    with_respect_to_class: int = Option(  # noqa: B008
        0, help="Class with respect to which protein concentration should be computed."
    ),
    geod_distance_method: str = Option(  # noqa: B008
        "exact",
        help="Method to use for computing geodesic distances. Can be either 'exact' or 'fast'.",
    ),
    distance_matrix_method: str = Option(  # noqa: B008
        "geodesic",
        help="Method to use for computing the distance matrix. Can be either 'geodesic' or 'euclidean'.",
    ),
):
    """Compute protein concentrations with respect to distance to a specific point class.

    Example
    -------
    membrain_stats protein_concentration_wrt --in-folder <path-to-your-folder> --out-folder <path-to-store-meshes> --consider-classes 0 --num-bins 25 --with-respect-to-class 1
    """

    from membrain_stats.protein_concentration import (
        protein_concentration_wrt_folder,
    )

    assert not (
        only_one_side and exclude_edges
    ), "Only one of only_one_side and exclude_edges can be True."

    protein_concentration_wrt_folder(
        in_folder=in_folder,
        out_folder=out_folder,
        pixel_size_multiplier=pixel_size_multiplier,
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
        exclude_edges=exclude_edges,
        edge_exclusion_width=edge_exclusion_width,
        only_one_side=only_one_side,
        consider_classes=consider_classes,
        num_bins=num_bins,
        with_respect_to_class=with_respect_to_class,
        geod_distance_method=geod_distance_method,
        distance_matrix_method=distance_matrix_method,
    )


@cli.command(name="geodesic_NN", no_args_is_help=True)
def geodesic_NN(
    in_folder: str = Option(  # noqa: B008
        ...,
        help="Path to the directory containing either .h5 files or .obj and .star files",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./stats/geodesic_distances",
        help="Path to the folder where computed stats should be stored.",
    ),
    pixel_size_multiplier: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.",
    ),
    pixel_size_multiplier_positions: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, loaded positions are multiplied by this value.",
    ),
    num_neighbors: int = Option(  # noqa: B008
        1, help="Number of nearest neighbors to consider."
    ),
    start_classes: List[int] = Option(  # noqa: B008
        [0], help="List of classes to consider for start points."
    ),
    target_classes: List[int] = Option(  # noqa: B008
        [0], help="List of classes to consider for target points."
    ),
    method: str = Option(  # noqa: B008
        "fast",
        help="Method to use for computing geodesic distances. Can be either 'exact' or 'fast'.",
    ),
    c2_symmetry: bool = Option(  # noqa: B008
        False,
        help="If True, the C2 symmetry of the protein will be considered when computing the angles.",
    ),
    project_to_plane: bool = Option(  # noqa: B008
        False,
        help="If True, the vectors will be projected to the mean plane of the two points to isolate in-plane angles.",
    ),
    plot: bool = Option(  # noqa: B008
        False,
        help="If True, the geodesic distances will be plotted for each mesh.",
    ),
    exclude_edges: bool = Option(  # noqa: B008
        False,
        help="If True, the edges of the membrane will be excluded from the nearest neighbor calculation.",
    ),
    edge_exclusion_width: float = Option(  # noqa: B008
        50.0, help="Width of the edge exclusion zone in Anstrom."
    ),
    edge_percentile: float = Option(  # noqa: B008
        95, help="Percentile to use for edge exclusion."
    ),
    store_sanity_meshes: bool = Option(  # noqa: B008
        False, help="If True, the sanity meshes will be stored in a folder."
    ),
):
    """Compute geometric distances between nearest neighbors in all membrane meshes in a folder.

    Example
    -------
    membrain_stats geodesic_NN --in-folder <path-to-your-folder> --out-folder <path-to-store-meshes> --num-neighbors 1
    """

    from membrain_stats.geodesic_distances import (
        geodesic_nearest_neighbors_folder,
    )

    geodesic_nearest_neighbors_folder(
        in_folder=in_folder,
        out_folder=out_folder,
        pixel_size_multiplier=pixel_size_multiplier,
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
        num_neighbors=num_neighbors,
        start_classes=start_classes,
        target_classes=target_classes,
        method=method,
        c2_symmetry=c2_symmetry,
        project_to_plane=project_to_plane,
        plot=plot,
        exclude_edges=exclude_edges,
        edge_exclusion_width=edge_exclusion_width,
        edge_percentile=edge_percentile,
        store_sanity_meshes=store_sanity_meshes,
    )


@cli.command(name="geodesic_NN_wrt", no_args_is_help=True)
def geodesic_NN_wrt(
    in_folder: str = Option(  # noqa: B008
        ...,
        help="Path to the directory containing either .h5 files or .obj and .star files",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./stats/geodesic_distances",
        help="Path to the folder where computed stats should be stored.",
    ),
    exclude_edges: bool = Option(  # noqa: B008
        False,
        help="If True, the edges of the membrane will be excluded from the area calculation.",
    ),
    edge_exclusion_width: float = Option(  # noqa: B008
        50.0, help="Width of the edge exclusion zone in Anstrom."
    ),
    pixel_size_multiplier: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.",
    ),
    pixel_size_multiplier_positions: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, loaded positions are multiplied by this value.",
    ),
    num_neighbors: int = Option(  # noqa: B008
        1, help="Number of nearest neighbors to consider."
    ),
    start_classes: List[int] = Option(  # noqa: B008
        [0], help="List of classes to consider for start points."
    ),
    target_classes: List[int] = Option(  # noqa: B008
        [0], help="List of classes to consider for target points."
    ),
    with_respect_to_class: int = Option(  # noqa: B008
        0, help="Class with respect to which protein concentration should be computed."
    ),
    num_bins: int = Option(  # noqa: B008
        25, help="Number of bins to use for the histogram."
    ),
    geod_distance_method: str = Option(  # noqa: B008
        "fast",
        help="Method to use for computing geodesic distances. Can be either 'exact' or 'fast'.",
    ),
    distance_matrix_method: str = Option(  # noqa: B008
        "geodesic",
        help="Method to use for computing the distance matrix. Can be either 'geodesic' or 'euclidean'.",
    ),
):
    """Compute geodesic distances between nearest neighbors with respect to a specific class.

    Example
    -------
    membrain_stats geodesic_NN_wrt --in-folder <path-to-your-folder> --out-folder <path-to-store-meshes> --num-neighbors 1 --num-bins 25 --with-respect-to-class 1
    """

    from membrain_stats.geodesic_distances import (
        geodesic_nearest_neighbors_wrt_folder,
    )

    geodesic_nearest_neighbors_wrt_folder(
        in_folder=in_folder,
        out_folder=out_folder,
        exclude_edges=exclude_edges,
        edge_exclusion_width=edge_exclusion_width,
        pixel_size_multiplier=pixel_size_multiplier,
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
        num_neighbors=num_neighbors,
        start_classes=start_classes,
        target_classes=target_classes,
        geod_distance_method=geod_distance_method,
        distance_matrix_method=distance_matrix_method,
        with_respect_to_class=with_respect_to_class,
        num_bins=num_bins,
    )


@cli.command(name="geodesic_ripley", no_args_is_help=True)
def geodesic_ripley(
    in_folder: str = Option(  # noqa: B008
        ...,
        help="Path to the directory containing either .h5 files or .obj and .star files",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./stats/geodesic_ripley",
        help="Path to the folder where computed stats should be stored.",
    ),
    pixel_size_multiplier: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value.",
    ),
    pixel_size_multiplier_positions: float = Option(  # noqa: B008
        None,
        help="Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, loaded positions are multiplied by this value.",
    ),
    start_classes: List[int] = Option(  # noqa: B008
        [0], help="List of classes to consider for start points."
    ),
    target_classes: List[int] = Option(  # noqa: B008
        [0], help="List of classes to consider for target points."
    ),
    ripley_type: str = Option(  # noqa: B008
        "O",
        help="Which type of Ripley statistic should be computed? Choose between O, L, and K",
    ),
    num_bins: int = Option(  # noqa: B008
        50, help="Into how many bins should the ripley statistics be split?"
    ),
    method: str = Option(  # noqa: B008
        "fast",
        help="Method to use for computing geodesic distances. Can be either 'exact' or 'fast'.",
    ),
    exclude_edges: bool = Option(  # noqa: B008
        False,
        help="If True, the edges of the membrane will be excluded from the area calculation.",
    ),
    edge_exclusion_width: float = Option(  # noqa: B008
        50.0, help="Width of the edge exclusion zone in Anstrom."
    ),
    edge_percentile: float = Option(  # noqa: B008
        95, help="Percentile to use for edge exclusion."
    ),
    plot: bool = Option(  # noqa: B008
        False,
        help="If True, the Ripley statistics will be plotted for each mesh.",
    ),
):
    """Compute Ripley statistics for all membrane meshes in a folder.

    Example
    -------
    membrain_stats geodesic_ripley --in-folder <path-to-your-folder> --out-folder <path-to-store-meshes> --ripley-type L --num-bins 50
    """
    from membrain_stats.geodesic_distances import (
        geodesic_ripleys_folder,
    )

    geodesic_ripleys_folder(
        in_folder=in_folder,
        out_folder=out_folder,
        pixel_size_multiplier=pixel_size_multiplier,
        pixel_size_multiplier_positions=pixel_size_multiplier_positions,
        start_classes=start_classes,
        target_classes=target_classes,
        ripley_type=ripley_type,
        num_bins=num_bins,
        method=method,
        exclude_edges=exclude_edges,
        edge_exclusion_width=edge_exclusion_width,
        edge_percentile=edge_percentile,
        plot=plot,
    )
