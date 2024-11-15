from typing import List
import os
import pandas as pd
import starfile
from membrain_stats.utils.io_utils import (
    get_mesh_filenames,
    get_mesh_from_file,
    get_geodesic_distance_input,
)
from membrain_stats.utils.ripley_utils import (
    compute_ripleys_stats,
    aggregate_ripleys_stats,
)

from membrain_stats.membrane_edges.edge_from_curvature import exclude_edges_from_mesh


def geodesic_ripleys_folder(
    in_folder: str,
    out_folder: str,
    pixel_size_multiplier: float = None,
    start_classes: List[int] = [0],
    target_classes: List[int] = [0],
    ripley_type: str = "O",
    num_bins: int = 100,
    method: str = "fast",
    exclude_edges: bool = False,
    edge_exclusion_width: float = 50.0,
    plot: bool = False,
    edge_percentile: float = 95,
):
    # get filenames from folder
    filenames = get_mesh_filenames(in_folder)

    # load mehes
    mesh_dicts = [
        get_mesh_from_file(filename, pixel_size_multiplier=pixel_size_multiplier)
        for filename in filenames
    ]

    # exclude edges
    if exclude_edges:
        mesh_dicts = [
            exclude_edges_from_mesh(
                out_folder=out_folder,
                filename=filename,
                mesh_dict=mesh_dict,
                edge_exclusion_width=edge_exclusion_width,
                percentile=edge_percentile,
                force_recompute=False,
            )
            for (filename, mesh_dict) in zip(filenames, mesh_dicts)
        ]

    # prepare input for computation of geodesic distances
    mesh_dicts = [
        get_geodesic_distance_input(mesh_dict, start_classes, target_classes)
        for mesh_dict in mesh_dicts
    ]

    # compute values necessary for ripley's statistics
    ripley_stats = [
        compute_ripleys_stats(mesh_dict, method=method) for mesh_dict in mesh_dicts
    ]

    # aggregate computed values to output global ripley's statistics
    ripley_stats = aggregate_ripleys_stats(
        ripley_stats=ripley_stats, ripley_type=ripley_type, num_bins=num_bins
    )

    # store in star file
    out_data = {
        "ripleyType": ripley_type,
        "x_values": ripley_stats[0],
        "y_values": ripley_stats[1],
    }
    out_data = pd.DataFrame(out_data)
    out_file = os.path.join(out_folder, f"ripleys{ripley_type}.star")
    os.makedirs(out_folder, exist_ok=True)
    starfile.write(out_data, out_file)

    if plot:
        from matplotlib import pyplot as plt

        # plot Ripley's statistics
        plt.plot(ripley_stats[0], ripley_stats[1])
        plt.xlabel("Distance (nm)")
        plt.ylabel(f"Ripley's {ripley_type}")
        plt.title(f"Ripley's {ripley_type}")
        plt.savefig(os.path.join(out_folder, f"ripleys{ripley_type}.png"))
