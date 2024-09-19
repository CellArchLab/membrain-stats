# Nearest Neighbors Calculation

Goal: Compute the nearest neighbors distances of a given point class in the membrane.

Currently, we support to compute the general nearest neighbor distances, as well as the nearest neighbor distances with respect to the distance to another point class (e.g. membrane edge; needs to be manually defined).

## General Nearest Neighbors

This function computes the nearest neighbor distances of a given point class in the membrane:
Formula:
$$
\text{Nearest Neighbors Distance} = \frac{\text{Sum of Nearest Neighbors Distances}}{\text{Number of Nearest Neighbors}}
$$

It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats nearest_neighbors --in-folder <path/to/folder> --start-classes <list_of_classes> --target-classes <list_of_classes>
```

Other optional arguments:
- `--out-folder`: Path to the output folder. Default: `./stats/nearest_neighbors`
- `--pixel-size-multiplier`: Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value. Default: `None`
- `--num-neighbors`: Number of nearest neighbors to consider. Default: `1`
- `--start-classes`: List of classes to consider for start points. Default: `0`
- `--target-classes`: List of classes to consider for target points. Default: `0`
- `--method`: Method to use for computing geodesic distances. Can be either 'exact' or 'fast'. Default: `fast`
- `--c2-symmetry`: If True, the C2 symmetry of the protein will be considered when computing the angles. Default: `False`
- `--project-to-plane`: If True, the vectors will be projected to the mean plane of the two points to isolate in-plane angles. Default: `False`


### Nearest neighbor orientations
If your `in-folder` contains `.star` files with `rlnAngleRot`, `rlnAngleTilt`, and `rlnAnglePsi` columns, the orientations of the nearest neighbors can be computed as well.
It will store the angles in the per-membrane output star files for each nearest neighbor.

Corresponding command line arguments:
- `--c2-symmetry`: If True, the C2 symmetry of the protein will be considered when computing the angles. Default: `False`
- `--project-to-plane`: If True, the vectors will be projected to the mean plane of the two points to isolate in-plane angles. Default: `False`


## Nearest Neighbors Distance with respect to another point class

This function computes the nearest neighbor distances with respect to the distance to another point class (e.g. membrane edge; needs to be manually defined).
Hereby, the distances to the other class are binned and the average nearest neighbor distances are computed for each bin.

Formula for bins b=1,...,B:
$$
\text{Nearest Neighbors Distance}_b = \frac{\text{Sum of Nearest Neighbors Distances in Bin b}}{\text{Number of Nearest Neighbors in Bin b}}
$$
where bin b is defined by the distance to the other point class.

It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats nearest_neighbors_wrt --in-folder <path/to/folder> --start-classes <list_of_classes> --target-classes <list_of_classes> --with-respect-to-class <class>
```
This will compute the nearest neighbor distances with respect to the class `<class>` for the classes in `<list_of_classes>`.

Other optional arguments:
- `--out-folder`: Path to the output folder. Default: `./stats/nearest_neighbors`
- `--exclude-edges`: If True, the edges of the membrane will be excluded from the area calculation. Default: `False`
- `--edge-exclusion-width`: Width of the edge exclusion zone in Angstrom. Default: `50.0`
- `--pixel-size-multiplier`: Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value. Default: `None`
- `--num-neighbors`: Number of nearest neighbors to consider. Default: `1`
- `--start-classes`: List of classes to consider for start points. Default: `0`
- `--target-classes`: List of classes to consider for target points. Default: `0`
- `--with-respect-to-class`: Class with respect to which protein concentration should be computed. Default: `0`
- `--num-bins`: Number of bins to use for the histogram. Default: `25`
- `--geod-distance-method`: Method to use for computing geodesic distances. Can be either 'exact' or 'fast'. Default: `fast`
- `--distance-matrix-method`: Method to use for computing the distance matrix. Can be either 'geodesic' or 'euclidean'. Default: `geodesic`


<!-- │ *  --in-folder                                       TEXT     Path to the directory containing either .h5 files or .obj and .star files [default: None] [required]                                                                                                                                                                                               │
│    --out-folder                                      TEXT     Path to the folder where computed stats should be stored. [default: ./stats/geodesic_distances]                                                                                                                                                                                                    │
│    --exclude-edges             --no-exclude-edges             If True, the edges of the membrane will be excluded from the area calculation. [default: no-exclude-edges]                                                                                                                                                                                         │
│    --edge-exclusion-width                            FLOAT    Width of the edge exclusion zone in Anstrom. [default: 50.0]                                                                                                                                                                                                                                       │
│    --pixel-size-multiplier                           FLOAT    Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value. [default: None]                                                                                                                                                             │
│    --num-neighbors                                   INTEGER  Number of nearest neighbors to consider. [default: 1]                                                                                                                                                                                                                                              │
│    --start-classes                                   INTEGER  List of classes to consider for start points. [default: 0]                                                                                                                                                                                                                                         │
│    --target-classes                                  INTEGER  List of classes to consider for target points. [default: 0]                                                                                                                                                                                                                                        │
│    --with-respect-to-class                           INTEGER  Class with respect to which protein concentration should be computed. [default: 0]                                                                                                                                                                                                                 │
│    --num-bins                                        INTEGER  Number of bins to use for the histogram. [default: 25]                                                                                                                                                                                                                                             │
│    --geod-distance-method                            TEXT     Method to use for computing geodesic distances. Can be either 'exact' or 'fast'. [default: fast]                                                                                                                                                                                                   │
│    --distance-matrix-method                          TEXT     Method to use for computing the distance matrix. Can be either 'geodesic' or 'euclidean'. [default: geodesic]                                                                                                                                                                                      │
│    --help                                                     Show this message and exit.       -->

