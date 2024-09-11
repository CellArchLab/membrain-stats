# Protein Concentrations

Goal: Compute the protein concentration in a given region of interest (ROI) in the membrane.

Currently, we support to compute the general protein concentration, as well as the protein concentration with respect to the distance to another point class (e.g. membrane edge; needs to be manually defined).

## General Protein Concentration
This function computes the number of proteins per membrane area:
Formula:
$$
\text{Protein Concentration} = \frac{\text{Number of Proteins}}{\text{Membrane Area}}
$$

It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats protein_concentration --in-folder <path/to/folder>
```

Other optional arguments:
- `--out-folder`: Path to the output folder. Default: `./stats/protein_concentration`
- `--pixel-size-multiplier`: Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value. Default: `None`
- `--only-one-side`: If True, only one side of the membrane will be considered for area calculation. Default: `False`
- `--exclude-edges`: If True, the edges of the membrane will be excluded from the area calculation. Default: `False`
- `--edge-exclusion-width`: Width of the edge exclusion zone in Angstrom. Default: `50.0`



## Protein Concentration with respect to another point class
This function computes the protein concentration with respect to the distance to another point class (e.g. membrane edge; needs to be manually defined).
Hereby, the distances to the other class are binned and the protein concentration is computed for each bin.

Formula for bins b=1,...,B:
$$
\text{Protein Concentration}_b = \frac{\text{Number of Proteins in Bin b}}{\text{Membrane Area in Bin b}}
$$
where bin b is defined by the distance to the other point class.

It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats protein_concentration_wrt --in-folder <path/to/folder> --consider-classes <list_of_classes> --with-respect-to-class <class>
```
This will compute the protein concentration with respect to the class `<class>` for the classes in `<list_of_classes>`.

Other optional arguments:
- `--out-folder`: Path to the output folder. Default: `./stats/protein_concentration`
- `--pixel-size-multiplier`: Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value. Default: `None`
- `--exclude-edges`: If True, the edges of the membrane will be excluded from the area calculation. Default: `False`
- `--edge-exclusion-width`: Width of the edge exclusion zone in Angstrom. Default: `50.0`
- `--num-bins`: Number of bins to use for the histogram. Default: `25`
- `--consider-classes`: List of classes to consider for protein concentration calculation. If set to -1, all classes will be considered. Default: `-1`
- `--with-respect-to-class`: Class with respect to which protein concentration should be computed. Default: `0`
- `--geod-distance-method`: Method to use for computing geodesic distances. Can be either 'exact' or 'fast'. Default: `exact`
- `--distance-matrix-method`: Method to use for computing the distance matrix. Can be either 'geodesic' or 'euclidean'. Default: `geodesic`


## Example
```bash
membrain_stats protein_concentration --in-folder ./data --out-folder ./stats/protein_concentration --pixel-size-multiplier 14.08 --only-one-side True --exclude-edges True --edge-exclusion-width 50.0
```

```bash
membrain_stats protein_concentration_wrt --in-folder ./data --out-folder ./stats/protein_concentration --pixel-size-multiplier 14.08 --exclude-edges True --edge-exclusion-width 50.0 --num-bins 25 --consider-classes 1 --consider-classes 2 --consider-classes 3 --with-respect-to-class 0 --geod-distance-method fast --distance-matrix-method euclidean
```