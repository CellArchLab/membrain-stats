# Geodesic Ripleys Functions

Goal: Compute the geodesic Ripley's functions of a given point class in the membrane.

We support to compute Ripley's K, L and O functions, similarly defined as in [1].

```
[1] Martinez-Sanchez, Antonio, Wolfgang Baumeister, and Vladan Lučić. "Statistical spatial analysis for cryo-electron tomography." Computer methods and programs in biomedicine 218 (2022): 106693.
```

## How to access the functions

You can access all three functions via the command line interface (CLI) by running:

```bash
membrain_stats geodesic_ripley --in-folder <path/to/folder> --start-classes <list_of_classes> --target-classes <list_of_classes> --ripley-type <type>
```

where `<type>` can be either `O`, `L`, or `K`.

### Optional arguments
- `--out-folder`: Path to the output folder. Default: `./stats/geodesic_ripley`
- `--pixel-size-multiplier`: Pixel size multiplier if mesh is not scaled in unit Angstrom. If provided, mesh vertices are multiplied by this value. Default: `None`
- `--start-classes`: List of classes to consider for start points. Default: `0`
- `--target-classes`: List of classes to consider for target points. Default: `0`
- `--ripley-type`: Which type of Ripley statistic should be computed? Choose between `O`, `L`, and `K`. Default: `O`
- `--num-bins`: Into how many bins should the ripley statistics be split? Default: `50`
- `--method`: Method to use for computing geodesic distances. Can be either `exact` or `fast`. Default: `fast`
- `--exclude-edges`: If True, the edges of the membrane will be excluded from the area calculation. Default: `no-exclude-edges`
- `--edge-exclusion-width`: Width of the edge exclusion zone in Anstrom. Default: `50.0`


### Ripley's K Function
Formula:
$$
K(r) = \frac{\sum_{i=0}^n\mathcal{C}(x, \mathcal{S}_L (x_i, r))}{\lambda \cdot \sum_{i=0}^n\mathcal{V}(\mathcal{S}_L (x_i, r))}
$$
where 
- $\mathcal{S}_L (x_i, r)$ is the geodesic "circle" of radius $r$ centered at $x_i$, i.e. all points on the mesh that are at most $r$ geodesic distance away from $x_i$
- $\mathcal{C}(x, \mathcal{S}_L (x_i, r))$ is the number of points in $\mathcal{S}_L (x_i, r)$
- $\mathcal{V}(\mathcal{S}_L (x_i, r))$ is the area of $\mathcal{S}_L (x_i, r)$
- $\lambda$ is the global concentration of the points on the mesh

### Ripley's L Function
Formula:
$$
L(r) = \sqrt{K(r)} - r
$$

### Ripley's O Function

Formula:
$$
O(r) = \frac{\sum_{i=0}^n\mathcal{C}(x, \mathcal{S}_O (x_i, r, \Delta r))}{\sum_{i=0}^n\mathcal{V}(\mathcal{S}_O (x_i, r, \Delta r))}
$$
where
- $\mathcal{S}_O (x_i, r, \Delta r)$ is the geodesic "annulus" of inner radius $r$ and outer radius $r + \Delta r$ centered at $x_i$, i.e. all points on the mesh that are at most $r + \Delta r$ geodesic distance away from $x_i$ and at least $r$ geodesic distance away from $x_i$
- Other variables are defined as in Ripley's K function



### What's happening in the background?
1. Preprocess meshes: Optionally exclude edges from the meshes
2. Compute distance matrices between start and target classes <br> Note: This is only computed within each connected component, where geodesic distances make sense.
3. Compute distances from each start point to each vertex of the mesh
4. Compute all barycentric areas for all vertices
5. Flatten protein-protein distance matrix and divide into bins
6. For each bin, compute the number of protein pairs within the bin <br> This is computed by summing the number of protein pairs within the bin and dividing by the area of the bin.
7. Compute the Ripley's function for each bin