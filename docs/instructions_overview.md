 # MemBrain-stats Instructions

MemBrain-stats is a Python project developed by the [CellArchLab](https://www.cellarchlab.com/) for computing membrane protein statistics in 3D for cryo-electron tomography (cryo-ET).

## Data Preparation
As a first step, you need to prepare the data. The most important ingredients for MemBrain-stats are protein locations and membrane meshes. More information on how to prepare the data can be found [here](data_preparation.md).

## Functionalities
All functionalities can be accessed via the command line interface (CLI). To get an overview of all functionalities, run:
```bash
membrain_stats
```

### Protein Concentration
The command `protein_concentration` computes the number of proteins per membrane area. It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats protein_concentration --in-folder <path/to/folder>
```
More information can be found [here](protein_concentrations.md).

### (geodesic) Nearest Neighbors
The command `geodesic_NN` computes the geometric distances between nearest neighbors in the membrane. It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats geodesic_NN --in-folder <path/to/folder> --start-classes <list_of_classes> --target-classes <list_of_classes>
```
More information can be found [here](nearest_neighbors.md).
### (geodesic) Ripley's Statistics
The command `geodesic_ripley` computes Ripley's statistics for all membrane meshes in a folder. It can be accessed via the command line interface (CLI) by running:
```bash
membrain_stats geodesic_ripley --in-folder <path/to/folder> --start-classes <list_of_classes> --target-classes <list_of_classes>
```
More information can be found [here](ripley_statistics.md).

### Edge exclusion
The edge exclusion functionality can be used to exclude the edges of the membrane from the analysis. It can be accessed as an optional argument to many of the functionalities. More information can be found [here](edge_exclusion.md).


