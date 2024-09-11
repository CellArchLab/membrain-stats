 # MemBrain-stats Instructions

MemBrain-stats is a Python project developed by the [CellArchLab](https://www.cellarchlab.com/) for computing membrane protein statistics in 3D for cryo-electron tomography (cryo-ET).

## Data Preparation
As a first step, you need to prepare the data. The most important ingredients for MemBrain-stats are protein locations and membrane meshes. More information on how to prepare the data can be found [here](data_preparation.md).

## Functionalities
All functionalities can be accessed via the command line interface (CLI). To get an overview of all functionalities, run:
```bash
membrain-stats
```

### Protein Concentration
The command `protein_concentration` computes the number of proteins per membrane area. It can be accessed via the command line interface (CLI) by running:
```bash
membrain-stats protein_concentration --in-folder <path/to/folder>
```
More information can be found [here](protein_concentrations.md).

### (geodesic) Nearest Neighbors
tbd.

### (geodesic) Ripley's Statistics
tbd.

### Edge exclusion
tbd.