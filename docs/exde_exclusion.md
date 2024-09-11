# Edge exclusion

Many of MemBrain-stats' functionalities enable the user to exclude the edges of the membrane from the analysis. This can be useful to avoid artifacts that might be introduced by the membrane edges. The edge exclusion can be enabled by setting the `--exclude-edges` flag to `True`. The width of the edge exclusion zone can be adjusted by setting the `--edge-exclusion-width` flag to the desired value in Angstrom.

Here, we want to briefly explain what's happening in the background when the edge exclusion is enabled.

The edge exlusion is based on extracting the membrane edges via high-curvature regions of the mesh. Once these are identified, the user-defined width is used to exclude the edges from the analysis. The width is defined in Angstrom (or respective other unit of the mesh) and is used to exclude a certain number of vertices from the edge of the membrane.

### Detailed steps:
1. Compute discrete mean curvature of the membrane mesh using trimesh's discrete_mean_curvature_measure function.
2. Identify high-curvature regions of the mesh by taking the top 5% of the curvature values.
3. Define edge vertices as those vertices that have at most `edge-exclusion-width` distance to the high-curvature regions.
4. Exclude the edge vertices from the analysis by masking them out of the mesh.