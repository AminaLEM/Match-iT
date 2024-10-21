# Match-iT
Unbalanced Mapping Approach for High-Resolution Spatial Transcriptomics
![Workflow](https://github.com/AminaLEM/Match-iT/blob/main/Algorithm.png)


# Overview
The notebook main_annotate.ipynb shows how to annotate High-Resolution Spatial Transcriptomics data with high accuracy using a reference dataset.
The following is the structure of the package:

- **`src/ot_annotator.py`**: Used for subclustering and OT mapping.
- **`src/plot_hp.py`**: Used to plot hyperparameters scatterplot.
- **`src/compare_viz.py`**: Used to evaluate results (comparison to other methods and visualization).
- **`main_annotate.ipynb`**: example ...

The following are the main objects (useful outputs) of the annotator class:
- **`annotator.T`**: The transport plan.
- **`annotator.X_reconstructed`**: the reconstructed matrix.
- **`annotator.adata`**: the updated ST target modality with annotator.adata.obs['predicted_annotation'] is the transferred annotation and annotator.adata.obs['central_cell_membership'] is the subclustering labeled with the central cell of the subcluster (the same is in annotator.adata_ref).
- **`selected_central_cells`**: the subclusters that are annotated with high probability based on T. The threshold is selected based on the elbow approach.


