import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
import ot
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class ot_annotator:
    def __init__(self, adata, adata_ref, gene_interest,param_space,key_ref='leiden',
                 key_tar='leiden'):
        """
        Initialize the OptimalTransportAnnotator.

        Parameters:
        adata: AnnData object (target)
        adata_ref: AnnData object (reference)
        gene_interest: List of gene names to be used
        """
        self.adata = adata
        self.adata_ref = adata_ref
        self.gene_interest = gene_interest
        self.T=None
        self.atlas=None
        self.trials = Trials()  # For storing hyperopt trials

        self.key_ref=key_ref
        self.key_tar=key_tar
        self.central_cells_ref=None
        self.central_cells_tar=None
        self.param_space=param_space
        self.X_reconstructed=None
        # Preprocess the data
        self.preprocess_adata()

    def preprocess_adata(self):
        """Preprocess both reference and target AnnData objects."""
        sc.pp.filter_cells(self.adata_ref, min_genes=1)
        sc.pp.filter_cells(self.adata, min_genes=1)

        sc.pp.normalize_total(self.adata_ref, layer='counts')
        sc.pp.log1p(self.adata_ref, layer='counts')

        sc.pp.normalize_total(self.adata, layer='counts')
        sc.pp.log1p(self.adata, layer='counts')

    def extract_central_cells_leiden(self, adata, leiden_key, n_clusters=1.0, min_cluster_size=1):
        """
        Extract central cells from an AnnData object using Leiden subclustering.

        Parameters:
        adata: AnnData object from which to extract central cells.
        leiden_key: The key in the AnnData object for the original Leiden clustering.
        n_clusters: Resolution parameter for the Leiden algorithm to control the number of subclusters.
        min_cluster_size: Minimum size of subclusters to consider.

        Returns:
        central_cells: List of central cells.
        cell_membership: Array indicating membership of each cell.
        """
        # Initialize list to store central cells
        central_cells = []

        # Initialize cell_membership array with None
        cell_membership = np.full(adata.n_obs, None, dtype=object)

        # Iterate over the original Leiden clusters
        for cell_type in adata.obs[leiden_key].unique():
            # Get only the cells from the current original Leiden cluster
            cell_type_data = adata[adata.obs[leiden_key] == cell_type].copy()

            # Perform subclustering with the Leiden algorithm if the cluster is large enough
            if cell_type_data.shape[0] > min_cluster_size:
                # Compute the neighborhood graph
                sc.pp.neighbors(cell_type_data)
                sc.tl.leiden(cell_type_data, resolution=n_clusters, key_added='sub_leiden')

                # Iterate over the subclusters created by the Leiden algorithm
                for subcluster in cell_type_data.obs['sub_leiden'].unique():
                    subcluster_data = cell_type_data[cell_type_data.obs['sub_leiden'] == subcluster]

                    # Process only subclusters that are larger than the minimum size
                    if subcluster_data.shape[0] >= min_cluster_size:
                        X = subcluster_data[:, self.gene_interest].X

                        # Convert sparse matrix to dense, if necessary
                        if not isinstance(X, np.ndarray):
                            X = X.toarray()

                        # Calculate the mean (subcluster center) for this subcluster
                        cluster_center = np.mean(X, axis=0)

                        # Calculate the Euclidean distance of each cell from the subcluster center
                        distances = np.linalg.norm(X - cluster_center, axis=1)

                        # Find the index of the cell closest to the subcluster center
                        central_cell_index = np.argmin(distances)
                        central_cell = subcluster_data.obs_names[central_cell_index]

                        # Append the central cell to the list
                        central_cells.append(central_cell)
                        cluster_indices = adata.obs_names.isin(subcluster_data.obs_names)
                        # Store the central cell in the cell_membership array
                        cell_membership[cluster_indices] = central_cell

        return central_cells, cell_membership


    def extract_central_cells_kmeans(self, adata, leiden_key, n_clusters, min_cluster_size=2):
        """
        Extract central cells from an AnnData object using KMeans clustering.

        Parameters:
        adata: AnnData object from which to extract central cells.
        leiden_key: The key in the AnnData object for cluster identification.
        n_clusters: Number of clusters for KMeans.
        min_cluster_size: Minimum size of clusters to consider.

        Returns:
        central_cells: List of central cells.
        cell_membership: Array indicating membership of each cell.
        """
        # Create a cell_membership array to store membership information
        cell_membership = np.full(adata.n_obs, None, dtype=object)
        central_cells = []

        # Use defaultdict for combined clusters to avoid manual dict handling
#         combined_clusters = {}

        # Iterate over cell types (Leiden clusters) in the AnnData object
        for cell_type in adata.obs[leiden_key].unique():
            # Get indices of cells in the current cluster
            cell_type_indices = adata.obs[adata.obs[leiden_key] == cell_type].index
            cell_type_data = adata[cell_type_indices, :]

            X = cell_type_data[:, self.gene_interest].X
            if not isinstance(X, np.ndarray):
                X = X.toarray()  # Convert to dense only if needed

            # Check if there are enough cells for KMeans clustering
            if len(cell_type_data) > n_clusters * 1.2:
                # Perform KMeans clustering on the subset of cells
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                kmeans_labels = kmeans.labels_

                # Assign KMeans labels to the corresponding cells
                adata.obs.loc[cell_type_indices, 'celltype_kmeans'] = [f"{cell_type}**{label}" for label in kmeans_labels]
            else:
                # If the cluster is small, assign the original Leiden label
                adata.obs.loc[cell_type_indices, 'celltype_kmeans'] = adata.obs.loc[cell_type_indices, leiden_key]

            # Group cells based on their KMeans or original labels and process them in the same loop
            for label, group in adata.obs.loc[cell_type_indices].groupby('celltype_kmeans'):
                group_cells = group.index.tolist()

                # Only process clusters larger than the minimum size
                if len(group_cells) >= min_cluster_size:
#                     if label not in combined_clusters:
#                         combined_clusters[label] = []  # Initialize if not exists
#                     combined_clusters[label]=group_cells

                    # Compute the cluster center and central cell in the same loop
                    cluster_indices = adata.obs_names.isin(group_cells)
                    X_cluster = adata[cluster_indices, self.gene_interest].X
                    if not isinstance(X_cluster, np.ndarray):
                        X_cluster = X_cluster.toarray()

                    # Calculate the center of the cluster
                    cluster_center = np.mean(X_cluster, axis=0)
#                     print(cluster_center.shape)

                    # Compute distances and find the closest cell to the center
                    distances = np.linalg.norm(X_cluster - cluster_center, axis=1)
                    central_cell_index = np.argmin(distances)
                    central_cell = adata.obs_names[cluster_indices][central_cell_index]
                    central_cells.append(central_cell)

                    # Assign central cell as membership for each cell in the group
                    cell_membership[cluster_indices] = central_cell

        return central_cells, cell_membership

    def compute_corr_cos2(self, X_org, X_pred):
        """
        Compute Pearson correlation between two matrices.
        
        Parameters:
        X_org: Original data matrix.
        X_pred: Predicted data matrix.

        Returns:
        corr_coefficient: Correlation coefficient.
        """
        X_org_flat = X_org.values.flatten()
        X_pred_flat = X_pred.values.flatten()

        corr_matrix = np.corrcoef(X_org_flat, X_pred_flat)
        corr_coefficient = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
        if corr_coefficient == 1:
            corr_coefficient=0
        return corr_coefficient

    def objective_fun(self, params, cost_matrix, q, p, aggregated_ref):
        """
        Objective function for hyperparameter optimization using unbalanced Sinkhorn transport.
        
        Parameters:
        params: Hyperparameters for the objective function.
        cost_matrix: Cost matrix for optimal transport.
        q: Distribution for target.
        p: Distribution for source.
        aggregated_ref: Reference aggregated data.
        atlas: Target atlas data.

        Returns:
        Dictionary containing loss and model.
        """
        corr = 0
        Ts, log = ot.unbalanced.sinkhorn_unbalanced(
            q, p, cost_matrix.T, params['reg'],
            [params['reg_m_kl_1'], params['reg_m_kl_2']],
            method=params['method'], reg_type=params['reg_type'], log=True
        )

        T = pd.DataFrame(Ts, index=self.atlas.obs.index, columns=aggregated_ref.obs.index)
        X_pred = pd.DataFrame(aggregated_ref.X.T.dot(T.T).T)
        idx = np.where(np.isnan(X_pred[0]))[0]
        if len(idx) < X_pred.shape[0] - 1:
            X_pred.columns = aggregated_ref.var_names
            X_pred = X_pred.drop(idx).loc[:, self.gene_interest]
            X_org = pd.DataFrame(self.atlas.X.toarray(),columns=self.atlas.var_names
                                ).drop(idx).loc[:, X_pred.columns]

            corr = self.compute_corr_cos2(X_org, X_pred)

        return {'loss': -corr, 'model': T, 'status': STATUS_OK, 'params': params}

    def run_optimal_transport(self, aggregated_ref, op_iter, meteric):
        """
        Main function to run optimal transport and hyperparameter tuning.
        
        Parameters:
        aggregated_ref: Aggregated reference data.
        aggregated_tar: Aggregated target data.

        Returns:
        best_T: Best transport matrix found.
        """
        source_matrix = aggregated_ref[:, self.gene_interest].X.toarray()
        target_matrix = self.atlas.X.toarray()

        p = ot.unif(source_matrix.shape[0])
        q = ot.unif(target_matrix.shape[0])

        cost_matrix = ot.dist(source_matrix, target_matrix, metric=meteric)



        best_params = fmin(
            fn=lambda params: self.objective_fun(params, cost_matrix, q, p, aggregated_ref),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=op_iter,
            trials=self.trials
        )

        self.T = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['model']

    def plot_and_refine_annotations(self,refinement_iter=10,nb_nghb=5):
        """
        Plot the optimal transport matrix and refine annotations.
        """

        # Create annotation DataFrame
        ann = pd.DataFrame(self.atlas.obs[self.key_tar])
        ann['predicted_celltype'] = [self.T.columns[i].split('**')[1] for i in np.argmax(self.T.values, axis=1)]

        # Step 1: Calculate maximum values and define threshold
        max_values_per_row = np.max(self.T.values, axis=1)  # Ensure .values is used for numpy operations
        threshold = np.percentile(max_values_per_row, 25)

        # Step 2: Create a mask for rows exceeding the threshold
        exceeds_threshold = max_values_per_row > threshold

        # Step 3: Identify selected cells based on membership_cell condition
        selected_central_cells = self.atlas.obs_names[exceeds_threshold]


        # Create contingency matrix
        co_mat = pd.crosstab(ann[self.key_tar], ann['predicted_celltype'])

        # Plot the contingency matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(co_mat, annot=True, fmt="d", cmap='Blues', cbar=True)
        plt.title('Contingency Matrix of Annotations of central cells')
        plt.xlabel('Predicted Annotations')
        plt.ylabel(self.key_tar)
        plt.show()

        # Update self.adata with predicted annotations
        self.atlas.obs['predicted_annotation'] = ann['predicted_celltype']
        sc.pp.pca(self.atlas)
        sc.pp.neighbors(self.atlas, nb_nghb)

        # Refine annotations
        self.atlas.obs['refined_annotation'] = self.refine_annotations_until_convergence(
            max_iters=refinement_iter, tolerance=1e-3)
        
        central_cells = self.adata.obs['central_cell_membership'].values

        # Reindex atlas.obs to match central_cells, handling missing indices by keeping the index name
        mapped_celltypes = self.atlas.obs['refined_annotation'].reindex(central_cells)

        # Replace missing cell types with the corresponding cell name
        mapped_celltypes.fillna(pd.Series(central_cells, index=mapped_celltypes.index), inplace=True)

        # Assign the mapped cell types to adata.obs
        self.adata.obs['refined_annotation'] = mapped_celltypes.values

        mapped_celltypes = self.atlas.obs['predicted_annotation'].astype('str').reindex(central_cells)

        # Replace missing cell types with the corresponding cell name
        mapped_celltypes.fillna(pd.Series(central_cells, index=mapped_celltypes.index), inplace=True)

        # Assign the mapped cell types to adata.obs
        self.adata.obs['predicted_annotation'] = mapped_celltypes.values
        return selected_central_cells
        
    def refine_annotations_until_convergence(self, max_iters, tolerance):
        """
        Refine annotations until convergence.

        Parameters:
        adata: AnnData object.
        max_iters: Maximum number of iterations.
        tolerance: Tolerance for convergence.

        Returns:
        adata: Updated AnnData object with refined annotations.
        """

        # Ensure the initial annotations are properly converted to categorical
        self.atlas.obs['predicted_annotation'] = self.atlas.obs['predicted_annotation'].astype('category')

        # Get the unique annotations from the initial labels (this will be fixed throughout)
        unique_annotations = self.atlas.obs['predicted_annotation'].cat.categories

        # Create a one-hot encoded matrix for annotations (aligned with the fixed unique annotations)
        annotation_matrix = pd.get_dummies(self.atlas.obs['predicted_annotation'], columns=unique_annotations).reindex(columns=unique_annotations, fill_value=0)

        # Ensure the annotation matrix is numeric (convert to float32)
        annotation_matrix = annotation_matrix.astype(np.float32).values

        # Initialize with the current annotations
        refined_annotation_matrix = annotation_matrix.copy()

        # Check if the connectivities matrix is in the correct format
        connectivities = self.atlas.obsp['connectivities']

        # Ensure the connectivities matrix is in float32 (if needed)
        if connectivities.dtype != np.float32:
            connectivities = connectivities.astype(np.float32)

        # Perform iterative refinement until convergence
        for iteration in range(max_iters):
#             print(f"Iteration {iteration + 1}")

            # Perform matrix multiplication of the connectivities matrix with the refined annotation matrix
            neighbor_annotation_sums = connectivities.dot(refined_annotation_matrix)

            # Find the index of the maximum sum (i.e., the most frequent annotation among neighbors) for each cell
            refined_annotation_indices = np.argmax(neighbor_annotation_sums, axis=1)

            # Map indices back to the original annotation labels (this aligns with the fixed unique annotations)
            refined_annotations = unique_annotations[refined_annotation_indices]

            # Convert refined annotations to one-hot encoded matrix, ensuring alignment with the original set of unique annotations
            new_annotation_matrix = pd.get_dummies(refined_annotations, columns=unique_annotations).reindex(columns=unique_annotations, fill_value=0)

            # Ensure the new annotation matrix is numeric
            new_annotation_matrix = new_annotation_matrix.astype(np.float32).values

            # Calculate the fraction of cells whose annotations changed (element-wise comparison)
            changes = np.mean(refined_annotation_matrix.argmax(axis=1) != new_annotation_matrix.argmax(axis=1))
#             print(f"Fraction of changed annotations: {changes}")

            # Update the annotation matrix
            refined_annotation_matrix = new_annotation_matrix

            # If the change is smaller than the tolerance, stop the iteration
            if changes < tolerance:
                print(f"Converged after {iteration + 1} iterations.")
                break

        return refined_annotations
    def subcluster(self,nb_cluster=2, clus_meth='kmeans'):
        if clus_meth =='kmeans':
            """Main function to annotate the data using central cells."""
            self.central_cells_ref, ref_membership = self.extract_central_cells_kmeans(
                self.adata_ref, leiden_key=self.key_ref, n_clusters=nb_cluster)
            self.central_cells_tar, tar_membership = self.extract_central_cells_kmeans(
                self.adata, leiden_key=self.key_tar, n_clusters=nb_cluster)
        if clus_meth =='leiden':
            """Main function to annotate the data using central cells."""
            self.central_cells_ref, ref_membership = self.extract_central_cells_leiden(
                self.adata_ref, leiden_key=self.key_ref, n_clusters=nb_cluster)
            self.central_cells_tar, tar_membership = self.extract_central_cells_leiden(
                self.adata, leiden_key=self.key_tar, n_clusters=nb_cluster)

        self.adata_ref.obs['central_cell_membership'] = ref_membership
        self.adata.obs['central_cell_membership'] = tar_membership
    
    def annotate(self,op_iter, meteric):

        aggregated_ref = self.adata_ref[self.adata_ref.obs_names.isin(self.central_cells_ref)]

        aggregated_ref = aggregated_ref[aggregated_ref[:, self.gene_interest].X.sum(axis=1) > 0]
        print(f"shape of reference central cells matrix: {aggregated_ref.shape}")

        aggregated_tar = self.adata[self.adata.obs_names.isin(self.central_cells_tar)]
        aggregated_tar = aggregated_tar[aggregated_tar[:, self.gene_interest].X.sum(axis=1) > 0]
        self.atlas = aggregated_tar[:, self.gene_interest]
        print(f"shape of target central cells matrix: {self.atlas.shape}")

        self.run_optimal_transport(aggregated_ref,op_iter, meteric)
        pd.DataFrame(self.T)  # Use the transport matrix stored in the AnnData object
        self.T.columns = aggregated_ref.obs_names+'**'+aggregated_ref.obs[self.key_ref].astype(str)
        self.T.index = aggregated_tar.obs.index
        self.X_reconstructed = pd.DataFrame(aggregated_ref.X.T.dot(self.T.T).T, columns=aggregated_ref.var_names)

        