import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
import ot
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
np.random.seed(42) 
from sklearn.cluster import DBSCAN


class OTAnnotator:
    def __init__(self, adata: sc.AnnData, adata_ref: sc.AnnData, gene_interest: List[str],
                 param_space: Dict[str, Any], key_ref: str = 'leiden', key_tar: str = 'leiden',
                 chunks: int = 1):
        """
        Initialize the OptimalTransportAnnotator.

        Parameters:
        - adata: AnnData object (target)
        - adata_ref: AnnData object (reference)
        - gene_interest: List of gene names to be used
        - param_space: Parameter space for hyperparameter optimization
        - key_ref: Key for reference cluster annotations
        - key_tar: Key for target cluster annotations
        - chunks: Number of chunks for processing large datasets
        """
        self.adata = adata
        self.adata_ref = adata_ref
        self.gene_interest = gene_interest
        self.param_space = param_space
        self.key_ref = key_ref
        self.key_tar = key_tar
        self.chunks = chunks

        self.trials = Trials()
        self.T = None
        self.X_reconstructed = pd.DataFrame()
#         self.aggregated_tar = None
        self.central_cells_ref = None
        self.central_cells_tar = None
        self.best_params=None

        self._preprocess_adata()

    def _preprocess_adata(self) -> None:
        """Preprocess both reference and target AnnData objects."""
        for data in [self.adata_ref, self.adata]:
            sc.pp.filter_cells(data, min_genes=1)
            sc.pp.normalize_total(data, layer='counts')
            sc.pp.log1p(data)

    def _process_cell_type_leiden(self, adata: sc.AnnData, leiden_key: str, cell_type: str,
                          n_clusters: float, min_cluster_size: int =3) -> List[Tuple[List[str], str]]:
        """
        Process a single cell type by subclustering and identifying central cells.

        Parameters:
        - adata: AnnData object
        - leiden_key: Key for Leiden clustering
        - cell_type: Cell type to process
        - n_clusters: Resolution for Leiden subclustering
        - min_cluster_size: Minimum cluster size for valid subclusters

        Returns:
        - List of tuples with cell group and central cell
        """
        cell_type_data = adata[adata.obs[leiden_key] == cell_type].copy()

        if cell_type_data.shape[0] > min_cluster_size:
            sc.pp.neighbors(cell_type_data)
            sc.tl.leiden(cell_type_data, resolution=n_clusters, key_added='sub_leiden')

            results = []
            for subcluster in cell_type_data.obs['sub_leiden'].unique():
                subcluster_data = cell_type_data[cell_type_data.obs['sub_leiden'] == subcluster]
                if subcluster_data.shape[0] >= min_cluster_size:
                    X = subcluster_data[:, self.gene_interest].X
                    X = X.toarray() if not isinstance(X, np.ndarray) else X

                    cluster_center = np.mean(X, axis=0)
                    distances = np.linalg.norm(X - cluster_center, axis=1)
                    central_cell_index = np.argmin(distances)
                    central_cell = subcluster_data.obs_names[central_cell_index]
                    results.append((subcluster_data.obs_names.tolist(), central_cell))
            return results
        return []

    def _extract_central_cells_leiden(self, adata: sc.AnnData, leiden_key: str, n_clusters: float,
                              min_cluster_size: int=3, max_workers: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
        """
        Extract central cells using Leiden subclustering with parallel processing.

        Parameters:
        - adata: AnnData object
        - leiden_key: Key for Leiden clustering
        - n_clusters: Resolution parameter for subclustering
        - min_cluster_size: Minimum cluster size for valid subclusters
        - max_workers: Maximum number of parallel workers

        Returns:
        - List of central cells
        - Cell membership array
        """
        central_cells = []
        cell_membership = np.full(adata.n_obs, None, dtype=object)


        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_cell_type_leiden, adata, leiden_key, cell_type, n_clusters, min_cluster_size): cell_type for cell_type in adata.obs[leiden_key].unique()}
            for future in as_completed(futures):
                cell_type = futures[future]
                try:
                    results = future.result()
                    for group, central_cell in results:
                        central_cells.append(central_cell)
                        cell_membership[adata.obs_names.isin(group)] = central_cell
                except Exception as e:
                    print(f"Error processing cell type {cell_type}: {e}")

        return central_cells, cell_membership

    def _compute_corr(self, X_org: pd.DataFrame, X_pred: pd.DataFrame) -> float:
        """
        Compute Pearson correlation between two matrices.

        Parameters:
        - X_org: Original data matrix
        - X_pred: Predicted data matrix

        Returns:
        - Correlation coefficient
        """
        X_org_flat = X_org.values.flatten()
        X_pred_flat = X_pred.values.flatten()

        corr_matrix = np.corrcoef(X_org_flat, X_pred_flat)
        corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
        if corr == 1:
            corr=0

        return corr

    def _objective_function(self, params: Dict[str, Any], cost_matrix: np.ndarray,
                           p: np.ndarray, q: np.ndarray, aggregated_ref: sc.AnnData) -> Dict[str, Any]:
        """
        Objective function for hyperparameter optimization using unbalanced Sinkhorn transport.

        Parameters:
        - params: Hyperparameters for optimization
        - cost_matrix: Cost matrix for optimal transport
        - p: Source distribution
        - q: Target distribution
        - aggregated_ref: Aggregated reference AnnData object

        Returns:
        - Loss and model
        """
        Ts, log = ot.unbalanced.sinkhorn_unbalanced(
            p, q, cost_matrix, params['reg'],
            [params['reg_m_kl_1'], params['reg_m_kl_2']],
            method=params['method'], reg_type=params['reg_type'], log=True
        )

        T = pd.DataFrame(Ts.T, index=self.atlas.obs_names, columns=aggregated_ref.obs_names)
        X_pred = pd.DataFrame(aggregated_ref.X.T.dot(T.T), index=aggregated_ref.var_names).T

        valid_idx = ~np.isnan(X_pred.iloc[:, 0])
        if valid_idx.sum() > 1:
            X_pred = X_pred.loc[valid_idx, self.gene_interest]
            X_org = pd.DataFrame(self.atlas.X.toarray(), columns=self.atlas.var_names).loc[valid_idx, self.gene_interest]
            corr = self._compute_corr(X_org, X_pred)
        else:
            corr = 0

        return {'loss': -corr, 'model': T, 'status': STATUS_OK, 'params': params}

    def _run_optimal_transport(self, aggregated_ref: sc.AnnData, op_iter: int, metric: str) -> None:
        """
        Run optimal transport and hyperparameter tuning.

        Parameters:
        - aggregated_ref: Aggregated reference AnnData object
        - op_iter: Number of optimization iterations
        - metric: Distance metric for cost matrix
        """
        source_matrix = aggregated_ref[:, self.gene_interest].X.toarray()
        target_matrix = self.atlas[:, self.gene_interest].X.toarray()
        cost_matrix = ot.dist(source_matrix, target_matrix, metric=metric)
        p, q = ot.unif(source_matrix.shape[0]), ot.unif(target_matrix.shape[0])

        self.best_params = fmin(
            fn=lambda params: self._objective_function(params, cost_matrix, p, q, aggregated_ref),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=op_iter,
            trials=self.trials
        )
        self.T = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['model']


    def _process_cell_type_mean(self, adata, leiden_key, cell_type, n_clusters, min_cluster_size):
        """
        Processes a specific cell type and performs subclustering using the Leiden algorithm.
        """
        cell_type_data = adata[adata.obs[leiden_key] == cell_type].copy()
        print(cell_type)
        if cell_type_data.shape[0] > min_cluster_size:
            sc.tl.pca(cell_type_data)  # Reduce dimensionality
            sc.pp.neighbors(cell_type_data)
            sc.tl.leiden(cell_type_data, resolution=n_clusters, key_added='sub_leiden')
            cell_type_data.obs['sub_leiden'] = (cell_type_data.obs[leiden_key].astype(str) +
                                               '_' + cell_type_data.obs['sub_leiden'].astype(str))
            return cell_type_data.obs['sub_leiden']
        else:
            return []
    
    def _extract_central_cells_leiden_mean_(self, adata, leiden_key, n_clusters=1.0, min_cluster_size=3, max_workers=None):
        """
        Extracts central cells from an AnnData object using Leiden subclustering with parallel processing.
        """
        cell_membership = np.full(adata.n_obs, None, dtype=object)
    
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_cell_type_mean, adata, leiden_key, cell_type, n_clusters, min_cluster_size): cell_type
                       for cell_type in adata.obs[leiden_key].unique()}
    
            for future in as_completed(futures):
                cell_type = futures[future]
                try:
                    subcluster_labels = future.result()
                    mask = adata.obs[leiden_key] == cell_type
                    cell_membership[mask] = subcluster_labels
                except Exception as e:
                    print(f"Error processing cell type '{cell_type}': {e}")
    
        return cell_membership




    def _extract_central_cells_leiden_mean(self, adata, leiden_key, n_clusters=2, min_cluster_size=5, max_workers=None):
        """
        Extracts central cells from an AnnData object using DBSCAN clustering and returns a combined
        membership variable of Leiden clustering and DBSCAN clustering.
        """
        # Create an empty array to hold cell membership

        # Iterate over each unique cell type in the Leiden clustering
        for cell_type in adata.obs[leiden_key].unique():
            
            # Select cells of the current cell type
            cell_type_indices = adata.obs[adata.obs[leiden_key] == cell_type].index
            cell_type_data = adata[cell_type_indices, :]

            # Extract the gene expression data for the current cell type
            X = cell_type_data.X
            X = X.toarray() if not isinstance(X, np.ndarray) else X

            # Check if there are enough cells for DBSCAN clustering
            if len(cell_type_data) > min_cluster_size * 1.2:
                n_clusters_adjusted = min(n_clusters, int(len(cell_type_data)/2))                
                kmeans = KMeans(n_clusters=n_clusters_adjusted, random_state=42)
                kmeans_labels = kmeans.fit_predict(X)

                # Store KMeans labels in 'celltype_kmeans' for the selected cell type
                adata.obs.loc[cell_type_indices, 'kmeans'] = kmeans_labels.astype(str)
        adata.obs['celltype_kmeans'] = (
                    adata.obs[ leiden_key].astype(str) + '_' + adata.obs[ 'kmeans']
                )

        # Return the combined membership variable
        return adata.obs['celltype_kmeans']
    def _extract_central_cells_kmeans(self, adata, leiden_key, n_clusters, min_cluster_size=2):
        """
        Extracts central cells from an AnnData object using KMeans clustering.
        """
        cell_membership = np.full(adata.n_obs, None, dtype=object)
        central_cells = []
    
        # Iterate over each unique cell type in the Leiden clustering
        for cell_type in adata.obs[leiden_key].unique():
            cell_type_indices = adata.obs[adata.obs[leiden_key] == cell_type].index
            cell_type_data = adata[cell_type_indices, :]
    
            X = cell_type_data[:, self.gene_interest].X
            X = X.toarray() if not isinstance(X, np.ndarray) else X
    
            # Check if there are enough cells for KMeans clustering
            if len(cell_type_data) > n_clusters * 1.2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                adata.obs.loc[cell_type_indices, 'celltype_kmeans'] = [f"{cell_type}**{label}" for label in kmeans.labels_]
            else:
                adata.obs.loc[cell_type_indices, 'celltype_kmeans'] = adata.obs.loc[cell_type_indices, leiden_key]
    
            # Group by the KMeans or original Leiden label
            for label, group in adata.obs.loc[cell_type_indices].groupby('celltype_kmeans'):
                group_cells = group.index.tolist()
    
                # Skip small groups
                if len(group_cells) < min_cluster_size:
                    continue
    
                cluster_indices = adata.obs_names.isin(group_cells)
                X_cluster = adata[cluster_indices, self.gene_interest].X
                X_cluster = X_cluster.toarray() if not isinstance(X_cluster, np.ndarray) else X_cluster
    
                # Calculate the center of the cluster
                cluster_center = np.mean(X_cluster, axis=0)
    
                # Compute distances and find the central cell
                distances = np.linalg.norm(X_cluster - cluster_center, axis=1)
                central_cell_index = np.argmin(distances)
                central_cell = adata.obs_names[cluster_indices][central_cell_index]
                central_cells.append(central_cell)
    
                # Assign the central cell as the membership for the group
                cell_membership[cluster_indices] = central_cell
    
        return central_cells, cell_membership


    def subcluster(self,nb_cluster=2, clus_meth='kmeans',max_workers=None):
        if clus_meth in self.adata_ref.obs.columns:
            ref_membership=self.adata_ref.obs[clus_meth]
        else:
            if self.chunks<2:
                sc.pp.neighbors(self.adata)
                sc.tl.leiden(self.adata, resolution=0.5, key_added=self.key_tar)
            if clus_meth =='kmeans':
                """Main function to annotate the data using central cells."""
                self.central_cells_ref, ref_membership = self._extract_central_cells_kmeans(
                    self.adata_ref, leiden_key=self.key_ref, n_clusters=nb_cluster)
                if self.chunks<2:
                    self.central_cells_tar, tar_membership = self._extract_central_cells_kmeans(
                        self.adata[:, self.gene_interest], leiden_key=self.key_tar, n_clusters=nb_cluster)
            if clus_meth =='leiden':
                """Main function to annotate the data using central cells."""
                self.central_cells_ref, ref_membership = self._extract_central_cells_leiden(
                    self.adata_ref, leiden_key=self.key_ref, n_clusters=nb_cluster,max_workers=max_workers)
                if self.chunks<2:
                    self.central_cells_tar, tar_membership = self._extract_central_cells_leiden(
                        self.adata[:, self.gene_interest], leiden_key=self.key_tar, n_clusters=nb_cluster,max_workers=max_workers)
            if clus_meth =='leiden_mean':
                """Main function to annotate the data using central cells."""
                ref_membership = self._extract_central_cells_leiden_mean(
                    self.adata_ref, leiden_key=self.key_ref, n_clusters=nb_cluster,max_workers=max_workers)
                if self.chunks<2:                
                    tar_membership = self._extract_central_cells_leiden_mean(
                        self.adata[:, self.gene_interest], leiden_key=self.key_tar, n_clusters=nb_cluster,max_workers=max_workers)

        self.adata_ref.obs['membership'] = ref_membership
        if self.chunks<2:
            self.adata.obs['membership'] = tar_membership
            
    def _divide_atlas_into_chunks(self, atlas):
        """
        Divide the AnnData object into chunks based on the 'x' coordinate in the `obs` attribute.
        """
        x_min, x_max = atlas.obs['x'].min(), atlas.obs['x'].max()
        x_step = (x_max - x_min) / self.chunks
        boundaries = np.linspace(x_min, x_max, self.chunks + 1)
    
        chunks_dict = {i: [] for i in range(1, self.chunks + 1)}
        for idx, x_value in zip(atlas.obs_names, atlas.obs['x']):
            region = np.digitize(x_value, boundaries, right=True)
            if 1 <= region <= self.chunks:
                chunks_dict[region].append(idx)
            elif region == 0:
                chunks_dict[1].append(idx)
    
        all_cell_names = set(atlas.obs_names)
        all_chunk_cell_names = set(cell for chunk in chunks_dict.values() for cell in chunk)
    
        if all_cell_names == all_chunk_cell_names:
            chunks = [atlas[chunk] for chunk in chunks_dict.values() if chunk]
            if sum(chunk.shape[0] for chunk in chunks) == atlas.shape[0]:
                return chunks
            else:
                print("Warning: Some cells may be missing.")
                return None
        else:
            print("Mismatch in cell names.")
            return None

    def _compute_celltype_means(self,adata, key='membership', gene_interest=None):
        """Helper function to compute cell type means."""
        X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        celltype_means = (
            pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
            .groupby(adata.obs[key])
            .mean()
        )
        obs_grouped = adata.obs.groupby(key).first()
        obs_grouped = obs_grouped.loc[celltype_means.index]
        
        aggregated_adata = sc.AnnData(
            X=celltype_means.values,
            obs=obs_grouped,
            var=adata.var
        )
        aggregated_adata = aggregated_adata[aggregated_adata[:, gene_interest].X.sum(axis=1) > 0]
    
        return aggregated_adata
    
    def _process_chunk(self,chunks, aggregated_ref, op_iter, metric):
        """Helper function to process chunks and reconstruct data."""
        T_combined = pd.DataFrame()
        reconstructed_data = pd.DataFrame()
    
        for idx, chunk in enumerate(chunks):
            self.trials = Trials()
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            print(f"shape {chunk.shape}")
            
            
            self.atlas = chunk[:,self.gene_interest]
            self._run_optimal_transport(aggregated_ref, op_iter, metric)
            
            chunk_T = pd.DataFrame(self.T)
            chunk_T.columns = aggregated_ref.obs_names.astype(str) + '**' + aggregated_ref.obs[self.key_ref].astype(str)
            chunk_T.index = self.atlas.obs_names
    
            T_combined = pd.concat([T_combined, chunk_T], axis=0, sort=False)
            
            chunk_reconstructed = pd.DataFrame(
                aggregated_ref.X.T.dot(chunk_T.T).T,
                columns=aggregated_ref.var_names,
                index=self.atlas.obs_names
            )
            chunk_reconstructed = chunk_reconstructed.div(chunk_reconstructed.sum(axis=1), axis=0)
            
            reconstructed_data = pd.concat([reconstructed_data, chunk_reconstructed], axis=0, sort=False)
    
        return T_combined, reconstructed_data
    
    def annotate(self, op_iter, metric):
        # Step 1: Create aggregated reference AnnData
        if self.central_cells_tar is None:
            aggregated_ref = self._compute_celltype_means(self.adata_ref, key='membership', gene_interest=self.gene_interest)
            print(f"shape of reference central cells matrix: {aggregated_ref.shape}")
            
            # Step 2: Create aggregated target AnnData
            if self.chunks < 2:
                aggregated_tar = self._compute_celltype_means(self.adata, key='membership', gene_interest=self.gene_interest)
#                 self.aggregated_tar = aggregated_tar
                self.atlas = aggregated_tar[:, self.gene_interest]
                print(f"shape of target central cells matrix: {self.atlas.shape}")
            if self.chunks == 100:
                aggregated_tar = self.adata
#                 self.aggregated_tar = aggregated_tar
                self.atlas = aggregated_tar[:, self.gene_interest]
                print(f"shape of target central cells matrix: {self.atlas.shape}")
            
        else:
            aggregated_ref = self.adata_ref[self.adata_ref.obs_names.isin(self.central_cells_ref)]
            aggregated_ref = aggregated_ref[aggregated_ref[:, self.gene_interest].X.sum(axis=1) > 0]
            print(f"shape of reference central cells matrix: {aggregated_ref.shape}")
    
            if self.chunks < 2:
                aggregated_tar = self.adata[self.adata.obs_names.isin(self.central_cells_tar)]
                aggregated_tar = aggregated_tar[aggregated_tar[:, self.gene_interest].X.sum(axis=1) > 0]
                self.atlas = aggregated_tar[:, self.gene_interest]
                print(f"shape of target central cells matrix: {self.atlas.shape}")
    
        # Step 3: Run optimal transport and reconstruct data
        if (self.chunks < 2) or (self.chunks ==100):
            self._run_optimal_transport(aggregated_ref, op_iter, metric)
            pd.DataFrame(self.T)
            self.T.columns = aggregated_ref.obs_names.astype(str) + '**' + aggregated_ref.obs[self.key_ref].astype(str)
            self.T.index = self.atlas.obs.index
            self.X_reconstructed = pd.DataFrame(aggregated_ref.X.T.dot(self.T.T).T, columns=aggregated_ref.var_names)
        else:
            chunks = self._divide_atlas_into_chunks(self.adata)
            if chunks is not None:
                self.T, self.X_reconstructed = self._process_chunk(chunks, aggregated_ref, op_iter, metric)
                self.X_reconstructed = self.X_reconstructed.loc[self.adata.obs_names, :]
                self.T = self.T.loc[self.adata.obs_names, :]
#                 self.atlas = self.adata
#                 self.adata.obs['membership'] = self.adata.obs[self.key_tar]
            else:
                print("Chunks are not available. Please check the divide_atlas_into_chunks function.")
    
        # Step 4: Map and assign predicted cell types
        if self.chunks < 2:
            self.atlas.obs['predicted_annotation'] = [self.T.columns[i].split('**')[1] for i in np.argmax(self.T.values, axis=1)]
            mapped_celltypes = self.atlas.obs['predicted_annotation'].astype('str').reindex(self.adata.obs['membership'])
            mapped_celltypes.fillna(pd.Series(self.adata.obs['membership'].values, index=mapped_celltypes.index), inplace=True)
            self.adata.obs['predicted_annotation'] = mapped_celltypes.values
        else:
            self.adata.obs['predicted_annotation'] = [self.T.columns[i].split('**')[1] for i in np.argmax(self.T.values, axis=1)]
