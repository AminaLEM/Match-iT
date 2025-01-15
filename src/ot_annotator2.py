import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
import ot
from hyperopt import fmin, tpe, Trials, STATUS_OK 


class OTAnnotator:
    def __init__(self, adata: sc.AnnData, adata_ref: sc.AnnData, gene_interest,
                 param_space, key_ref = 'leiden', key_tar= 'leiden',
                 way = 'all',verbose=False):
        np.random.seed(42)        
        self.adata = adata
        self.adata_ref = adata_ref
        self.gene_interest = gene_interest
        self.param_space = param_space
        self.key_ref = key_ref
        self.key_tar = key_tar
        self.way = way
        self.trials = Trials()
        self.T = None
        self.X_reconstructed = pd.DataFrame()
        self.best_params=None
        self.verbose=verbose
        self._preprocess_adata()

    def _preprocess_adata(self):
        self.adata = self.adata[self.adata[:, self.gene_interest].X.sum(axis=1) > 0]
        sc.pp.filter_cells(self.adata, min_genes=1)
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)
        
        sc.pp.filter_cells(self.adata_ref, min_genes=1)
        sc.pp.normalize_total(self.adata_ref)
        sc.pp.log1p(self.adata_ref)


    def _compute_corr(self, X_org: pd.DataFrame, X_pred: pd.DataFrame):
        X_org_flat = X_org.values.flatten()
        X_pred_flat = X_pred.values.flatten()

        corr_matrix = np.corrcoef(X_org_flat, X_pred_flat)
        corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
        return 0 if corr == 1 else corr

    def _objective_function(self, params, cost_matrix, p, q, aggregated_tar,aggregated_ref):
        Ts = ot.unbalanced.sinkhorn_unbalanced(
            p, q, cost_matrix, params['reg'],
            [params['reg_m_kl_1'], params['reg_m_kl_2']],
            method=params['method'], reg_type=params['reg_type'], log=False)

        T = pd.DataFrame(Ts.T, index=aggregated_tar.obs_names, columns=aggregated_ref.obs_names)
        X_pred = pd.DataFrame(aggregated_ref.X.T.dot(T.T), index=aggregated_ref.var_names).T

        valid_idx = ~np.isnan(X_pred.iloc[:, 0])
        if valid_idx.sum() > 1:
            X_pred = X_pred.loc[valid_idx, self.gene_interest]
            X_org = pd.DataFrame(aggregated_tar.X.toarray(), columns=aggregated_tar.var_names).loc[valid_idx, self.gene_interest]
            corr = self._compute_corr(X_org, X_pred)
        else:
            corr = 0

        return {'loss': -corr, 'model': T,'status':STATUS_OK, 'params': params}

    def _run_optimal_transport(self, aggregated_tar,aggregated_ref, op_iter, metric):
        source_matrix = aggregated_ref[:, self.gene_interest].X.toarray()
        target_matrix = aggregated_tar[:, self.gene_interest].X.toarray()
        cost_matrix = ot.dist(source_matrix, target_matrix, metric=metric)
        if metric =='euclidean':
            cost_matrix /= cost_matrix.max()

        p, q = ot.unif(source_matrix.shape[0]), ot.unif(target_matrix.shape[0])

        self.best_params = fmin(
            fn=lambda params: self._objective_function(params, cost_matrix, p, q,aggregated_tar, aggregated_ref),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=op_iter,
            trials=self.trials
        )
        T = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['model']
        T=pd.DataFrame(T)
        return T


    def _kmean_membership(self, adata, leiden_key, n_clusters, min_cluster_size=5):
        adata.obs['kmeans'] = np.nan
        if adata.shape[1]>2000:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            adata = adata[:, adata.var['highly_variable']]        # Iterate over each unique cell type in the Leiden clustering
        for cell_type in adata.obs[leiden_key].unique():
            cell_type_indices = adata.obs[adata.obs[leiden_key] == cell_type].index
            cell_type_data = adata[cell_type_indices, :]
            X = cell_type_data.X
            X = X.toarray() if not isinstance(X, np.ndarray) else X                
            if len(cell_type_data) > min_cluster_size:
                K = min(n_clusters, max(2, int(len(cell_type_data)/50)))
                if self.verbose:
                        print(f"Clustering {len(X)} cells from {cell_type} into {K} clusters...")

                kmeans = KMeans(n_clusters=K,n_init=10, random_state=42)
                kmeans_labels = kmeans.fit_predict(X)
                label_counts = pd.Series(kmeans_labels).value_counts(normalize=True)
                noise_clusters = label_counts[label_counts < 0.02].index
                if self.verbose:
                    if len(noise_clusters)>0:
                        print(f"#noise clusters: {len(noise_clusters)}")
                kmeans_labels = np.where(np.isin(kmeans_labels, noise_clusters), np.nan, kmeans_labels)
                adata.obs.loc[cell_type_indices, 'kmeans'] = kmeans_labels.astype(str)
        adata.obs['celltype_kmeans'] = (
            adata.obs[leiden_key].astype(str) + '_' + adata.obs['kmeans'].astype(str))
        if self.verbose:
            print(f"#noise cells:{np.sum(adata.obs['celltype_kmeans'].str.contains('nan', na=False))}")
        return adata.obs['celltype_kmeans']

    def subcluster(self,nb_cluster=10,resolution=10):
        if 'celltype_kmeans' in self.adata_ref.obs.columns:
            ref_membership=self.adata_ref.obs[clus_meth]
        else:
            ref_membership = self._kmean_membership(
                    self.adata_ref, leiden_key=self.key_ref, n_clusters=nb_cluster)

        if self.way !='all':
                sc.pp.pca(self.adata)
                sc.pp.neighbors(self.adata)
                sc.tl.leiden(self.adata, resolution=resolution, key_added=self.key_tar)
                tar_membership = self._kmean_membership(
                        self.adata[:, self.gene_interest], leiden_key=self.key_tar, n_clusters=nb_cluster)
                self.adata.obs['membership'] = tar_membership
        self.adata_ref.obs['membership'] = ref_membership
        
    def _compute_celltype_means(self, adata, key='membership', gene_interest=None):
        adata =  adata[~adata.obs['membership'].str.contains('nan', na=False)]
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
            
    
    def annotate(self, op_iter=100, metric='cosine'):
        aggregated_ref = self._compute_celltype_means(self.adata_ref, key='membership', gene_interest=self.gene_interest)            
        if self.way !='all':
                aggregated_tar = self._compute_celltype_means(self.adata, key='membership', gene_interest=self.gene_interest)
                aggregated_tar = aggregated_tar[:, self.gene_interest]
        else:
                aggregated_tar = self.adata[:, self.gene_interest]
        print(f"shape of target matrix: {aggregated_tar.shape}")
        print(f"shape of reference matrix: {aggregated_ref.shape}")
    
        self.T=self._run_optimal_transport(aggregated_tar,aggregated_ref, op_iter, metric)
        self.T.columns = aggregated_ref.obs_names.astype(str) + '**' + aggregated_ref.obs[self.key_ref].astype(str)
        self.T.index = aggregated_tar.obs.index
        self.X_reconstructed = pd.DataFrame(aggregated_ref.X.T.dot(self.T.T).T, columns=aggregated_ref.var_names)
        if self.way !='all':
            aggregated_tar.obs['predicted_annotation'] = [self.T.columns[i].split('**')[1] for i in np.argmax(self.T.values, axis=1)]
            mapped_celltypes = aggregated_tar.obs['predicted_annotation'].astype('str').reindex(self.adata.obs['membership'])
            mapped_celltypes.fillna(pd.Series(self.adata.obs['membership'].values, index=mapped_celltypes.index), inplace=True)
            self.adata.obs['predicted_annotation'] = mapped_celltypes.values
        else:
            self.adata.obs['predicted_annotation'] = [self.T.columns[i].split('**')[1] for i in np.argmax(self.T.values, axis=1)]
