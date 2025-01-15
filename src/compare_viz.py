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
np.random.seed(42) 

class compare_viz:
    def __init__(self, adata, adata_ref, key_ref):

        self.adata = adata
        self.adata_ref=adata_ref
        self.key_ref=key_ref
        

    def preprocess_adata(self):
#         """Preprocess both reference and target AnnData objects."""
#         sc.pp.filter_cells(self.adata_ref, min_genes=1)
#         sc.pp.filter_cells(self.adata, min_genes=1)

#         sc.pp.normalize_total(self.adata_ref, layer='counts')
#         sc.pp.log1p(self.adata_ref)
        sc.pp.pca(self.adata_ref)
        sc.pp.neighbors(self.adata_ref)
        sc.tl.umap(self.adata_ref)

#         sc.pp.normalize_total(self.adata, layer='counts')
#         sc.pp.log1p(self.adata)
#         sc.pp.pca(self.adata)
#         sc.pp.neighbors(self.adata)

    def ingest(self,ngh=15):
        self.preprocess_adata()
        b = np.array(list(map(len, self.adata_ref.obsp['distances'].tolil().rows)))
        adata_ref_subset = self.adata_ref[np.where(b > 2)[0]]
        sc.pp.neighbors(adata_ref_subset, n_neighbors=ngh)
        sc.tl.ingest(self.adata, adata_ref_subset, obs=self.key_ref)
        
    def dotplot(self,key_label,target_celltype,groups,markers):
        adata_subset= self.adata[self.adata.obs[key_label]== target_celltype,:]
        for gr in groups:
            ss=sc.pl.dotplot(adata_subset, markers, 
                     return_fig=True, groupby= gr)
            ss.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5).show()
            
    def corr_cos_plot(self,X_org_, X_pred_, sample=True):
        if sample==True:
            X_pred_=X_pred_.T
            X_org_=X_org_.T
        num_cols = X_org_.shape[1]
        corr_matrix=np.corrcoef(X_org_,X_pred_, rowvar=False)
        correlations = np.diagonal(corr_matrix[:num_cols, num_cols:])
        correlation_results= pd.Series(correlations, index= X_org_.columns)
        mean_corr = correlation_results.mean()
        cos_sim = cosine_similarity(X_pred_.T, X_org_.T)
        cosine_results = np.diag(cos_sim)
        mean_cos=np.mean(cosine_results)

        # Print the mean correlation and cosine similarity
        print(f'mean correlation:{mean_corr}')
        print(f'mean ccosine:{mean_cos}')

#         # Set global font size for better readability
#         plt.rcParams.update({'font.size': 14})

#         # Plot the distribution of correlations
#         plt.figure(figsize=(12, 6))
#         sns.histplot(correlation_results, bins=20, kde=True, stat="density", color='skyblue', alpha=0.3)
#         plt.axvline(mean_corr, color='red', linestyle='dashed', linewidth=1)
#         plt.xlabel('Correlation', fontsize=16)
#         plt.ylabel('Density', fontsize=16)
#         plt.title('Correlation per Sample', fontsize=18)
#         plt.legend({'Mean': mean_corr}, fontsize=14)
#         plt.savefig('correlation_density_plot.pdf', format='pdf')
#         plt.show()

#         # Plot the distribution of cosine similarities
#         plt.figure(figsize=(12, 6))
#         sns.histplot(cosine_results, bins=20, kde=True, stat="density", color='skyblue', alpha=0.3)
#         plt.axvline(mean_cos, color='red', linestyle='dashed', linewidth=1)
#         plt.xlabel('Cosine Similarity', fontsize=16)
#         plt.ylabel('Density', fontsize=16)
#         plt.title('Cosine Similarity per Sample', fontsize=18)
#         plt.legend({'Mean': mean_cos}, fontsize=14)
#         plt.savefig('cosine_density_plot.pdf', format='pdf')
#         plt.show()
        return correlation_results,cosine_results


