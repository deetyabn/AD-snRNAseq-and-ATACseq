import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### For normalized RNA-seq data passed into the VAE with barcodes for the embeddings ###
file_path = "/orcd/data/manoli/001/tnikitha/scAD/multiome_rna.h5ad"
adata = sc.read_h5ad(file_path, backed='r')

csv_path = "vae_latent_space.csv"
latent_df = pd.read_csv(csv_path)

plot_adata = sc.AnnData(obs=adata.obs.copy())

latent_df.set_index('Sample_barcode', inplace=True)
plot_adata.obsm['X_vae'] = latent_df.loc[plot_adata.obs_names].values

sc.pp.neighbors(plot_adata, use_rep='X_vae')
sc.tl.umap(plot_adata)

plot_cols = ['RNA.Class.Mar31_2024', 'BrainRegion', 'RNA.Subclass.Mar31_2024']

sc.pl.umap(plot_adata, color=plot_cols, show=False, frameon=False, ncols=2)

plt.savefig("vae_umap_verification.png", bbox_inches='tight', dpi=300)


### For unnormalized RNA-seq data passed into the VAE ###
adata = sc.read_h5ad("/orcd/data/manoli/001/tnikitha/scAD/multiome_rna.h5ad", backed='r')
vae_z = np.load("vae_latent_space_no_normalization.npy")

plot_adata = sc.AnnData(obs=adata.obs.copy())
plot_adata.obsm['X_vae'] = vae_z

sc.pp.neighbors(plot_adata, use_rep='X_vae')
sc.tl.umap(plot_adata)

sc.pl.umap(plot_adata, color=['RNA.Class.Mar31_2024', 'BrainRegion'],
           show=False, frameon=False)
plt.savefig("vae_umap_verification.png", bbox_inches='tight')


### For scVI exploration ###
adata = sc.read_h5ad("/orcd/data/manoli/001/tnikitha/scAD/multiome_rna.h5ad", backed='r')

df_latent = pd.read_csv("scvi_latent_with_barcodes.csv")
df_latent.set_index('Sample_barcode', inplace=True)
df_latent = df_latent.loc[adata.obs_names]

plot_adata = sc.AnnData(obs=adata.obs.copy())
plot_adata.obsm['X_scvi'] = df_latent.values

sc.pp.neighbors(plot_adata, use_rep='X_scvi')
sc.tl.umap(plot_adata)

sc.pl.umap(plot_adata, color=['RNA.Class.Mar31_2024', 'BrainRegion', 'RNA.Subclass.Mar31_2024'],
           show=False, frameon=False)

plt.savefig("scvi_umap_verification.png", bbox_inches='tight')
