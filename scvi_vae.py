import scvi
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("/orcd/data/manoli/001/tnikitha/scAD/multiome_rna.h5ad")
scvi.model.SCVI.setup_anndata(adata)

model = scvi.model.SCVI(adata, n_latent=30, n_layers=2)
model.train()

latent_space = model.get_latent_representation()
adata.obsm['X_scvi'] = latent_space

barcodes = adata.obs['Sample_barcode'].values

column_names = [f"latent_{i}" for i in range(latent_space.shape[1])]
df = pd.DataFrame(latent_space, columns=column_names)

df.insert(0, 'Sample_barcode', barcodes)

df.to_csv("scvi_latent_with_barcodes.csv", index=False)
