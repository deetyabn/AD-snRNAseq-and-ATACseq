import os
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
import h5py
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import gc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── Load data ──────────────────────────────────────────────────────────────────
rna_multi  = sc.read_h5ad("/home/tnikitha/orcd/scratch/multiome_rna.h5ad")
atac_multi = sc.read_h5ad("/home/tnikitha/orcd/scratch/multiome_atac.h5ad")

# ── Stratified sampling ────────────────────────────────────────────────────────
cell_type_counts = atac_multi.obs['Celltype_Jan14_2024'].value_counts()
total_budget = 10000
min_per_type = 10
size_tag     = f"{total_budget // 1000}k"

tier1_allocation = {ct: min(count, min_per_type) for ct, count in cell_type_counts.items()}
tier1_total      = sum(tier1_allocation.values())
remaining_budget = total_budget - tier1_total

sqrt_counts      = np.sqrt(cell_type_counts.values)
sqrt_proportions = sqrt_counts / sqrt_counts.sum()

tier2_allocation = {}
for ct, proportion in zip(cell_type_counts.index, sqrt_proportions):
    additional = int(remaining_budget * proportion)
    tier2_allocation[ct] = min(additional, cell_type_counts[ct] - tier1_allocation[ct])

tier2_total = sum(tier2_allocation.values())
if tier2_total < remaining_budget:
    for ct in cell_type_counts.index[:remaining_budget - tier2_total]:
        if tier1_allocation[ct] + tier2_allocation[ct] < cell_type_counts[ct]:
            tier2_allocation[ct] += 1

final_allocation = {ct: tier1_allocation[ct] + tier2_allocation[ct] for ct in cell_type_counts.index}

rng = np.random.default_rng(42)
sampled_cells = []
for ct, n_sample in final_allocation.items():
    pool    = atac_multi.obs_names[atac_multi.obs['Celltype_Jan14_2024'] == ct].tolist()
    sampled = rng.choice(pool, size=n_sample, replace=False).tolist()
    sampled_cells.extend(sampled)

atac_multi_sampled = atac_multi[sampled_cells].copy()
atac_multi_sampled.write_h5ad(f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/atac_multi_sampled_{size_tag}.h5ad")
print(f"Total sampled cells: {len(sampled_cells)}")
print(atac_multi_sampled.obs['Celltype_Jan14_2024'].value_counts())

# ── Load full ATAC LSI + class labels ─────────────────────────────────────────
with h5py.File("atac4.cCRE_Matrix.1217165×1010153.Jan11_2025.h5ad", "r") as f:
    X_lsi_full        = f['obsm/X_lsi_harmony'][:]
    codes             = f['obs/Class_Jan15_2024/codes'][:]
    categories        = [x.decode() for x in f['obs/Class_Jan15_2024/categories'][:]]
    class_labels_full = pd.Categorical.from_codes(codes, categories)
    cell_names_full   = np.array([x.decode() for x in f['obs/_index'][:]])

print(f"Loaded LSI: {X_lsi_full.shape}")
print(f"Classes:    {len(set(class_labels_full))} unique")

# ── KNN per celltype ───────────────────────────────────────────────────────────
all_neighbor_indices = {}
query_order          = []
k                    = 100

for cls in tqdm(atac_multi_sampled.obs['Class_Jan15_2024'].unique(), desc="Building KNN per class"):
    full_mask      = class_labels_full == cls
    full_indices   = np.where(full_mask)[0]
    X_full_cls     = X_lsi_full[full_mask]

    query_mask     = atac_multi_sampled.obs['Class_Jan15_2024'] == cls
    X_query_cls    = atac_multi_sampled.obsm['X_lsi_harmony'][query_mask]
    query_cell_ids = atac_multi_sampled.obs_names[query_mask]

    if len(query_cell_ids) == 0:
        continue

    nn = NearestNeighbors(n_neighbors=min(k, X_full_cls.shape[0] - 1), metric='euclidean', n_jobs=-1)
    nn.fit(X_full_cls)
    distances, local_indices = nn.kneighbors(X_query_cls)

    global_neighbor_indices = full_indices[local_indices]
    for i, cell_id in enumerate(query_cell_ids):
        all_neighbor_indices[cell_id] = global_neighbor_indices[i]
        query_order.append(cell_id)

    del X_query_cls, distances, local_indices, global_neighbor_indices
    gc.collect()

print(f"Computed neighbors for {len(all_neighbor_indices)} cells")

del X_lsi_full, class_labels_full, cell_names_full
gc.collect()

# ── Pseudobulking ──────────────────────────────────────────────────────────────
atac_full = ad.read_h5ad(
    "/home/tnikitha/orcd/scratch/atac4.cCRE_Matrix.1217165×1010153.Jan11_2025.h5ad",
    backed='r'
)

n_cells     = len(query_order)
n_peaks     = atac_full.shape[1]
output_file = f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/pseudobulk_atac_stratified_{size_tag}.h5"

celltype_map      = atac_multi_sampled.obs['Class_Jan15_2024']
celltypes_ordered = celltype_map.reindex(query_order).values
barcode_map       = atac_multi_sampled.obs['Sample_barcode']
barcodes_ordered  = barcode_map.reindex(query_order).values

with h5py.File(output_file, 'w') as f:
    f.create_dataset('data',           shape=(0,),           maxshape=(None,), dtype='float32',
                     chunks=(100_000,), compression='gzip',  compression_opts=4)
    f.create_dataset('indices',        shape=(0,),           maxshape=(None,), dtype='int32',
                     chunks=(100_000,), compression='gzip',  compression_opts=4)
    f.create_dataset('indptr',         shape=(n_cells + 1,), dtype='int64')
    f.create_dataset('shape',          data=np.array([n_cells, n_peaks], dtype='int64'))
    f.create_dataset('cell_ids',       data=np.array(query_order,        dtype='S'))
    f.create_dataset('celltypes',      data=np.array(celltypes_ordered,  dtype='S'))
    f.create_dataset('Sample_barcode', data=np.array(barcodes_ordered,   dtype='S'))

    indptr         = [0]
    data_buffer    = []
    indices_buffer = []
    batch_size     = 2000
    write_every    = 5

    for batch_idx in tqdm(range(0, n_cells, batch_size), desc="Pseudobulking"):
        batch_cells = query_order[batch_idx: batch_idx + batch_size]

        all_neighbors_in_batch = []
        for cell_id in batch_cells:
            all_neighbors_in_batch.extend(all_neighbor_indices[cell_id])

        unique_neighbors = np.unique(all_neighbors_in_batch)
        neighbor_data    = atac_full.X[unique_neighbors]
        neighbor_map     = {idx: i for i, idx in enumerate(unique_neighbors)}

        for cell_id in batch_cells:
            positions      = [neighbor_map[n] for n in all_neighbor_indices[cell_id]]
            pseudobulk_row = sp.csr_matrix(neighbor_data[positions].sum(axis=0))
            data_buffer.append(pseudobulk_row.data)
            indices_buffer.append(pseudobulk_row.indices)
            indptr.append(indptr[-1] + len(pseudobulk_row.data))

        if (batch_idx // batch_size + 1) % write_every == 0 or batch_idx + batch_size >= n_cells:
            if data_buffer:
                all_data     = np.concatenate(data_buffer)
                all_indices  = np.concatenate(indices_buffer)
                current_size = f['data'].shape[0]
                f['data'].resize((current_size + len(all_data),))
                f['indices'].resize((current_size + len(all_indices),))
                f['data'][current_size:]    = all_data
                f['indices'][current_size:] = all_indices
                data_buffer    = []
                indices_buffer = []
                gc.collect()

    f['indptr'][:] = indptr

atac_full.file.close()
del atac_full
gc.collect()

print("✓ Pseudobulking complete!")
print(f"File size: {os.path.getsize(output_file) / 1e6:.1f} MB")

# ── Subset and reorder RNA to match pseudobulk barcode order ──────────────────
with h5py.File(output_file, "r") as f:
    barcodes = f["Sample_barcode"][:].astype(str)

rna_multi_sampled = rna_multi[rna_multi.obs['Sample_barcode'].isin(barcodes)].copy()
barcode_to_obs    = rna_multi_sampled.obs.reset_index().set_index('Sample_barcode')['index']
ordered_obs_names = barcode_to_obs.reindex(barcodes).values
rna_multi_sampled = rna_multi_sampled[ordered_obs_names].copy()
rna_multi_sampled.write_h5ad(f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/rna_multi_sampled_{size_tag}.h5ad")

with h5py.File(output_file, "r") as f:
    atac_barcodes = f["Sample_barcode"][:].astype(str)
print(f"RNA/ATAC barcode match: {all(rna_multi_sampled.obs['Sample_barcode'].values == atac_barcodes)}")

# ── Train / val / test / held-out split ───────────────────────────────────────
with h5py.File(output_file, "r") as f:
    cell_ids  = f["cell_ids"][:].astype(str)
    celltypes = f["celltypes"][:].astype(str)

held_out_type  = 'OPC'
held_out_mask  = celltypes == held_out_type
held_out_ids   = cell_ids[held_out_mask]
remaining_mask = ~held_out_mask

remaining_ids       = cell_ids[remaining_mask]
remaining_celltypes = celltypes[remaining_mask]

train_ids, temp_ids = train_test_split(
    remaining_ids, test_size=0.2,
    stratify=remaining_celltypes, random_state=42
)
temp_celltypes = remaining_celltypes[np.isin(remaining_ids, temp_ids)]
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.5,
    stratify=temp_celltypes, random_state=42
)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Held-out: {held_out_mask.sum()}")

np.save(f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/train_ids_{size_tag}.npy",    train_ids)
np.save(f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/val_ids_{size_tag}.npy",      val_ids)
np.save(f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/test_ids_{size_tag}.npy",     test_ids)
np.save(f"/orcd/data/manoli/001/tnikitha/scAD/pseudobulking/held_out_ids_{size_tag}.npy", held_out_ids)
