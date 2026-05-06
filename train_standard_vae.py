import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import time
import os

FILE_PATH = "/orcd/data/manoli/001/tnikitha/scAD/multiome_rna.h5ad"
OUTPUT_PATH = "/orcd/home/002/avdusen/orcd/pool/68711_final_project_code/vae_latent_space.csv"
INPUT_DIM = 36601
LATENT_DIM = 30
HIDDEN_DIM = 256
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 0.001

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

class SparseExpressionDataset(Dataset):
    def __init__(self, sparse_matrix):
        self.data = sparse_matrix

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32)

class scVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(scVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adata = sc.read_h5ad(FILE_PATH)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    dataset = SparseExpressionDataset(adata.X)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = scVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        epoch_time = time.time() - start_time
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    model.eval()
    latent_space = []

    extract_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    with torch.no_grad():
        for data in extract_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_space.append(mu.cpu().numpy())

    latent_space = np.vstack(latent_space)

    # adding barcodes to latent embeddings
    barcodes = adata.obs['Sample_barcode'].values
    latent_cols = [f"latent_{i}" for i in range(LATENT_DIM)]
    df = pd.DataFrame(latent_space, columns=latent_cols)
    df.insert(0, 'Sample_barcode', barcodes)
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
