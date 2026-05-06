import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import h5py
from scipy.sparse import csr_matrix
import os
import json


OUTDIR = "deetyabn/outputs_diffusion_10k_vae5_"
os.makedirs(OUTDIR, exist_ok=True)

from torch.utils.data import Dataset, DataLoader
from multimodal_dataset import MultimodalDataManager
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

# use efficient data loader for RNAseq and pseudobulked ATACseq data (10k cells between splits)
dm = MultimodalDataManager("pseudobulking", suffix="10k")

# -----------------------------
# Training / Val / Test Splits
# -----------------------------
train_ids = np.load("./pseudobulking/train_ids_10k.npy", allow_pickle=True)
val_ids   = np.load("./pseudobulking/val_ids_10k.npy", allow_pickle=True)
test_ids  = np.load("./pseudobulking/test_ids_10k.npy", allow_pickle=True)
held_ids  = np.load("./pseudobulking/held_out_ids_10k.npy", allow_pickle=True)

train_ids = train_ids.astype(str)
val_ids   = val_ids.astype(str)
test_ids  = test_ids.astype(str)
held_ids  = held_ids.astype(str)

# -----------------------------
# VAE Embeddings (Match by cell barcode)
# -----------------------------
embeddings_df = pd.read_csv('/orcd/data/manoli/001/tnikitha/scAD/vae_latent_space.csv')
embeddings_df["Sample_barcode"] = embeddings_df["Sample_barcode"].astype(str)
latent_cols = [c for c in embeddings_df.columns if c.startswith("latent_")]
X = embeddings_df[latent_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std < 1e-6] = 1  # prevent division issues
X = (X - mean) / std
embeddings_df[latent_cols] = X
embeddings_indexed = embeddings_df.set_index("Sample_barcode")

class VAEEmbeddingDataset(Dataset):
    def __init__(self, base_dataset, embeddings_indexed, ids, latent_cols):
        self.base_dataset = base_dataset
        self.embeddings_indexed = embeddings_indexed
        self.ids = np.asarray(ids).astype(str)
        self.latent_cols = latent_cols

        assert len(self.base_dataset) == len(self.ids), (
            f"Dataset length {len(self.base_dataset)} != IDs length {len(self.ids)}"
        )

        missing = set(self.ids) - set(self.embeddings_indexed.index)
        if len(missing) > 0:
            raise ValueError(f"{len(missing)} IDs missing from embeddings_df.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        barcode = self.ids[idx]

        x = self.embeddings_indexed.loc[barcode, self.latent_cols].values.astype("float32")
        y = item["atac"].float()

        return {
            "rna_emb": torch.from_numpy(x),
            "atac": y,
            "barcode": barcode,
        }

# -----------------------------
# Build datasets
# -----------------------------
train_ds = VAEEmbeddingDataset(
    dm.get_dataset("train"),
    embeddings_indexed,
    train_ids,
    latent_cols,
)

val_ds = VAEEmbeddingDataset(
    dm.get_dataset("val"),
    embeddings_indexed,
    val_ids,
    latent_cols,
)

test_ds = VAEEmbeddingDataset(
    dm.get_dataset("test"),
    embeddings_indexed,
    test_ids,
    latent_cols,
)

held_ds = VAEEmbeddingDataset(
    dm.get_dataset("held_out"),
    embeddings_indexed,
    held_ids,
    latent_cols,
)

# -----------------------------
# Build dataloaders
# -----------------------------
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)
held_loader  = DataLoader(held_ds, batch_size=128, shuffle=False)

# -----------------------------
# Check shapes
# -----------------------------
sample_batch = next(iter(train_loader))

print(sample_batch["rna_emb"].shape, flush=True)
print(sample_batch["atac"].shape, flush=True)
print(sample_batch["barcode"][:5], flush=True)

input_size = sample_batch["rna_emb"].shape[1]
output_size = sample_batch["atac"].shape[1]

print(f"Input size: {input_size}", flush=True)
print(f"Output size: {output_size}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Weighted MSE by ATACseq peak variance
# -----------------------------
n = 0
sum_y = None
sum_y2 = None

from tqdm import tqdm

stats_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

for batch in tqdm(stats_loader, desc="Computing ATAC mean/variance"):
    y = batch["atac"].float()
    y = torch.log1p(y)

    if sum_y is None:
        sum_y = torch.zeros(y.shape[1], dtype=torch.float64)
        sum_y2 = torch.zeros(y.shape[1], dtype=torch.float64)

    sum_y += y.sum(dim=0).double()
    sum_y2 += (y ** 2).sum(dim=0).double()
    n += y.shape[0]

mean_atac = sum_y / n
peak_var = (sum_y2 / n) - (mean_atac ** 2)

weights = peak_var / peak_var.mean()
weights = torch.clamp(weights, max=10.0)

weights = weights.to(device)
mean_atac = mean_atac.to(device)

def weighted_mse(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)

# -----------------------------
# Diffusion Model Architecture
# -----------------------------

# Nichol and Dhariwal, 2021 - Improved DDPM, Cosine Noise Schedule
def cosine_noise_schedule(T, s=0.008):
  # define alpha_t as cosine curve
  x = torch.linspace(0, T, T+1)
  alpha_bar_prod = torch.cos(((x/T) + s) / (1+s) * torch.pi/2) ** 2 # fraction of signal left after t steps (cosine with slight offset)
  alpha_bar_prod = alpha_bar_prod/alpha_bar_prod[0] # correct for offset so a_0 = 1
  betas = 1 - (alpha_bar_prod[1:] / alpha_bar_prod[:-1])
  return torch.clamp(betas, min=1e-5, max=0.999)

# diffusion step
class ATACDiffusion:
  def __init__(self, T=1000, device='cuda'):
    self.T = T
    self.device = device
    # precompute noise scale, signal scale at each time step
    betas = cosine_noise_schedule(T).to(device)
    alphas = 1.0 - betas
    self.betas = betas
    self.alphas = alphas
    self.alpha_bar = torch.cumprod(alphas, dim=0)
    self.sqrt_alpha_cp = self.alpha_bar.sqrt()
    self.sqrt_noise_scale = (1.0 - self.alpha_bar).sqrt()

    # used later for p sampling
    prev_alpha_bar = torch.cat([torch.ones(1).to(device), self.alpha_bar[:-1]])
    self.posterior_variance = betas * (1.0 - prev_alpha_bar) / (1.0 - self.alpha_bar)

  # forward: add noise
  def q_sample(self, x_0, t, noise=None):
    if noise is None:
      noise = torch.randn_like(x_0)
    sqrt_alpha = self.sqrt_alpha_cp[t].view(-1, 1)
    sqrt_noise_sc = self.sqrt_noise_scale[t].view(-1, 1)
    return sqrt_alpha * x_0 + sqrt_noise_sc * noise, noise

  # backward: remove noise using denoising model
  @torch.no_grad()
  def p_sample(self, model, xt, z, t):
    t_tensor = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
    eps_pred = model(xt, z, t_tensor)
    alpha = self.alphas[t]
    mean = (1 / alpha.sqrt()) * (xt - (self.betas[t] / self.sqrt_noise_scale[t]) * eps_pred)
    if t == 0:
        return mean
    return mean + self.posterior_variance[t].sqrt() * torch.randn_like(xt)

  # repeat denoising for T time steps to generate sample
  @torch.no_grad()
  def sample(self, model, z, atac_dim, n_samples=1):
    x = torch.randn(n_samples, atac_dim).to(self.device)
    for t in reversed(range(self.T)):
      x = self.p_sample(model, x, z, t)
    return x

# map timesteps to continuous vector to be meaningful for model
class SinusoidalEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, t):
    device = t.device
    half = self.dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)

# core residual block for denoiser
class ResidualBlock(nn.Module):
  def __init__(self, dim, cond_dim):
    super().__init__()
    self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
      )
    # project conditional vector (z = RNAseq latent embedding from Schema + t from timestep)
    self.cond_proj = nn.Linear(cond_dim, dim)

  def forward(self, x, cond):
    return x + self.net(x + self.cond_proj(cond))

# simple denoiser based on residual blocks --> predicts noise
class ATACDenoiser(nn.Module):
  def __init__(self, atac_dim, z_dim=64, hidden=512, depth=6, t_emb_dim=128):
    super().__init__()

    # concatenate z + t to single conditioning vec, project to hidden dim
    self.t_emb = SinusoidalEmbedding(t_emb_dim)
    cond_dim = z_dim + t_emb_dim
    self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
    )

    # project input to hidden dimension, compress atac peaks
    self.input_proj = nn.Linear(atac_dim, hidden)

    # denoising core with residual blocks
    self.blocks = nn.ModuleList([
            ResidualBlock(hidden, hidden) for _ in range(depth)
    ])
    self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, atac_dim),
    )

  def forward(self, xt, z, t):
    t_emb = self.t_emb(t)
    cond = self.cond_proj(torch.cat([z, t_emb], dim=-1))
    h = self.input_proj(xt)
    for block in self.blocks:
      h = block(h, cond)
    return self.output(h)


# -----------------------------
# TRAINING HELPER
# -----------------------------

def train_epoch(model, diffusion, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        z = batch["rna_emb"].to(device)
        atac = torch.log1p(batch["atac"].to(device)) # log normalize

        t = torch.randint(0, diffusion.T, (atac.shape[0],), device=device) # sample random timesteps

        x_noisy, noise = diffusion.q_sample(atac, t) # forward process: add noise

        noise_pred = model(x_noisy, z, t) # predict noise

        loss = weighted_mse(noise_pred, noise, weights) # mse on noise

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# -----------------------------
# TRAINING BLOCK
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
atac_dim = sample_batch["atac"].shape[1]
diffusion = ATACDiffusion(device=device)
model = ATACDenoiser(atac_dim, z_dim=30)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

N_EPOCHS   = 30
EVAL_EVERY = 5

# to implement early stopping
patience = 5
epochs_no_improve = 0

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'val_corr': [], 'val_r2': []}

diffusion_fast = ATACDiffusion(T=200, device=device) # smaller diffusion for intermediate validation

for epoch in range(N_EPOCHS):
    train_loss = train_epoch(model, diffusion, train_loader, optimizer, device)

    # validation loss at each epoch
    model.eval()
    total_val_loss = 0.
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            z = batch["rna_emb"].to(device)
            atac = torch.log1p(batch["atac"].to(device))
            t = torch.randint(0, diffusion.T, (atac.shape[0],), device=device)
            x_noisy, noise = diffusion.q_sample(atac, t)
            noise_pred = model(x_noisy, z, t)
            total_val_loss += weighted_mse(noise_pred, noise, weights).item()

    val_loss = total_val_loss / len(val_loader)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    # generation quality evaluation every 5 epochs
    if epoch % EVAL_EVERY == 0:
        with torch.no_grad():
            quick_batch = next(iter(val_loader))
            z_q = quick_batch["rna_emb"].to(device)
            atac_q = torch.log1p(quick_batch["atac"].to(device))
            gen_q = diffusion_fast.sample(model, z_q, atac_q.shape[1], n_samples=z_q.shape[0])
            val_r2 = r2_score(atac_q.cpu().numpy(), gen_q.cpu().numpy())
            quick_corrs = []
            for i in range(atac_q.shape[0]):
                r = torch.corrcoef(torch.stack([gen_q[i], atac_q[i]]))[0, 1]
                quick_corrs.append(r.item())
            val_corr_quick = np.mean(quick_corrs)
        history['val_corr'].append((epoch, val_corr_quick))
        history['val_r2'].append((epoch, val_r2))

    # save best model and update history
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{OUTDIR}/best_10k_model.pt")
    else:
       epochs_no_improve += 1

    with open(f"{OUTDIR}/history.json", "w") as f:
      json.dump(history, f)

    print(f"epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | pearson_r = {val_corr_quick:.3f} | r2 = {val_r2:.3f}", flush=True)

    # early stop if not improving (loss stabilization)
    if epochs_no_improve >= patience:
        print("Early stopping triggered", flush=True)
        break


# -----------------------------
# TRAINING PLOT - Loss, Pearson R
# -----------------------------

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='train')
ax1.plot(history['val_loss'],   label='val')
ax1.set_xlabel('epoch')
ax1.set_ylabel('noise MSE')
ax1.set_title('Training loss')
ax1.legend()

epochs_corr, corrs = zip(*history['val_corr'])
ax2.plot(epochs_corr, corrs, marker='o')
ax2.set_xlabel('epoch')
ax2.set_ylabel('Pearson r')
ax2.set_title('Generation quality (val)')

plt.tight_layout()
plt.savefig(f"{OUTDIR}/training_curves.png", dpi=150)
#plt.show()


# -----------------------------
# TEST BLOCK - Pearson R (generation quality proxy)
# -----------------------------
model.load_state_dict(torch.load(f"{OUTDIR}/best_10k_model.pt"))

model.eval()
corrs = []
with torch.no_grad():
    for batch in test_loader:
        z = batch["rna_emb"].to(device)
        atac = torch.log1p(batch["atac"].to(device))
        generated = diffusion.sample(model, z, atac.shape[1], n_samples=z.shape[0])
        for i in range(atac.shape[0]):
            r = torch.corrcoef(torch.stack([generated[i], atac[i]]))[0, 1]
            corrs.append(r.item())

print(f"Test Pearson r: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}", flush=True)

np.save(f"{OUTDIR}/test_corrs.npy", np.array(corrs))

# -----------------------------
# ADDITIONAL EVAL TO MATCH MLP + TEST ON HELD OUT DATA (unseen cell type)
# -----------------------------

# note: for some plots, randomly sampled 500 cells and used 50 DDIM steps to denoise for less computational cost

def evaluate(loader):
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            z = batch["rna_emb"].to(device)
            atac = torch.log1p(batch["atac"].to(device))

            generated = diffusion.sample(model, z, atac.shape[1], n_samples=z.shape[0])

            preds_all.append(generated.cpu().numpy())
            targets_all.append(atac.cpu().numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    return (
        weighted_mse(targets_all, preds_all, weights),
        r2_score(targets_all, preds_all)
    )

test_mse, test_r2 = evaluate(test_loader)
held_mse, held_r2 = evaluate(held_loader)

print("\n Diffusion Model Performance:", flush=True)
print(f"Test MSE: {test_mse}", flush=True)
print(f"Test R2: {test_r2}", flush=True)
print(f"Held-out MSE: {held_mse}", flush=True)
print(f"Held-out R2: {held_r2}", flush=True)
print(model, flush=True)

preds = []
targets = []
model.eval()

with torch.no_grad():
    for batch in val_loader:
        x = batch["rna_emb"].to(device)
        y = torch.log1p(batch["atac"].to(device))

        pred = diffusion.sample(model, x, y.shape[1], n_samples=x.shape[0])

        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

preds = np.vstack(preds)
targets = np.vstack(targets)

n_cells, n_peaks = preds.shape

cell_idx = np.random.choice(n_cells, size=min(500, n_cells), replace=False)
peak_idx = np.random.choice(n_peaks, size=min(5000, n_peaks), replace=False)

true_sample = targets[np.ix_(cell_idx, peak_idx)].flatten()
pred_sample = preds[np.ix_(cell_idx, peak_idx)].flatten()

plt.figure(figsize=(6,6))
plt.scatter(true_sample, pred_sample, alpha=0.1, s=3)
plt.xlabel("True log1p(ATAC)")
plt.ylabel("Predicted")
plt.title("Predicted vs True")
lo = min(true_sample.min(), pred_sample.min())
hi = max(true_sample.max(), pred_sample.max())
plt.plot([lo, hi], [lo, hi], 'r--')
plt.savefig(f"{OUTDIR}/predicted_true.png", dpi=150)


residuals = (preds - targets).flatten()
plt.figure()
plt.hist(residuals, bins=100)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Count")
#plt.show()
plt.savefig(f"{OUTDIR}/residuals.png", dpi=150)

from sklearn.metrics import r2_score

r2_per_peak = [
    r2_score(targets[:, i], preds[:, i])
    for i in range(targets.shape[1])
]
plt.figure()
plt.hist(r2_per_peak, bins=100)
plt.title("Per-peak R² distribution")
plt.xlabel("R²")
plt.ylabel("Number of peaks")
#plt.show()
plt.savefig(f"{OUTDIR}/r2_per_peak.png", dpi=150)

peak_var = np.var(targets, axis=0)
plt.figure()
plt.scatter(peak_var, r2_per_peak, alpha=0.3, s=5)
plt.xlabel("Peak variance")
plt.ylabel("R²")
plt.title("R² vs Variance")
#plt.show()
plt.savefig(f"{OUTDIR}/r2_variance.png", dpi=150)

mean_true = targets.mean(axis=0)
mean_pred = preds.mean(axis=0)
plt.figure()
plt.scatter(mean_true, mean_pred, alpha=0.3, s=5)
plt.xlabel("Mean true")
plt.ylabel("Mean predicted")
plt.title("Mean ATAC profile")
plt.plot([mean_true.min(), mean_true.max()], [mean_true.min(), mean_true.max()], 'r--')
#plt.show()
plt.savefig(f"{OUTDIR}/mean_profile.png", dpi=150)
