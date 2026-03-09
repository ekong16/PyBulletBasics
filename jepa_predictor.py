import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob, math, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import time

# --- 1. CONFIGURATION ---
LATENT_DIR = "world_model_latents"
BATCH_SIZE = 2  # [B] - Keep small to spare your RAM
EPOCHS = 30
LR = 2e-4
DEVICE = torch.device("mps")
DEVICE_STR = "mps"


# --- 2. DATA LOADER ---
class JEPADataset(Dataset):
    def __init__(self, path):
        self.files = sorted(glob.glob(os.path.join(path, "**/*.pt"), recursive=True))
        import numpy as np

        torch.serialization.add_safe_globals(
            [np._core.multiarray._reconstruct, np.ndarray, np.dtype]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        # z0: [2048, 1024] | act: [28] | z1: [2048, 1024]
        return d["state_0"].float(), d["action"].float(), d["state_1"].float()


# --- 3. THE WORLD PREDICTOR ---
class WorldPredictorPro(nn.Module):
    def __init__(self, latent_dim=1024, action_dim=28, num_tokens=2048, hidden_dim=512):
        super().__init__()

        # 1. THE INTERNAL GUARD (Normalization)
        # It looks at the 1024 features of each token and centers them
        self.input_norm = nn.LayerNorm(latent_dim)

        # A. THE MAP (Stays the same, applied to the raw 1024-dim video)
        self.register_buffer("pos_embed", self._get_pos_embed(num_tokens, latent_dim))

        # B. THE COMPRESSOR (The secret to speed)
        # Squashes the heavy 1024-dim tokens down to a nimble 256 dims
        self.state_compressor = nn.Linear(latent_dim, hidden_dim)

        # C. THE ACTION TRANSLATOR
        # Now it only needs to project the 28 torques to 256 dims to match the compressed state
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128), nn.GELU(), nn.Linear(128, hidden_dim)
        )

        # D. THE LIGHTWEIGHT TRANSFORMER
        # - d_model is 256 (4x smaller = 16x less memory for attention matrices)
        # - nhead is 4 (down from 8)
        # - dim_feedforward is 1024 (down from 2048)
        # - num_layers is 2 (down from 6. We don't need 6 layers to learn basic physics!)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

        # E. THE DECOMPRESSOR (Output Head)
        # Blows the 256-dim prediction back up to 1024 dims so it matches your V-JEPA target
        self.output_head = nn.Linear(hidden_dim, latent_dim)

    def _get_pos_embed(self, num_tokens, latent_dim):
        """Clean, standard PyTorch implementation of the Sin/Cos Barcodes."""
        pe = torch.zeros(1, num_tokens, latent_dim)  # [1, 2048, 1024]
        pos = torch.arange(num_tokens).unsqueeze(1)  # [2048, 1]
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim)
        )  # [512]

        pe[0, :, 0::2] = torch.sin(pos * div_term)  # Even columns
        pe[0, :, 1::2] = torch.cos(pos * div_term)  # Odd columns
        return pe

    def forward(self, z0, action):
        # z0: [B, 2048, 1024] | action: [B, 28]

        z0_norm = self.input_norm(z0)

        # 1. Add GPS Map BEFORE shrinking: [B, 2048, 1024]
        # We want the positional data stamped at full resolution
        x = z0_norm + self.pos_embed.to(device=z0.device, dtype=z0.dtype)

        # NEW STEP: Compress the Video State
        # [B, 2048, 1024] -> [B, 2048, 256]
        x = self.state_compressor(x)

        # 2. Translate Action to the new smaller dimension
        # [B, 28] -> [B, 256] -> [B, 1, 256]
        act_token = self.action_encoder(action).unsqueeze(1)

        # 3. Combine: [B, 1, 256] + [B, 2048, 256] -> [B, 2049, 256]
        # combined = torch.cat([act_token, x], dim=1)

        # action broadcast to all tokens [B, 2048, 256] + [B, 1, 256] -> [B, 2048, 256]
        combined = x + act_token

        # 4. Fast Transformer Processing (Now doing 16x less math)
        # -> [B, 2049, 256]
        out = self.transformer(combined)

        # 5. Extract Video, DECOMPRESS, and Add to Original
        # Extract: [B, 2048, 256]
        # Decompress: -> [B, 2048, 1024]
        delta = self.output_head(out[:, :, :])

        # Return the final full-size prediction
        return z0 + delta


# --- 4. TRAINING LOOP ---
def train():
    dataset = JEPADataset(LATENT_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = WorldPredictorPro().to(DEVICE)
    summary(
        model, input_size=[(BATCH_SIZE, 2048, 1024), (BATCH_SIZE, 28)], device=DEVICE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.L1Loss()

    start_training_time = time.time()  # Start the "Total Elapsed" clock
    print(f"🚀 Training on {len(dataset)} transitions using {DEVICE}...")

    for ep in range(EPOCHS):
        model.train()
        total_loss, total_base = 0, 0

        for z0, act, z1 in loader:
            z0, act, z1 = z0.to(DEVICE), act.to(DEVICE), z1.to(DEVICE)

            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                optimizer.zero_grad()
                pred = model(z0, act)
                loss = criterion(pred, z1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_base += criterion(z0, z1).item()

        # Calculation Phase
        if ep % 5 == 0:
            avg_loss = total_loss / len(loader)
            avg_base = total_base / len(loader)
            improvement = ((avg_base - avg_loss) / avg_base) * 100

            # Time Metrics
            current_time = time.time()
            elapsed_total = current_time - start_training_time
            # We average the time since the start over the number of completed epochs
            avg_epoch_time = elapsed_total / (ep + 1)

            # Format times for readability (MM:SS)
            total_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
            avg_str = f"{avg_epoch_time:.2f}s"

            print(
                f"Epoch {ep:03d} | Loss: {avg_loss:.5f} | Base: {avg_base:.5f} | "
                f"Progress: {improvement:.1f}% | Avg Epoch: {avg_str} | Total: {total_str}"
            )

    torch.save(model.state_dict(), "world_model_final.pth")
    print("✅ Model weights saved to world_model_final.pth")


if __name__ == "__main__":
    train()
