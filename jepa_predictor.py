import os
from datetime import datetime

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
ACCUM_BACKWARDS_STEPS = 16
EPOCHS = 60
LR = 3e-4
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
    def __init__(self, latent_dim=1024, action_dim=28, num_tokens=2048, hidden_dim=256):
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
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

        # E. THE DECOMPRESSOR (Output Head)
        # Blows the 256-dim prediction back up to 1024 dims so it matches your V-JEPA target
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

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


# --- 4. THE AUDITOR (New) ---
class InverseDynamicsNet(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, action_dim=28):
        super().__init__()
        # Tiny MLP that runs in milliseconds
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, z_0, z_1):
        # 1. Isolate the movement
        delta_z = z_1 - z_0
        # 2. Delta-Max: Find the strongest signal, drop the background
        z_pooled = delta_z.max(dim=1).values
        # 3. Guess the action
        return self.mlp(z_pooled)


class TransformerInverseNet(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256, num_queries=8, action_dim=28):
        super().__init__()

        # 1. THE COMPRESSOR (Squash 1024 -> 256 to save Mac memory)
        self.compressor = nn.Linear(feature_dim * 2, hidden_dim)

        # 2. THE 8 DETECTIVES (Learnable Queries)
        self.readout_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # 3. THE MATCHER & EXTRACTOR (Cross-Attention)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        # 4. THE BRAIN (Translates the 8 tokens into 28 torques)
        # 8 queries * 256 dims = 2048 total inputs
        mlp_input_dim = num_queries * hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, z_0, z_1):
        # INPUT SHAPES:
        # z_0: [Batch, 2048, 1024] (The starting video frame latents)
        # z_1: [Batch, 2048, 1024] (The ending video frame latents)

        # 1. Delete the static background room
        # delta_z = z_1 - z_0
        z_cat = torch.cat([z_0, z_1], dim=-1)
        # SHAPE: [Batch, 2048, 1024]
        # (Subtraction doesn't change the size, just the values)

        # 2. Compress the video patches to save RAM
        x = self.compressor(z_cat)
        # SHAPE: [Batch, 2048, 256]
        # (The 1024 features are squashed down to 256)

        # 3. Photocopy the 8 Detectives for however many videos are in the batch
        B = x.size(0)
        q = self.readout_queries.expand(B, -1, -1)
        # SHAPE: [Batch, 8, 256]
        # (We expanded the '1' into 'B'. Now we have B sets of 8 detectives, each with a 256-D brain)

        # 4. Match and Extract (The magic line)
        # Query (q) searches Key (x), and extracts from Value (x)
        attn_out, _ = self.attention(query=q, key=x, value=x)
        # SHAPE: [Batch, 8, 256]
        # (The 2048 patches have been collapsed. Only the 8 extracted evidence tokens remain)

        # 5. Lay the 8 tokens end-to-end into a single flat line
        z_pooled = attn_out.flatten(start_dim=1)
        # SHAPE: [Batch, 2048]
        # (8 tokens * 256 features = 2048. We flattened the grid into a single 1D array for the MLP)

        # 6. Predict the 28 joint torques
        actions_pred = self.mlp(z_pooled)
        # SHAPE OUTPUT: [Batch, 28]
        # (The MLP reads the 2048 evidence array and outputs exactly 28 motor torques)

        return actions_pred


# --- 4. TRAINING LOOP ---
def train():
    dataset = JEPADataset(LATENT_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 1. Instantiate Both Models
    model = WorldPredictorPro().to(DEVICE)
    inverse_net = TransformerInverseNet().to(DEVICE)

    # 2. Two Separate Wallets (Optimizers)
    # We give the Inverse Net a slightly higher LR so it stays smarter than the Predictor
    opt_predictor = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    opt_inverse = torch.optim.AdamW(
        inverse_net.parameters(), lr=LR * 3, weight_decay=0.01
    )

    # Anneal both schedules
    sched_predictor = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_predictor, T_max=EPOCHS, eta_min=1e-5
    )
    sched_inverse = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_inverse, T_max=EPOCHS, eta_min=1e-5
    )

    l1_criterion = nn.L1Loss()
    # mse_criterion = nn.MSELoss()  # Actions usually use MSE
    criterion_inv = nn.SmoothL1Loss(beta=0.1)

    start_training_time = time.time()
    print(
        f"🚀 Training on {len(dataset)} transitions using {DEVICE} with Inverse Cycle..."
    )

    for ep in range(EPOCHS):
        model.train()
        inverse_net.train()

        (
            total_loss_pred_only,
            total_loss_cyc_fake_only,
            total_loss_inv,
            total_base_pure_diff,
            total_loss_backprop,
        ) = 0, 0, 0, 0, 0

        steps_taken = 0
        for z0, act, z1 in loader:
            z0, act, z1 = z0.to(DEVICE), act.to(DEVICE), z1.to(DEVICE)
            # ==========================================
            # PHASE 1: TRAIN THE DETECTIVE (Reality Only)
            # ==========================================
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                # .detach() is critical here to prevent cross-contamination
                pred_act_real = inverse_net(z0, z1)
                inv_loss = criterion_inv(pred_act_real, act)

            (inv_loss / ACCUM_BACKWARDS_STEPS).backward()

            if (steps_taken + 1) % ACCUM_BACKWARDS_STEPS == 0:
                # Update only after certain number of steps
                opt_inverse.step()
                opt_inverse.zero_grad()

            # ==========================================
            # PHASE 2: TRAIN THE PREDICTOR (Cycle Loss)
            # ==========================================
            # 1. Freeze the Teacher so it doesn't learn from hallucinations
            for param in inverse_net.parameters():
                param.requires_grad = False

            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                # 1. Predict the Hallucination
                z1_pred = model(z0, act)
                fwd_loss_pred_only = l1_criterion(z1_pred, z1)

                # 2. Interrogate the Hallucination
                # No .detach() on z1_pred because gradients MUST flow back to the Predictor
                pred_act_fake = inverse_net(z0, z1_pred)
                cyc_loss = criterion_inv(pred_act_fake, act)

                # 3. The 0.5 Leash
                loss_backprop = fwd_loss_pred_only + (0.5 * cyc_loss)

            # print("INPUTS", z0, act, z1)
            # print("OUTPUTS", fwd_loss_pred_only, pred_act_fake, pred_act_real)

            (loss_backprop / ACCUM_BACKWARDS_STEPS).backward()
            if (steps_taken + 1) % ACCUM_BACKWARDS_STEPS == 0:
                # Update only after certain number of steps
                opt_predictor.step()
                opt_predictor.zero_grad()

            # 2. Unfreeze the Teacher for the next loop's Phase 1
            for param in inverse_net.parameters():
                param.requires_grad = True

            # Tracking
            steps_taken += 1
            total_loss_inv += inv_loss.item()
            total_loss_pred_only += fwd_loss_pred_only.item()
            total_loss_cyc_fake_only += cyc_loss.item()
            total_loss_backprop += loss_backprop.item()
            total_base_pure_diff += l1_criterion(z0, z1).item()

            # break
        # Flush any remaining accumulated gradients at the end of the epoch
        if steps_taken % ACCUM_BACKWARDS_STEPS != 0:
            opt_inverse.step()
            opt_predictor.step()
            opt_inverse.zero_grad()
            opt_predictor.zero_grad()

        # Step both schedules
        sched_predictor.step()
        sched_inverse.step()
        current_lr = sched_predictor.get_last_lr()[0]
        current_lr_inv = sched_inverse.get_last_lr()[0]

        # ==========================================
        # TELEMETRY & REPORTING
        # ==========================================
        # print("STEPS TAKEN", steps_taken)
        avg_loss_inv = total_loss_inv / steps_taken
        avg_loss_pred_only = total_loss_pred_only / steps_taken
        avg_loss_cyc_fake_only = total_loss_cyc_fake_only / steps_taken
        avg_base_pure_diff = total_base_pure_diff / steps_taken
        avg_loss_backprop = total_loss_backprop / steps_taken

        # 1. Predictor Improvement (vs. doing nothing)
        pred_imp = (
            (avg_base_pure_diff - avg_loss_pred_only) / avg_base_pure_diff
        ) * 100

        # 2. Cycle Consistency Improvement (vs. 0.333 random guessing)
        #        cyc_imp = ((0.333 - avg_loss_cyc_fake_only) / 0.333) * 100
        cyc_imp = ((0.5 - avg_loss_cyc_fake_only) / 0.5) * 100

        # 3. Inverse Net Improvement (vs. 0.333 random guessing)
        #        inv_imp = ((0.333 - avg_loss_inv) / 0.333) * 100
        inv_imp = ((0.5 - avg_loss_inv) / 0.5) * 100

        # Time Metrics
        current_time = time.time()
        elapsed_total = current_time - start_training_time
        total_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
        now = datetime.now().replace(microsecond=0).strftime("%H:%M:%S")

        print(
            f"Epoch {ep:03d} | "
            f"Loss_Backprop: {avg_loss_backprop:7.5f} | "
            f"Base: {avg_base_pure_diff:7.5f} | "
            f"Loss_Pred: {avg_loss_pred_only:7.5f} ({pred_imp:>+5.1f}%) | "
            f"Loss_Cyc: {avg_loss_cyc_fake_only:7.5f} ({cyc_imp:>+5.1f}%) | "
            f"Inv Loss: {avg_loss_inv:7.5f} ({inv_imp:>+5.1f}%) | "
            f"Elapsed: {total_str} | "
            f"Time: {now} | "
            f"LR_m: {current_lr:.5f} | "
            f"LR_i: {current_lr_inv:.5f}"
        )

    # Save a combined checkpoint so you don't lose the auditor's brain
    checkpoint = {
        "model_state": model.state_dict(),
        "inverse_state": inverse_net.state_dict(),
    }
    torch.save(checkpoint, "world_model_with_cycle.pth")
    print("✅ Model weights saved to world_model_with_cycle.pth")


if __name__ == "__main__":
    train()
