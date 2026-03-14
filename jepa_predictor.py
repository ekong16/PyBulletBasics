import os
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob, math, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import time
import numpy as np

# --- 1. CONFIGURATION ---
LATENT_DIR = "world_model_latents"
BATCH_SIZE = 2  # [B] - Keep small to spare your RAM
ACCUM_BACKWARDS_STEPS = 16
EPOCHS = 120
LR = 3e-4
DEVICE = torch.device("mps")
DEVICE_STR = "mps"


# --- 2. DATA LOADER ---
class JEPADataset(Dataset):
    def __init__(self, path):
        self.files = sorted(glob.glob(os.path.join(path, "**/*.pt"), recursive=True))

        # self.files = self.files[:400]

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
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),  # The Peacekeeper
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
        # nn.init.zeros_(self.output_head.weight)
        # nn.init.zeros_(self.output_head.bias)

        # The neural network will dynamically adjust this single parameter to find the right "volume"
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

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
        x = x + act_token
        combined = torch.cat([act_token, x], dim=1)

        # action broadcast to all tokens [B, 2048, 256] + [B, 1, 256] -> [B, 2048, 256]
        # combined = x + act_token

        # 4. Fast Transformer Processing (Now doing 16x less math)
        # -> [B, 2049, 256]
        out = self.transformer(combined)

        # 5. Extract Video, DECOMPRESS, and Add to Original
        # Extract: [B, 2048, 256]
        # Decompress: -> [B, 2048, 1024]
        delta = self.output_head(out[:, 1:, :])

        # Return the final full-size prediction
        return z0 + (delta * self.delta_scale)


# --- 4. TRAINING LOOP ---
def train():
    dataset = JEPADataset(LATENT_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 1. Instantiate Both Models
    model = WorldPredictorPro()
    # 2. Print the summary safely on the CPU
    print("\n" + "=" * 50)
    summary(
        model,
        input_size=[(BATCH_SIZE, 2048, 1024), (BATCH_SIZE, 28)],
        depth=3,
    )
    print("=" * 50 + "\n")

    model = model.to(DEVICE)

    # 2. Two Separate Wallets (Optimizers)
    # We give the Inverse Net a slightly higher LR so it stays smarter than the Predictor
    opt_predictor = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Anneal both schedules
    sched_predictor = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_predictor, T_max=EPOCHS, eta_min=1e-6
    )

    l1_criterion = nn.L1Loss()
    # mse_criterion = nn.MSELoss()  # Actions usually use MSE
    # criterion_inv = nn.SmoothL1Loss(beta=0.1)

    start_training_time = time.time()
    print(f"🚀 Training on {len(dataset)} transitions using {DEVICE}")

    best_pred_imp = 0
    for ep in range(EPOCHS):
        model.train()

        total_loss_backprop, total_loss_pred_only, total_base_pure_diff = 0, 0, 0

        steps_taken = 0
        for z0, act, z1 in loader:
            z0, act, z1 = z0.to(DEVICE), act.to(DEVICE), z1.to(DEVICE)

            # ==========================================
            # PHASE 2: TRAIN THE PREDICTOR (Cycle Loss)
            # ==========================================

            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                # Prediction and Raw element wise error
                z1_pred = model(z0, act)
                raw_l1 = torch.abs(z1_pred - z1)

                # B. Build the Searchlight Mask
                # We use no_grad() because we don't want to calculate gradients for the mask itself
                with torch.no_grad():
                    # Calculate how much each token ACTUALLY changed in the simulator
                    # [Batch, 2048, 1] represents average change of each token.
                    motion_magnitude = torch.abs(z1 - z0).mean(dim=-1, keepdim=True)

                    # Base weight is 1.0. Moving parts get multiplied by up to 50x.
                    weight_mask = 1.0 + (motion_magnitude * 50.0)

                # C. Supercharge the Backprop
                # multiply the change ... tokens that change more have their l1 score multiplied by 50x....
                loss_backprop = (raw_l1 * weight_mask).mean()

                # D. Honest Telemetry
                # We calculate the normal unweighted L1 just for print statements,
                fwd_loss_pred_only = raw_l1.mean()

                # -----------------------------

            # (loss_backprop).backward() will now use the supercharged gradients!
            (loss_backprop).backward()

            # OLD LOSS CALC
            # # 1. Predict the Hallucination
            # z1_pred = model(z0, act)
            # fwd_loss_pred_only = l1_criterion(z1_pred, z1)

            # loss_backprop = fwd_loss_pred_only

            # print("INPUTS", z0, act, z1)
            # print("OUTPUTS", fwd_loss_pred_only, pred_act_fake, pred_act_real)

            # (loss_backprop).backward()

            if (steps_taken + 1) % ACCUM_BACKWARDS_STEPS == 0:
                # Update only after certain number of steps
                opt_predictor.step()
                opt_predictor.zero_grad()

            # Tracking
            steps_taken += 1
            total_loss_backprop += loss_backprop.item()
            total_loss_pred_only += fwd_loss_pred_only.item()
            total_base_pure_diff += l1_criterion(z0, z1).item()

            # break

        # Flush any remaining accumulated gradients at the end of the epoch
        if steps_taken % ACCUM_BACKWARDS_STEPS != 0:
            opt_predictor.step()
            opt_predictor.zero_grad()

        # Step both schedules
        sched_predictor.step()
        current_lr = sched_predictor.get_last_lr()[0]

        # ==========================================
        # TELEMETRY & REPORTING
        # ==========================================
        # print("STEPS TAKEN", steps_taken)
        avg_loss_backprop = total_loss_backprop / steps_taken
        avg_loss_pred_only = total_loss_pred_only / steps_taken
        avg_base_pure_diff = total_base_pure_diff / steps_taken

        # 1. Predictor Improvement (vs. doing nothing)
        pred_imp = (
            (avg_base_pure_diff - avg_loss_pred_only) / avg_base_pure_diff
        ) * 100
        if pred_imp > best_pred_imp:
            checkpoint = {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": opt_predictor.state_dict(),
                "scheduler_state": sched_predictor.state_dict(),  # Keeps the Cosine curve correct
                "best_pred_imp": best_pred_imp,
                "pred_imp": pred_imp,
                "scale": model.delta_scale.item(),  # Useful for logging later
            }

            best_pred_imp = pred_imp
            pct_str = f"{pred_imp:.2f}".replace(".", "_") + "_pct"
            save_path = os.path.join("predictor_weights", f"world_model_{pct_str}.pth")

            torch.save(checkpoint, save_path)
            # print(f"🌟 New Best! Saved weights at {pred_imp:+.1f}% improvement.")

        # Time Metrics
        current_time = time.time()
        elapsed_total = current_time - start_training_time
        total_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
        now = datetime.now().replace(microsecond=0).strftime("%H:%M:%S")

        print(
            f"Epoch {ep:03d} | "
            f"Loss_BackProp: {avg_loss_backprop:7.5f} | "
            f"Base: {avg_base_pure_diff:7.5f} | "
            f"Loss_Pred: {avg_loss_pred_only:7.5f} ({pred_imp:>+5.1f}%) | "
            f"Elapsed: {total_str} | "
            f"Time: {now} | "
            f"LR_m: {current_lr:.5f} | "
            f"Scale: {model.delta_scale.item():.4f}"
        )

    # Save a combined checkpoint so you don't lose the auditor's brain
    checkpoint = {
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, "world_model_masked.pth")
    print("✅ Model weights saved to world_model_with_cycle.pth")


if __name__ == "__main__":
    train()
