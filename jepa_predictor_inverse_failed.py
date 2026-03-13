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
    def __init__(
        self,
        feature_dim=1024,
        hidden_dim=256,
        num_tokens=2048,
        num_queries=32,
        action_dim=28,
    ):
        super().__init__()

        # 1. THE MAP (1024 dims - perfectly matches the raw JEPA latents)
        self.register_buffer("pos_embed", self._get_pos_embed(num_tokens, feature_dim))

        # 2. THE INTERNAL GUARD (Normalizes the concatenated 2048-dim vector)
        self.input_norm = nn.LayerNorm(feature_dim * 2)

        # 3. THE COMPRESSOR (Shrink 2048 -> 256)
        self.compressor = nn.Linear(feature_dim * 2, hidden_dim)

        # 4. THE 8 DETECTIVES
        self.readout_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # 5. THE ATTENTION
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        # 3. THE HOURGLASS BRAIN (The new Learnt Bottleneck)
        self.mlp = nn.Sequential(
            # Phase 1: Process the visual consensus
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            # Phase 2: THE BOTTLENECK (Force it to find the 8 Macro Commands)
            nn.Linear(128, 8),
            nn.GELU(),
            # Phase 3: THE EXPANSION (Translate the 8 commands into 28 physical torques)
            nn.Linear(8, action_dim),
            # OPTIONAL BUT HIGHLY RECOMMENDED:
            # If your ground truth actions are ALWAYS between -1.0 and 1.0, uncomment this!
            nn.Tanh(),
        )

    def _get_pos_embed(self, num_tokens, dim):
        pe = torch.zeros(1, num_tokens, dim)
        pos = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, z_0, z_1):
        # Calculate raw diff before PE
        z_diff = z_1 - z_0

        # Stamp both (GPS for state, GPS for motion)
        z_0_s = z_0 + self.pos_embed.to(device=z_0.device, dtype=z_0.dtype)
        z_d_s = z_diff + self.pos_embed.to(device=z_diff.device, dtype=z_diff.dtype)

        # Concat State + Velocity
        z_cat = torch.cat([z_0_s, z_d_s], dim=-1)

        # Below is code to pass in both z0 and z1
        # # 1. Stamp GPS coordinates on both frames INDEPENDENTLY
        # # This aligns them perfectly in physical space
        # z_0_stamped = z_0 + self.pos_embed.to(device=z_0.device, dtype=z_0.dtype)
        # z_1_stamped = z_1 + self.pos_embed.to(device=z_1.device, dtype=z_1.dtype)

        # # 2. Concatenate the spatially-aligned frames
        # z_cat = torch.cat([z_0_stamped, z_1_stamped], dim=-1)

        # 3. Normalize to protect the Attention mechanism from massive variance
        x = self.input_norm(z_cat)

        # 4. Compress 2048 -> 256
        x = self.compressor(x)

        # 5. Extract Evidence using the 8 Detectives
        B = x.size(0)
        q = self.readout_queries.expand(B, -1, -1)
        attn_out, _ = self.attention(query=q, key=x, value=x)

        query_votes = self.mlp(attn_out)
        actions_pred = query_votes.mean(dim=1)

        return actions_pred
        # # 6. Flatten and Predict
        # z_pooled = attn_out.flatten(start_dim=1)


class InversePredictorPro(nn.Module):
    def __init__(self, latent_dim=1024, action_dim=28, num_tokens=2048, hidden_dim=256):
        super().__init__()

        # 1. THE MAP & COMPRESSOR (Exactly like Main Net)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.register_buffer("pos_embed", self._get_pos_embed(num_tokens, latent_dim))

        # We compress the 1024-dim features down to 256
        self.state_compressor = nn.Linear(latent_dim, hidden_dim)
        self.diff_compressor = nn.Linear(latent_dim, hidden_dim)

        # 2. THE ACTION TOKEN (The "Blank Check")
        # Instead of 32 blind queries, we use 1 special token that flows WITH the image
        self.action_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 3. THE LIGHTWEIGHT TRANSFORMER (Exactly like Main Net)
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

        # 4. THE HOURGLASS BRAIN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 8),
            nn.GELU(),
            nn.Linear(8, action_dim),
            nn.Tanh(),
        )

    def _get_pos_embed(self, num_tokens, latent_dim):
        pe = torch.zeros(1, num_tokens, latent_dim)
        pos = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim)
        )
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, z_0, z_1):
        B = z_0.size(0)

        # 1. State + Velocity Logic
        z_diff = z_1 - z_0

        z_0_norm = self.input_norm(z_0)
        z_d_norm = self.input_norm(z_diff)

        # 2. Add GPS Map
        z_0_stamped = z_0_norm + self.pos_embed.to(device=z_0.device, dtype=z_0.dtype)
        z_d_stamped = z_d_norm + self.pos_embed.to(
            device=z_diff.device, dtype=z_diff.dtype
        )

        # 3. Compress to 256
        x_0 = self.state_compressor(z_0_stamped)
        x_d = self.diff_compressor(z_d_stamped)

        # 4. Combine into one sequence: [B, 2048, 256]
        # We add the state and diff together so the sequence length stays 2048 (fast!)
        x_combined = x_0 + x_d

        # 5. Prepend the Action Token
        # [B, 1, 256] cat with [B, 2048, 256] -> [B, 2049, 256]
        act_t = self.action_token.expand(B, -1, -1)
        seq = torch.cat([act_t, x_combined], dim=1)

        # 6. Process with Self-Attention (The patches finally talk to each other)
        out = self.transformer(seq)

        # 7. Extract the processed Action Token (it's at index 0) and decode it
        # [B, 1, 256] -> [B, 256] -> Hourglass -> [B, 28]
        pred_action = self.mlp(out[:, 0, :])

        return pred_action


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
    # criterion_inv = nn.SmoothL1Loss(beta=0.1)
    criterion_inv = nn.L1Loss()

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

            (inv_loss).backward()

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

            (loss_backprop).backward()
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


def train_teacher_only():
    dataset = JEPADataset(LATENT_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    inverse_net = InversePredictorPro()

    # 2. Print the summary safely on the CPU
    print("\n" + "=" * 50)
    summary(
        inverse_net,
        input_size=[(BATCH_SIZE, 2048, 1024), (BATCH_SIZE, 2048, 1024)],
        depth=3,
    )
    print("=" * 50 + "\n")

    # 3. NOW move it to the Mac's GPU for actual training
    inverse_net = inverse_net.to(DEVICE)
    # 2. Optimizer & Scheduler
    opt_inverse = torch.optim.AdamW(
        inverse_net.parameters(), lr=LR * 3, weight_decay=0.01
    )
    sched_inverse = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_inverse, T_max=EPOCHS, eta_min=1e-5
    )

    # Trusting your gut: Huber/SmoothL1 is the way to go
    # criterion_inv = nn.SmoothL1Loss(beta=0.1)
    criterion_inv = nn.L1Loss()
    # criterion_inv = nn.MSELoss()
    start_training_time = time.time()
    print(
        f"🚀 ISOLATED TEACHER TRAINING on {len(dataset)} transitions using {DEVICE}..."
    )

    for ep in range(EPOCHS):
        inverse_net.train()

        total_loss_inv = 0
        steps_taken = 0

        for z0, act, z1 in loader:
            z0, act, z1 = z0.to(DEVICE), act.to(DEVICE), z1.to(DEVICE)

            # ==========================================
            # ISOLATED PHASE: TRAIN THE DETECTIVE
            # ==========================================
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                pred_act_real = inverse_net(z0, z1)

                # --- DIAGNOSTIC CHECK (Runs only on Epoch 0, Batch 0) ---
                if ep == 0 and steps_taken == 0:
                    print(f"🔍 DEBUG SIGNAL CHECK:")
                    print(
                        f"   Ground Truth Action Range: min={act.min().item():.5f}, max={act.max().item():.5f}"
                    )
                    print(
                        f"   Teacher Prediction Range : min={pred_act_real.min().item():.5f}, max={pred_act_real.max().item():.5f}"
                    )
                    print(
                        f"   Initial Loss: {criterion_inv(pred_act_real, act).item():.5f}\n"
                    )

                inv_loss = criterion_inv(pred_act_real, act)

            # Backprop (Keeping your un-divided loud gradient)
            (inv_loss).backward()

            if (steps_taken + 1) % ACCUM_BACKWARDS_STEPS == 0:
                opt_inverse.step()
                opt_inverse.zero_grad()

            # Tracking
            steps_taken += 1
            total_loss_inv += inv_loss.item()

        # Flush any remaining accumulated gradients at the end of the epoch
        if steps_taken % ACCUM_BACKWARDS_STEPS != 0:
            opt_inverse.step()
            opt_inverse.zero_grad()

        # Step schedule
        sched_inverse.step()
        current_lr_inv = sched_inverse.get_last_lr()[0]

        # ==========================================
        # TELEMETRY & REPORTING
        # ==========================================
        avg_loss_inv = total_loss_inv / steps_taken
        inv_imp = ((0.5 - avg_loss_inv) / 0.5) * 100

        # Time Metrics
        current_time = time.time()
        elapsed_total = current_time - start_training_time
        total_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
        now = datetime.now().replace(microsecond=0).strftime("%H:%M:%S")

        print(
            f"Epoch {ep:03d} | "
            f"Inv Loss: {avg_loss_inv:7.5f} ({inv_imp:>+5.1f}%) | "
            f"Elapsed: {total_str} | "
            f"Time: {now} | "
            f"LR_i: {current_lr_inv:.5f}"
        )

    torch.save(inverse_net.state_dict(), "inverse_net_isolated.pth")
    print("✅ Teacher weights saved to inverse_net_isolated.pth")


if __name__ == "__main__":
    # train()
    train_teacher_only()
