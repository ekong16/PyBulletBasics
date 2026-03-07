import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import torch
import numpy as np
import utils  # Your custom JEPA wrapper

# ==========================================
# 1. CONFIGURATION
# ==========================================
SOURCE_DIR = "world_model_dataset"
DEST_DIR = "world_model_latents"
USE_HALF_PRECISION = True  # Reduces disk footprint by 50%


def estimate_footprint(num_transitions):
    """Calculates expected disk usage for the full [2048, 1024] latents."""
    bytes_per_float = 2 if USE_HALF_PRECISION else 4
    # (State0 + State1) * Tokens * EmbedDim * Bytes
    mb_per_file = (2 * 2048 * 1024 * bytes_per_float) / (1024 * 1024)
    total_gb = (mb_per_file * num_transitions) / 1024
    return mb_per_file, total_gb


def encode_high_fidelity_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: {SOURCE_DIR} not found.")
        return

    # 1. Scan for work
    episode_folders = sorted(glob.glob(os.path.join(SOURCE_DIR, "episode_*")))
    all_files = []
    for ep in episode_folders:
        all_files.extend(glob.glob(os.path.join(ep, "transition_*.pt")))

    num_files = len(all_files)
    mb_per, total_gb = estimate_footprint(num_files)

    print(f"📊 DATA AUDIT:")
    print(f"   - Transitions found: {num_files}")
    print(f"   - Size per file: ~{mb_per:.1f} MB")
    print(f"   - Total expected footprint: ~{total_gb:.2f} GB")
    print("-" * 30)

    # 2. Initialize V-JEPA
    print(f"🚀 Spinning up V-JEPA Engine...")
    jepa_engine = utils.JEPAEngine()

    os.makedirs(DEST_DIR, exist_ok=True)
    total_encoded = 0

    for trans_path in all_files:
        print("WORKING ON: ", trans_path)
        # Create mirrored path: world_model_latents/episode_X/transition_Y.pt
        relative_path = os.path.relpath(trans_path, SOURCE_DIR)
        dest_path = os.path.join(DEST_DIR, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # --- RESUME LOGIC ---
        if os.path.exists(dest_path):
            continue

        # 3. Load and Encode
        try:
            raw_data = torch.load(trans_path, weights_only=False)

            with torch.no_grad():
                # l_raw is [1, 2048, 1024]
                l_0 = jepa_engine.get_latent(raw_data["state_0"], verbose=True)
                l_1 = jepa_engine.get_latent(raw_data["state_1"], verbose=True)

            # Ensure they are Tensors and Squeeze the batch dimension
            if not isinstance(l_0, torch.Tensor):
                l_0 = torch.tensor(l_0)
                l_1 = torch.tensor(l_1)

            # Resulting Shape: [2048, 1024]
            l_0 = l_0.squeeze(0).cpu()
            l_1 = l_1.squeeze(0).cpu()

            if USE_HALF_PRECISION:
                l_0 = l_0.half()
                l_1 = l_1.half()

            # 4. Save the "Rich" Macro Dictionary
            latent_dict = {
                "state_0": l_0,
                "action": torch.tensor(raw_data["action"]).float(),
                "state_1": l_1,
            }

            torch.save(latent_dict, dest_path)
            total_encoded += 1

            if total_encoded % 10 == 0:
                print(f"✅ Processed {total_encoded}/{num_files} transitions...")

        except Exception as e:
            print(f"⚠️ Failed to process {trans_path}: {e}")

    print(f"\n🎉 SUCCESS: High-Fidelity dataset encoded in '{DEST_DIR}'.")


if __name__ == "__main__":
    encode_high_fidelity_dataset()
