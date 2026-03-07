from jepa_data_farm import DataCollector, resetJointMotorsAndState, save_debug_mp4
import utils
from jepa_predictor import WorldPredictorPro

import os
import torch
import numpy as np
import time
from PIL import Image
import pybullet as p

# --- YOUR CUSTOM IMPORTS ---
import random

# IMPORTANT: Import your model class from wherever you defined it!
# Example: from model import WorldPredictorPro
my_jepa_model = utils.JEPAEngine()


def get_latent_from_file(file):
    target_path = file
    img = Image.open(target_path).convert("RGB")
    video_clip = np.asarray([img] * 16)
    ret = my_jepa_model.get_latent(video_clip, verbose=True)
    return ret


TARGET_LATENT = get_latent_from_file("poses/standing_pose.jpg")
START_LATENT = get_latent_from_file("poses/lying_pose.jpg")
DEVICE = torch.device("mps")
DEVICE_STR = "mps"


def generate_dream_sequence(
    model, z_start, z_goal, num_samples=1000, horizon=6, device=DEVICE
):
    model.eval()
    sequence = []
    mse_log = []

    # Standardize input shape
    z_step = (
        z_start.unsqueeze(0).to(device) if z_start.dim() == 2 else z_start.to(device)
    )
    z_goal = z_goal.to(device)

    for step in range(horizon):
        start_time = time.time()
        all_actions = torch.randn((num_samples, 28), device=device)
        all_preds = []

        print(f"Step {step + 1} | Processing {num_samples} samples 1-by-1...")

        for i in range(num_samples):
            # 1. Grab exactly one action: [1, 28]
            action_single = all_actions[i : i + 1]

            with torch.no_grad():
                # Prediction for a batch of 1: [1, 2048, 1024]
                with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                    # Start a timer

                    pred = model(z_step, action_single)

                    # if i % (num_samples / 2) == 0:
                    #     print(f"Sample {i} inference time: {elapsed:.4f} seconds")

                    all_preds.append(pred)

            # 2. THE MAC SPECIAL: Flush memory every 100 iterations to prevent the 7.8GiB crash
            if i % 100 == 0:
                torch.mps.empty_cache()

        # 3. Combine: [1000, 2048, 1024]
        z_next_preds = torch.cat(all_preds, dim=0)

        # 4. METRICS
        # l1_diff = torch.abs(z_next_preds - z_goal)
        # l1_dists = l1_diff.sum(dim=(1, 2))

        mse_diff = (z_next_preds - z_goal) ** 2
        mse_dists = mse_diff.mean(dim=(1, 2))

        # 5. WINNER
        best_idx = torch.argmin(mse_dists)
        best_mse = mse_dists[best_idx].item()

        z_step = z_next_preds[best_idx].unsqueeze(0)
        sequence.append(all_actions[best_idx].cpu().numpy())
        mse_log.append(best_mse)

        torch.mps.empty_cache()
        step_elapsed = time.time() - start_time

        # print(
        #     f"   Step {step + 1} Complete | BEST L1: {best_l1:.2f} | BEST MSE: {best_mse:.6f}"
        # )
        print(
            f"Step {step + 1} Complete | BEST MSE: {best_mse:.6f} | Step total time for {num_samples} samples: {step_elapsed:.4f} seconds"
        )

    return sequence, mse_log


def get_multi_trial_sequence_to_disk(
    model, z_start, z_goal, num_trials=10, num_samples=1000, horizon=6, device=DEVICE
):
    all_sequences = []
    all_mse = []
    for i in range(num_trials):
        sequence, mse = generate_dream_sequence(
            model, z_start, z_goal, num_samples, horizon, device
        )
        all_sequences.append(sequence)
        all_mse.append(mse)

    final_array = np.array(all_sequences)
    final_mse_arr = np.array(all_mse)
    np.save("multi_trial_strategy_actions.npy", final_array)
    np.save("multi_trial_strategy_mse.npy", final_mse_arr)
    print(f"\n✅ All {num_trials} trials saved to multi_trial_strategy.npy")


if __name__ == "__main__":
    model = WorldPredictorPro().to(DEVICE)
    model.load_state_dict(torch.load("world_model_final.pth"))
    get_multi_trial_sequence_to_disk(
        model, START_LATENT, TARGET_LATENT, num_trials=2, num_samples=10, horizon=6
    )
