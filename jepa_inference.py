import torch
import numpy as np
import time
import os
import Image

# --- YOUR CUSTOM IMPORTS ---
from jepa_data_farm import DataCollector
import utils

from jepa_predictor import WorldPredictorPro

# IMPORTANT: Import your model class from wherever you defined it!
# Example: from model import WorldPredictorPro
my_jepa_model = utils.JEPAEngine()


def get_target_latent():
    global Target_Video
    target_path = "poses/standing_pose.jpg"
    img = Image.open(target_path).convert("RGB")
    video_clip = np.asarray([img] * 16)
    Target_Video = video_clip
    ret = my_jepa_model.get_latent(video_clip, verbose=True)
    return ret


TARGET_LATENT = get_target_latent()


def generate_dream_sequence(
    model, z_current, z_goal, num_samples=1000, horizon=8, device="mps"
):
    """
    Greedy Monte Carlo Search in the Latent Space.
    Simulates `num_samples` actions, picks the lowest L1 distance to `z_goal`.
    """
    model.eval()
    sequence = []

    # Standardize input shape to [1, 2048, 1024]
    if z_current.dim() == 2:
        z_step = z_current.unsqueeze(0).to(device)
    else:
        z_step = z_current.clone().to(device)

    z_goal = z_goal.to(device)

    print(f"🧠 Dreaming {horizon} steps ahead (1,000 parallel paths per step)...")

    for step in range(horizon):
        # 1. Generate random torque vectors for exploration
        actions = torch.randn((num_samples, 28), device=device)

        # 2. Parallelize the batch
        z_batch = z_step.expand(num_samples, -1, -1)

        # 3. Predict futures (Flash Attention / Mixed Precision for speed)
        with torch.no_grad():
            with torch.autocast(device_type="mps", dtype=torch.float16):
                z_next_preds = model(z_batch, actions)

        # 4. L1 Distance Search
        l1_diff = torch.abs(z_next_preds - z_goal)
        distances = l1_diff.sum(dim=(1, 2))

        # 5. Pick the winner
        best_idx = torch.argmin(distances)
        best_action = actions[best_idx]

        # 6. Anchor the next step
        z_step = z_next_preds[best_idx].unsqueeze(0)

        # Save to our action plan
        sequence.append(best_action.cpu().numpy())
        print(
            f"   Step {step + 1}/{horizon} | Best L1 Distance to Target: {distances[best_idx].item():.2f}"
        )

    return sequence


def run_live_inference():
    DEVICE = "mps"

    print("Initializing Systems...")
    # 1. INITIALIZE THE BROKER (Simulator)
    dc = DataCollector(gui=True)

    # 3. LOAD THE ALPHA ENGINE (Model)
    model = WorldPredictorPro().to(
        DEVICE
    )  # <-- Make sure WorldPredictorPro is imported
    model.load_state_dict(
        torch.load("world_model_final.pth")
    )  # <-- Point to your best weights

    # 4. LOAD THE BENCHMARK (Hero Pose)
    z_goal = TARGET_LATENT

    print("\nStarting the 10-Trial Run...")
    for trial in range(10):
        print(f"\n========================================")
        print(f"🎬 TRIAL {trial + 1} / 10")
        print(f"========================================")

        # B. Get Initial State
        print("Capturing start state...")
        zero_action = np.zeros(28)
        start_video_numpy = dc.execute_and_record(zero_action)

        # Encode physical video to latent vector using your specific engine
        with torch.no_grad():
            # Ensure the numpy array is formatted correctly for your get_latent method
            z_start = my_jepa_model.get_latent(start_video_numpy, verbose=True)

        # C. Dream the Plan
        optimal_actions = generate_dream_sequence(
            model, z_start, z_goal, num_samples=1000, horizon=8, device=DEVICE
        )

        # D. Execute the Plan in Reality
        print(f"\n🚀 Executing Strategy in PyBullet...")
        for i, action in enumerate(optimal_actions):
            _ = dc.execute_and_record(action)
            print(f"   Physical Step {i + 1} applied.")
            time.sleep(0.05)  # Tiny pause for visual tracking in the GUI

        print(f"Trial {trial + 1} Finished. Observing final state...")
        time.sleep(2)


if __name__ == "__main__":
    run_live_inference()
