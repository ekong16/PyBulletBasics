from jepa_data_farm import DataCollector, resetJointMotorsAndState, save_debug_mp4
import utils
from jepa_predictor import WorldPredictorPro

import os
import torch
from torch.distributions import Uniform
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

    action_dist = Uniform(low=-1.0, high=1.0)
    for step in range(horizon):
        start_time = time.time()
        all_actions = action_dist.sample((num_samples, 28)).to(DEVICE)
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


def generate_dream_sequence_cem(
    model,
    z_start,
    z_goal,
    num_samples=100,
    horizon=6,
    cem_iters=5,
    elite_frac=0.15,
    device=DEVICE,
):
    model.eval()
    sequence = []
    l1_log = []

    # 1. INITIAL SHAPE ENFORCEMENT
    # Ensure z_start and z_goal have a batch dimension of 1.
    # Expected target shape: [1, 2048, 1024]
    if z_start.dim() == 2:
        z_start = z_start.unsqueeze(0)
    if z_goal.dim() == 2:
        z_goal = z_goal.unsqueeze(0)

    z_step = z_start.to(device)
    z_goal = z_goal.to(device)

    assert z_step.shape == (1, 2048, 1024), f"z_step shape wrong: {z_step.shape}"
    assert z_goal.shape == (1, 2048, 1024), f"z_goal shape wrong: {z_goal.shape}"

    # Calculate Elites (e.g., 15% of 100 = 15)
    n_elites = max(1, int(num_samples * elite_frac))

    for step in range(horizon):
        start_time = time.time()
        print(f"\n🚀 --- Step {step + 1}/{horizon} ---")

        # Start with a wide search area
        mean = torch.zeros(28, device=device)
        std = torch.ones(28, device=device)

        # --- CEM OPTIMIZATION LOOP ---
        for cem_iter in range(cem_iters):
            # Step A: Generate Action Guesses
            if cem_iter == 0:
                if step == 0:
                    # True Uniform Cold Start [-1, 1]
                    actions = (
                        torch.rand(num_samples * 10, 28, device=device) * 2.0
                    ) - 1.0

                    assert actions.shape == (num_samples * 10, 28), (
                        f"Actions shape wrong: {actions.shape}"
                    )
                else:
                    inflated_std = prev_std * 5.0
                    inflated_std = torch.clamp(inflated_std, min=0.25, max=2.0)

                    actions = torch.normal(
                        prev_mean.repeat(num_samples * 3, 1),
                        inflated_std.repeat(num_samples * 3, 1),
                    )
                    actions = torch.clamp(actions, min=-1.0, max=1.0)
                    assert actions.shape == (num_samples * 3, 28), (
                        f"Actions shape wrong: {actions.shape}"
                    )
            else:
                # Gaussian Squeeze around previous winners
                actions = torch.normal(
                    mean.repeat(num_samples, 1), std.repeat(num_samples, 1)
                )
                actions = torch.clamp(actions, min=-1.0, max=1.0)  # Physical limits

                assert actions.shape == (num_samples, 28), (
                    f"Actions shape wrong: {actions.shape}"
                )

            # Step B: Evaluate Guesses in Chunks
            all_preds = []
            chunk_size = 10

            for start_idx in range(0, num_samples, chunk_size):
                end_idx = start_idx + chunk_size
                action_chunk = actions[start_idx:end_idx]

                # Figure out how many items are actually in this chunk (usually 50)
                current_batch_size = action_chunk.size(0)

                # Copy the current state 'current_batch_size' times so we can predict them all at once
                # Shape goes from [1, 2048, 1024] -> [50, 2048, 1024]
                z_step_chunk = z_step.repeat(current_batch_size, 1, 1)

                assert z_step_chunk.shape == (current_batch_size, 2048, 1024), (
                    "Repeat logic failed!"
                )

                # Predict the next frame
                with torch.no_grad():
                    with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                        pred_chunk = model(z_step_chunk, action_chunk)
                        all_preds.append(pred_chunk)

            # Glue the chunks back together into one massive tensor
            # Shape: [num_samples, 2048, 1024]
            z_next_preds = torch.cat(all_preds, dim=0)
            assert z_next_preds.shape == (num_samples, 2048, 1024), "Cat logic failed!"

            # Step C: Calculate the Score (Mean L1 Loss)
            # Find the absolute difference, then average across Tokens (dim 1) and Features (dim 2)
            # This leaves us with exactly 1 score per sample in the batch.
            l1_diffs = torch.abs(z_next_preds - z_goal)
            l1_scores = torch.mean(l1_diffs, dim=(1, 2))

            assert l1_scores.shape == (num_samples,), (
                f"Scores shape wrong: {l1_scores.shape}"
            )

            # Step D: Find the Winners (Elites)
            best_l1_scores, top_idx = torch.topk(l1_scores, n_elites, largest=False)

            # Extract the actual 28-D actions that scored the best
            # Shape: [n_elites, 28]
            elites = actions[top_idx]
            assert elites.shape == (n_elites, 28), "Elite extraction failed!"

            # Step E: Update the Search Area for the next loop
            mean = elites.mean(dim=0)
            std = elites.std(dim=0) + 1e-3  # Tiny noise floor to prevent 0.0 collapse

            assert mean.shape == (28,), "Mean calculation failed!"
            prev_mean = mean.clone()
            prev_std = std.clone()

            top_elite_action = elites[0].clone()

            # --- THE TELEMETRY PRINTER ---
            # Convert to CPU numpy array, then format each number to strictly take up 5 spaces: "+0.00"
            top_elite_action_np = top_elite_action.cpu().numpy()
            formatted_action = " ".join([f"{x:+5.2f}" for x in top_elite_action_np])
            mean_str = " ".join([f"{x:+5.2f}" for x in mean.cpu().numpy()])

            # Std Dev is strictly positive, so we drop the '+' sign to keep it clean
            std_str = " ".join([f"{x:5.2f}" for x in std.cpu().numpy()])
            print(
                f"  CEM Iter {cem_iter + 1}/{cem_iters} | Best L1: {best_l1_scores[0].item():.5f}"
            )
            print(f"  ├─ Best Act: [{formatted_action}]")
            print(f"  ├─ Dist Mean:[{mean_str}]")
            print(f"  └─ Dist Std: [{std_str}]")

        # --- END CEM THINKING LOOP ---

        # After 5 iterations, the 'mean' is our chosen surgical action for this step.
        # Reshape from [28] -> [1, 28] so the model can process it
        best_action = mean.unsqueeze(0)
        assert best_action.shape == (1, 28), "Final action shape wrong!"

        # --- THE FINAL CHOSEN ACTION PRINTER ---
        chosen_act_str = " ".join(
            [f"{x:+5.2f}" for x in best_action.squeeze(0).cpu().numpy()]
        )
        final_std_str = " ".join([f"{x:5.2f}" for x in std.cpu().numpy()])

        print(f"  ========================================")
        print(f"  🎯 FINAL CHOSEN DIST FOR STEP {step + 1}:")
        print(f"  ├─ Final Mean: [{chosen_act_str}]")
        print(f"  └─ Final Std:  [{final_std_str}]")
        print(f"  ========================================")
        # ---------------------------------------

        # Get the actual predicted next state for this chosen action to carry forward
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                chosen_z_next = model(z_step, best_action)

        assert chosen_z_next.shape == (1, 2048, 1024), "Chosen state shape wrong!"

        # Log metrics and prepare for next real-world step
        final_step_l1 = torch.mean(torch.abs(chosen_z_next - z_goal)).item()

        # Save the action to our sequence list (strip the batch dimension to make it just 28 numbers)
        sequence.append(best_action.squeeze(0).cpu().numpy())
        l1_log.append(final_step_l1)

        # Advance the simulation state
        z_step = chosen_z_next

        torch.mps.empty_cache()
        step_elapsed = time.time() - start_time
        print(
            f"✅ Step {step + 1} Complete | Action L1: {final_step_l1:.5f} | Time: {step_elapsed:.2f}s"
        )

    return sequence, l1_log


def get_multi_trial_sequence_to_disk(
    model,
    z_start,
    z_goal,
    num_samples=100,
    horizon=6,
    cem_iters=5,
    elite_frac=0.15,
    num_trials=10,
    device=DEVICE,
):
    all_sequences = []
    all_l1 = []
    for i in range(num_trials):
        sequence, l1_loss = generate_dream_sequence_cem(
            model, z_start, z_goal, num_samples, horizon, cem_iters, elite_frac, device
        )
        all_sequences.append(sequence)
        all_l1.append(l1_loss)

    final_array = np.array(all_sequences)
    final_l1_arr = np.array(all_l1)
    np.save("multi_trial_cem_strategy_actions.npy", final_array)
    np.save("multi_trial_strategy_l1.npy", final_l1_arr)
    print(f"\n✅ All {num_trials} trials saved to multi_trial_cem_strategy_actions.npy")


if __name__ == "__main__":
    model = WorldPredictorPro().to(DEVICE)
    model.load_state_dict(torch.load("world_model_final.pth"))
    get_multi_trial_sequence_to_disk(
        model,
        START_LATENT,
        TARGET_LATENT,
        num_samples=100,
        horizon=6,
        cem_iters=5,
        elite_frac=0.15,
        num_trials=1,
        device=DEVICE,
    )
