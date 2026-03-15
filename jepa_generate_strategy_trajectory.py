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
PLANK_LATENT = get_latent_from_file("poses/plank_pose.jpg")
START_LATENT = get_latent_from_file("poses/lying_pose.jpg")
DEVICE = torch.device("mps")
DEVICE_STR = "mps"


def generate_dream_sequence_cem(
    model,
    z_start,
    z_goals,  # <--- CHANGED THIS TO A LIST
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

    z_step = z_start.to(device)
    assert z_step.shape == (1, 2048, 1024), f"z_step shape wrong: {z_step.shape}"

    z_step = z_start.to(device)

    assert z_step.shape == (1, 2048, 1024), f"z_step shape wrong: {z_step.shape}"

    for step in range(horizon):
        start_time = time.time()
        print(f"\n🚀 --- Step {step + 1}/{horizon} ---")

        current_goal = z_goals[step].to(device)
        if current_goal.dim() == 2:
            current_goal = current_goal.unsqueeze(0)
        assert current_goal.shape == (1, 2048, 1024), (
            f"current_goal shape wrong: {current_goal.shape}"
        )

        # Start with a wide search area
        mean = torch.zeros(28, device=device)
        std = torch.ones(28, device=device)

        # --- CEM OPTIMIZATION LOOP ---
        for cem_iter in range(cem_iters):
            step_samples = None
            # Step A: Generate Action Guesses
            if cem_iter == 0:
                # True Uniform Cold Start [-1, 1]
                step_samples = num_samples
                actions = (torch.rand(step_samples, 28, device=device) * 2.0) - 1.0

                assert actions.shape == (step_samples, 28), (
                    f"Actions shape wrong: {actions.shape}"
                )

            else:
                step_samples = num_samples
                # Gaussian Squeeze around previous winners
                actions = torch.normal(
                    mean.repeat(step_samples, 1), std.repeat(step_samples, 1)
                )
                actions = torch.clamp(actions, min=-1.0, max=1.0)  # Physical limits

                assert actions.shape == (num_samples, 28), (
                    f"Actions shape wrong: {actions.shape}"
                )

            # Calculate Elites
            n_elites = max(1, int(step_samples * elite_frac))

            # Step B: Evaluate Guesses in Chunks and Score IMMEDIATELY
            all_scores = []
            chunk_size = 10

            for start_idx in range(0, step_samples, chunk_size):
                end_idx = start_idx + chunk_size
                action_chunk = actions[start_idx:end_idx]

                # Figure out how many items are actually in this chunk (usually 10)
                current_batch_size = action_chunk.size(0)

                # Copy the current state 'current_batch_size' times so we can predict them all at once
                z_step_chunk = z_step.repeat(current_batch_size, 1, 1)

                # ASSERT 1: Check Input Shape
                # Note: We check against current_batch_size, not chunk_size, to prevent crashes on the final uneven chunk
                assert z_step_chunk.shape == (current_batch_size, 2048, 1024), (
                    "Repeat logic failed!"
                )
                assert z_step_chunk.shape[0] <= chunk_size, (
                    f"Too big chunk size shape {z_step_chunk.shape[0]}"
                )

                # Predict the next frame
                with torch.no_grad():
                    with torch.autocast(device_type=DEVICE_STR, dtype=torch.float16):
                        pred_chunk = model(z_step_chunk, action_chunk)

                # ASSERT 2: Check Prediction Shape
                assert pred_chunk.shape == (current_batch_size, 2048, 1024), (
                    "Prediction shape wrong!"
                )

                # --- THE MEMORY SAVER: Score Immediately ---
                # Calculate the L1 loss for just this tiny chunk
                chunk_l1_diffs = torch.abs(pred_chunk - current_goal)
                chunk_l1_scores = torch.mean(chunk_l1_diffs, dim=(1, 2))

                # ASSERT 3: Check Chunk Score Shape
                assert chunk_l1_scores.shape == (current_batch_size,), (
                    f"Chunk scores shape wrong: {chunk_l1_scores.shape}"
                )

                # Save the tiny 1D array of scores (just 10 numbers)
                all_scores.append(chunk_l1_scores)

                # FORCE MAC MEMORY FLUSH
                # We strictly delete the heavy [10, 2048, 1024] tensors so they don't pile up in memory
                del pred_chunk, chunk_l1_diffs
                torch.mps.empty_cache()

            # Step C: Glue the tiny score chunks back together into one 1D tensor
            # Shape: [step_samples]
            l1_scores = torch.cat(all_scores, dim=0)

            # ASSERT 4: Check Final Score Shape
            assert l1_scores.shape == (step_samples,), (
                f"Final scores shape wrong: {l1_scores.shape}"
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
            print(f"  ├─ Best Act:   [{formatted_action}]")
            print(f"  ├─ Dist Mean:  [{mean_str}]")
            print(f"  ├─ Dist Std:   [{std_str}]")
            print(f"  └─ Act, Elites:[{actions.shape[0]}, {n_elites}]")

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
        final_step_l1 = torch.mean(torch.abs(chosen_z_next - current_goal)).item()

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
    z_goals,  # <--- CHANGED HERE
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
            model, z_start, z_goals, num_samples, horizon, cem_iters, elite_frac, device
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
    model.load_state_dict(
        torch.load("world_model_masked.pth", map_location="cpu")["model_state"]
    )

    # --- YOUR NEW WAYPOINT RECIPE ---
    # Step 1: Aim for Plank
    # Step 2: Aim for Plank
    # Step 3: Aim for Stand
    # Step 4: Aim for Stand
    TARGET_TRAJECTORY = [PLANK_LATENT, PLANK_LATENT, TARGET_LATENT, TARGET_LATENT]

    get_multi_trial_sequence_to_disk(
        model,
        START_LATENT,
        TARGET_TRAJECTORY,
        num_samples=500,
        horizon=4,  # <--- MUST MATCH THE LENGTH OF TARGET_TRAJECTORY
        cem_iters=6,
        elite_frac=0.10,
        num_trials=3,
        device=DEVICE,
    )
