from jepa_data_farm import DataCollector, resetJointMotorsAndState
import utils

import os
import numpy as np
import time
import pybullet as p
import cv2
import os


my_jepa_model = utils.JEPAEngine()
INITIAL_POSITION = utils.SimConfig.INITIAL_POS
ROLL, PITCH, YAW = (
    utils.SimConfig.START_ORI[0],
    utils.SimConfig.START_ORI[1],
    utils.SimConfig.START_ORI[2],
)
START_ORIENTATION = p.getQuaternionFromEuler([ROLL, PITCH, YAW])
ACTIONS_NAME = "multi_trial_cem_strategy_actions.npy"
L1_NAME = "multi_trial_strategy_l1.npy"


def save_trial_mp4(
    full_video_buffer, actions, loss_log, trial_idx, folder="eval_replays"
):
    """Stitches 6 steps into a single MP4 with telemetry HUD."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"trial_{trial_idx:03d}_replay.mp4")

    # Dimensions: 256x256 (Native JEPA size)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(filepath, fourcc, 8.0, (256, 256))

    def draw_text(img, text, pos, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, font, 0.45, color, 1, cv2.LINE_AA)

    # full_video_buffer is a list of 6 step_videos, each [16, 256, 256, 3]
    for step_idx, step_video in enumerate(full_video_buffer):
        action = actions[step_idx]
        loss_val = loss_log[step_idx]

        # Action Summary
        act_max = np.max(np.abs(action))
        act_norm = np.linalg.norm(action)

        for frame_idx in range(16):
            frame = step_video[frame_idx]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- HUD OVERLAY ---
            draw_text(
                bgr,
                f"TRIAL: {trial_idx:02d} | STEP: {step_idx + 1}/{len(actions)} | FPS: 8.0",
                (10, 20),
                (0, 215, 255),
            )
            draw_text(bgr, f"DREAM L1: {loss_val:.4f}", (10, 40), (100, 255, 100))
            draw_text(
                bgr,
                f"MAX ACT.: {act_max:.2f} | ACT. NORM: {act_norm:.2f}",
                (10, 240),
                (255, 255, 255),
            )

            out.write(bgr)

    out.release()
    print(f"🎬 Saved Replay to: {filepath}")


if __name__ == "__main__":
    with utils.PyBulletSim(gui=True, disableRender=False) as client:
        humanoid_id, plane_id = utils.setup_humanoid_scene(p)
        collector = DataCollector(humanoid_id)

        all_strategies = np.load(ACTIONS_NAME)
        all_l1 = np.load(L1_NAME)
        print("all_strategies shape:", all_strategies.shape)
        print("all_l1 shape:", all_l1.shape)
        print(f"📂 Loaded {all_strategies.shape[0]} trials.")

        for trial_idx in range(all_strategies.shape[0]):
            # reset after each episode...
            resetJointMotorsAndState(
                humanoid_id,
                target_pos=INITIAL_POSITION,
                target_orn=START_ORIENTATION,
            )

            # 3. Settle Window (Gravity takes over)
            for _ in range(256):
                p.stepSimulation()

            print(f"\n🎬 EXECUTING TRIAL {trial_idx + 1}")

            trial_actions = all_strategies[trial_idx]  # (6, 28)
            trial_l1 = all_l1[trial_idx]
            trial_video_frames = []

            for step_idx, action in enumerate(trial_actions):
                print(f"   Step {step_idx + 1}/{len(trial_actions)}...")
                step_video = collector.execute_and_record(action)
                trial_video_frames.append(step_video)

            save_trial_mp4(trial_video_frames, trial_actions, trial_l1, trial_idx + 1)

            print(f"Trial {trial_idx + 1} finished. Resetting in 2 seconds...")
            time.sleep(2)
