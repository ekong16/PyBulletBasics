import time
import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import utils

# 1. Import the original environment from your training script
# (Change 'train' to whatever you named your main training python file)
from standing_human_jepa import HumanStandEnv


# --- THE MANAGER'S TRICK: INHERITANCE ---
# We create a lightweight version of your environment that perfectly
# mimics the physical observation/action spaces, but kills the heavy ML vision.
class HumanStandEvalEnv(HumanStandEnv):
    def _get_reward(self, action, video_buffer):
        """
        Override the reward function to bypass V-JEPA and OpenCV.
        We don't need MSE or video saving when we are just watching the GUI.
        """
        done = False

        # Keep the 1m boundary check so it resets if it runs away
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        if np.linalg.norm(np.array(base_pos[:2])) > 1.0:
            done = True

        # Return dummy reward (0.0) since the model is already trained
        return 0.0, done, {"decomposition": {"z_TOTAL": 0.0}}


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # GUI = True so we can actually see it!
    with utils.PyBulletSim(gui=True, disableRender=False) as client:
        humanoid_id, plane_id = utils.setup_humanoid_scene(p)

        # 2. Instantiate the NEUTERED environment
        env = HumanStandEvalEnv(humanoid_id, plane_id, video_dir="dummy")

        # 3. Wrap exactly as you did in training (Notice: No FrameStack anymore!)
        env_single = DummyVecEnv([lambda: env])

        # Load the normalization stats (CRITICAL for the 102-vector to make sense)
        env_normalized = VecNormalize.load(
            "jepa_humanoid_vecnormalize.pkl", venv=env_single
        )

        # Freeze the normalization so it doesn't update during evaluation
        env_normalized.training = False
        env_normalized.norm_reward = False

        # 4. Load the Lean PPO Model
        print("🧠 Loading Model...")
        model = PPO.load("jepa_humanoid_v1_final.zip", env=env_normalized)

        obs = env_normalized.reset()
        episodes_played = 0

        print("\n🎬 Starting Playback...")
        while episodes_played < 10:
            done = False
            step_count = 0

            while not done:
                # Predict the action deterministically (no exploration noise)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_normalized.step(action)

                step_count += 1
                print("STEP: ", step_count)

                # Because your physics loop runs 160 ticks internally,
                # we don't need a massive sleep, but a tiny one keeps it smooth
                time.sleep(0.01)

            # Note: DummyVecEnv returns done as an array of booleans
            if done[0]:
                episodes_played += 1
                print(
                    f"✅ Episode {episodes_played} finished. Steps taken: {step_count}"
                )
                # Optional pause between episodes
                time.sleep(0.5)
