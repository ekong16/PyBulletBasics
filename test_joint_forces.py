import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import utils
from standing_human import HumanStandEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Match your training physics exactly
SCALE = 1.0  # 1.0 = Giant
PHYSICS_FREQ = 480
STEPS_PER_SECOND = PHYSICS_FREQ // 8  # Since you have 8 frame skips

with utils.PyBulletSim(gui=True) as client:
    humanoid_id, plane_id = utils.setup_humanoid_scene(p)

    env = HumanStandEnv(humanoid_id, plane_id)
    env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=8)
    env.reset()

    utils.print_joint_info(humanoid_id)
    utils.print_dynamics_info(humanoid_id)
    utils.print_link_states(humanoid_id)

    print(f"\n--- SUSTAINED PHYSICAL LIMIT TEST ---")

    # 1.5 seconds of Max Power, then 1.5 seconds of Relax
    # At 60Hz (480/8), that is 90 steps per phase
    PHASE_STEPS = int(1.5 * STEPS_PER_SECOND)

    for step in range(1000):
        # SUSTAINED LOGIC
        if (step // PHASE_STEPS) % 2 == 0:
            # THE BLAST: Hold +1.0 for 90 steps
            action_val = 1.0
            label = "BLASTING"
        else:
            # THE COLLAPSE: Hold 0.0 for 90 steps
            action_val = 0.0
            label = "RELAXING"

        # Force the assist OFF to see real physics
        env.assist_factor = 0.0

        action_vector = np.ones(env.action_space.shape) * action_val
        obs, reward, done, info = env.step([action_vector])

        if step % 20 == 0:
            chest_z = p.getLinkState(humanoid_id, 1)[0][2]
            print(f"Step {step:4} | {label:<10} | Height: {chest_z:.2f}m")

        time.sleep(1.0 / 60.0)
