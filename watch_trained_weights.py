import pybullet as p
import pybullet_data
import time
import gymnasium
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import utils
import math

from standing_human import HumanStandEnv
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

###VARS
initial_position = [0, 0, 0.9]

roll = 0
pitch = math.pi / 2  # lying on belly
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])


with utils.PyBulletSim(gui=True) as client:
    # --- Simulation Initialization ---
    my_humanoid_id, planeId = utils.setup_humanoid_scene(p)

    env = HumanStandEnv(my_humanoid_id, planeId)
    env_monitored = Monitor(env)
    env_single = DummyVecEnv([lambda: env_monitored])
    env_stacked = VecFrameStack(env_single, n_stack=8)
    env_normalized = VecNormalize.load("vec_normalize_v12.pkl", venv=env_stacked)
    # env.training = False
    # env.norm_reward = False
    model = PPO.load("humanoid_v12_final.zip", env_normalized)
    env_normalized.training = False
    env_normalized.norm_reward = False

    utils.print_joint_info(my_humanoid_id)
    utils.print_dynamics_info(my_humanoid_id)
    utils.print_link_states(my_humanoid_id)

    obs = env_normalized.reset()
    # Burn 10 steps to fill the frame stack with real physics data
    # zero_action = env_normalized.action_space.sample() * 0.0
    # for _ in range(10):
    #     # CRITICAL: Wrap it in brackets [] so VecEnv treats it as
    #     # "The action for Environment #0"
    #     obs, _, _, _ = env_normalized.step([zero_action])

    # episodes_played = 0
    # while episodes_played < 10:
    #     done = False
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=False)
    #         # action *= 0.66
    #         obs, reward, done, info = env_normalized.step(action)

    #     print("DONE", done)
    #     if done[0]:
    #         episodes_played += 1
    #         print(f"Episode {episodes_played} finished.")

    #     print("DONE STEPPING")

    while True:
        # 1. Get Physical States
        chest_state = p.getLinkState(my_humanoid_id, 1, computeLinkVelocity=1)
        root_state = p.getLinkState(my_humanoid_id, 0)

        chest_pos, chest_orn = chest_state[0], chest_state[1]
        chest_z = chest_pos[2]
        chest_vel_z = chest_state[6][2]  # Z-velocity in world space
        root_z = root_state[0][2]

        head_index = 2
        head_state = p.getLinkState(my_humanoid_id, head_index)

        head_pos, head_orn = head_state[0], head_state[1]
        head_z = head_pos[2]
        print("Head", head_z)
        print("Chest", chest_z)
        print("Root", root_z)
        p.stepSimulation()
