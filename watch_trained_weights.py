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
initial_position = [0, 0, 0.2]

roll = 0
pitch = math.pi / 2  # lying on belly
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])


with utils.PyBulletSim(gui=True) as client:
    # --- Simulation Initialization ---
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)

    planeId = p.loadURDF("plane.urdf")

    # Enable self-collision
    flags = p.URDF_USE_SELF_COLLISION
    my_humanoid_id = p.loadURDF(
        "humanoid/humanoid.urdf",
        initial_position,
        start_orientation,
        flags=flags,
        globalScaling=0.3,
    )

    env = HumanStandEnv(my_humanoid_id, planeId)
    env_monitored = Monitor(env)
    env_single = DummyVecEnv([lambda: env_monitored])
    env_stacked = VecFrameStack(env_single, n_stack=8)
    env_normalized = VecNormalize.load("vec_normalize_v13.pkl", venv=env_stacked)
    # env_normalized = VecNormalize(
    #     env_stacked, training=False, norm_obs=True, norm_reward=False
    # )
    env.training = False
    env.norm_reward = False
    model = PPO.load("humanoid_v13_final.zip", env_normalized)
    env_normalized.training = False

    for i in range(10):
        obs = env_normalized.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_normalized.step(action)

        print("DONE STEPPING")

    while True:
        chest_state = p.getLinkState(my_humanoid_id, 1, computeLinkVelocity=1)
        root_state = p.getLinkState(my_humanoid_id, 0)
        chest_pos, chest_orn = chest_state[0], chest_state[1]
        chest_z = chest_pos[2]
        root_z = root_state[0][2]
        head_index = 2
        head_state = p.getLinkState(my_humanoid_id, head_index)
        head_pos, head_orn = head_state[0], head_state[1]
        head_z = head_pos[2]

        print("Chext z", chest_z)
        print("Root z", root_z)
        print("HEAD Z", head_z)
        p.stepSimulation()
