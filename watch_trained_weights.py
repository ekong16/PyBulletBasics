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


###VARS
initial_position = [0, 0, 0.9]

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
        "humanoid/humanoid.urdf", initial_position, start_orientation, flags=flags
    )

    env = HumanStandEnv(my_humanoid_id)

    model = PPO.load("humanoid_final.zip", env=env)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

    print("DONE STEPPING")

    while True:
        p.stepSimulation()
