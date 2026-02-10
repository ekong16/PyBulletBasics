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
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)

    # p.setTimeStep(1 / 240.0)
    # p.setPhysicsEngineParameter(numSolverIterations=200)

    PHYSICS_FREQ = 480
    p.setTimeStep(1.0 / PHYSICS_FREQ)

    p.setPhysicsEngineParameter(
        # 1. SUB-STEPPING (The Accuracy Multiplier)
        # This runs 4 internal physics ticks for every 1 stepSimulation call.
        numSubSteps=4,
        # 2. THE SLIDE CURE (Friction ERP)
        # Replaces 'frictionAnchor'. 0.2 helps lock those rectangular feet (11, 14).
        frictionERP=0.2,
        # 3. SOLVER STRENGTH
        # 150-200 iterations ensure the constraints don't drift.
        numSolverIterations=150,
        # 4. ERROR REDUCTION (ERP)
        # 0.2 is standard for joint stability.
        erp=0.2,
        # 5. CONTACT STABILITY
        # Prevents micro-bounces on the floor.
        contactSlop=0.001,
    )
    planeId = p.loadURDF("plane.urdf")

    # Enable self-collision
    flags = p.URDF_USE_SELF_COLLISION
    my_humanoid_id = p.loadURDF(
        "humanoid/humanoid.urdf", initial_position, start_orientation, flags=flags
    )

    env = HumanStandEnv(my_humanoid_id, planeId)
    env_monitored = Monitor(env)
    env_single = DummyVecEnv([lambda: env_monitored])
    env_stacked = VecFrameStack(env_single, n_stack=8)
    env_normalized = VecNormalize.load("vec_normalize_v12.pkl", venv=env_stacked)
    env.training = False
    env.norm_reward = False
    model = PPO.load("humanoid_v12_final.zip", env_normalized)
    env_normalized.training = False

    obs = env_normalized.reset()
    episodes_played = 0
    while episodes_played < 10:
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_normalized.step(action)

        print("DONE", done)
        if done[0]:
            episodes_played += 1
            print(f"Episode {episodes_played} finished.")

        print("DONE STEPPING")

    while True:
        # 1. Get Physical States
        # chest_state = p.getLinkState(my_humanoid_id, 1, computeLinkVelocity=1)
        # root_state = p.getLinkState(my_humanoid_id, 0)

        # chest_pos, chest_orn = chest_state[0], chest_state[1]
        # chest_z = chest_pos[2]
        # chest_vel_z = chest_state[6][2]  # Z-velocity in world space
        # root_z = root_state[0][2]

        # head_index = 2
        # head_state = p.getLinkState(my_humanoid_id, head_index)

        # head_pos, head_orn = head_state[0], head_state[1]
        # head_z = head_pos[2]
        # print("Head", head_z)
        # print("Chest", chest_z)
        # print("Root", root_z)
        p.stepSimulation()
