import pybullet as p
import pybullet_data
import time
import gymnasium
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import utils
import math

###VARS
initial_position = [0, 0, 0.9]

roll = 0
pitch = math.pi / 2  # lying on belly
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])


# ----------------
# Gym environment
# ----------------
class HumanStandEnv(gymnasium.Env):
    def __init__(self, humanoid_id):
        super().__init__()
        n_joints = p.getNumJoints(humanoid_id)
        self.action_space = spaces.Box(
            low=-50, high=50, shape=(n_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_joints * 2 + 7,), dtype=np.float32
        )
        self.max_steps = 1000
        self.steps_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium compatibility
        self.steps_count = 0
        # Reset PyBullet humanoid
        p.resetBasePositionAndOrientation(
            humanoid_id, initial_position, start_orientation
        )
        for j in range(p.getNumJoints(humanoid_id)):
            p.resetJointState(humanoid_id, j, 0, 0)

        # Settle the robot
        print("Settling robot in RESET")
        for _ in range(100):
            p.stepSimulation()
        print("DONE Settling robot in RESET")

        obs = self._get_obs()  # get initial observation
        info = {}
        return obs, info

    def step(self, action):
        for j, torque in enumerate(action):
            p.setJointMotorControl2(humanoid_id, j, p.TORQUE_CONTROL, force=torque)
        p.stepSimulation()
        # time.sleep(1 / 240.0)  # visualization
        obs = self._get_obs()
        reward = self._get_reward(action)
        done = self._is_done()

        info = {}
        truncated = False

        self.steps_count += 1
        return obs, reward, done, truncated, info

    def _get_obs(self):
        angles, velocities = [], []
        for j in range(p.getNumJoints(humanoid_id)):
            js = p.getJointState(humanoid_id, j)
            angles.append(js[0])
            velocities.append(js[1])
        pos, orn = p.getBasePositionAndOrientation(humanoid_id)
        lin_vel, ang_vel = p.getBaseVelocity(humanoid_id)
        return np.array(
            angles + velocities + list(orn) + list(ang_vel), dtype=np.float32
        )

    def _get_reward(self, action):
        torso_height = p.getBasePositionAndOrientation(humanoid_id)[0][2]
        reward = 1.0 - 0.001 * np.sum(np.square(action))
        if torso_height < 1.0:
            reward -= 5.0
        return reward

    def _is_done(self):
        torso_height = p.getBasePositionAndOrientation(humanoid_id)[0][2]
        if torso_height < 0.01:
            return True
        if self.steps_count > self.max_steps:
            return True


with utils.PyBulletSim(gui=True) as client:
    # --- Simulation Initialization ---
    # physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)

    planeId = p.loadURDF("plane.urdf")

    humanoid_id = p.loadURDF(
        "humanoid/humanoid.urdf", initial_position, start_orientation
    )

    for j in range(p.getNumJoints(humanoid_id)):
        p.setJointMotorControl2(humanoid_id, j, p.VELOCITY_CONTROL, force=0)

    utils.print_joint_info(humanoid_id)
    utils.print_link_states(humanoid_id)
    utils.print_dynamics_info(humanoid_id)

    p.setTimeStep(1 / 240.0)
    p.setPhysicsEngineParameter(numSolverIterations=200)

    print("stepping sim 1000 steps to settle robot")
    for _ in range(100):
        p.stepSimulation()
    print("humanoid should be settled.")
    # ----------------
    # Train PPO
    # ----------------
    env = HumanStandEnv(humanoid_id)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
