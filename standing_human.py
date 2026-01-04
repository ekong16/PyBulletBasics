import pybullet as p
import pybullet_data
import gymnasium
from gymnasium import spaces
import numpy as np
import math
import random
import utils
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable

# ==========================================
# GLOBAL VARS & CONFIG
# ==========================================
INITIAL_POSITION = [0, 0, 0.2]
ROLL, PITCH, YAW = 0, math.pi / 2, 0
START_ORIENTATION = p.getQuaternionFromEuler([ROLL, PITCH, YAW])

# Max Force (Nm) per joint
MAX_TORQUE_MAP_OLD = {
    "chest": [100, 100, 100],
    "neck": [10, 10, 10],
    "right_shoulder": [100, 100, 100],
    "left_shoulder": [100, 100, 100],
    "right_elbow": 60,
    "left_elbow": 60,
    "right_hip": [200, 200, 200],
    "left_hip": [200, 200, 200],
    "right_knee": 150,
    "left_knee": 150,
    "right_ankle": [40, 40, 40],
    "left_ankle": [40, 40, 40],
}

MAX_TORQUE_MAP = {
    # --- ARMS (40Nm) ---
    # Strong enough to do the pushup (requires ~38Nm),
    # but limited so they don't act like "jackhammers" that break the floor.
    "right_shoulder": [40, 40, 40],
    "left_shoulder": [40, 40, 40],
    "right_elbow": 40,
    "left_elbow": 40,
    # --- LEGS (90Nm) ---
    # The "Engine." 90Nm is required to lift the 27kg upper body lever.
    "right_hip": [90, 90, 90],
    "left_hip": [90, 90, 90],
    "right_knee": 90,
    "left_knee": 90,
    # --- ANKLES (25Nm) ---
    # 25Nm provides balance. 40Nm was causing the "foot vibration" explosion.
    "right_ankle": [25, 25, 25],
    "left_ankle": [25, 25, 25],
    # --- STABILIZERS ---
    "neck": [5, 5, 5],
    "chest": [45, 45, 45],
}


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class RewardLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "decomposition" in info:
                for key, value in info["decomposition"].items():
                    self.logger.record(f"reward/{key}", value)
        return True


def resetJointMotorsAndState(humanoid_id):
    p.resetBasePositionAndOrientation(humanoid_id, INITIAL_POSITION, START_ORIENTATION)
    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        jt = info[2]
        if jt in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.resetJointState(humanoid_id, j, 0, 0)
            p.setJointMotorControl2(humanoid_id, j, p.VELOCITY_CONTROL, force=0)
        elif jt == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(humanoid_id, j, [0, 0, 0, 1], [0, 0, 0])
            p.setJointMotorControlMultiDof(
                humanoid_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                force=[0, 0, 0],
            )
        p.changeDynamics(
            humanoid_id,
            j,
            lateralFriction=1.0,
            linearDamping=0.0,
            angularDamping=0.0,
            jointDamping=0.7,  # <--- THIS STOPS THE FLAILING
            maxJointVelocity=6.0,
        )

    # Fix zero-inertia and zero-mass for IDs -1, 5, and 8
    # Using a small but stable inertia diagonal [0.001, 0.001, 0.001]

    # ID -1: The Base Link
    p.changeDynamics(
        humanoid_id, -1, mass=0.1, localInertiaDiagonal=[0.001, 0.001, 0.001]
    )

    # ID 5: Right Wrist
    p.changeDynamics(humanoid_id, 5, localInertiaDiagonal=[0.001, 0.001, 0.001])

    # ID 8: Left Wrist
    p.changeDynamics(humanoid_id, 8, localInertiaDiagonal=[0.001, 0.001, 0.001])

    print("Dynamics updated: IDs -1, 5, and 8 now have non-zero inertia.")
    for link in [5, 8]:
        p.changeDynamics(humanoid_id, link, lateralFriction=10.0)
    for link in [11, 14]:
        p.changeDynamics(humanoid_id, link, lateralFriction=3.0)


class HumanStandEnv(gymnasium.Env):
    def __init__(self, humanoid_id, plane_id):
        super().__init__()
        self.humanoid_id = humanoid_id
        self.plane_id = plane_id
        self.max_steps = 1024  # Increased slightly to allow for stability testing
        self.steps_count = 0
        self.episode_count = 0
        self.current_energy_cost = 0.0

        self.weights = {
            "chest_height": 2.0,  # Primary motivator
            "root_height": 1.0,  # Secondary motivator
            "neck_height": 1.5,  # High priority to encourage lifting the head
            "uprightness": 1.0,  # Orientation weight
            "neck_orientation": 1.0,  # Keeps the head looking forward/level
            "chest_vel": 0.1,  # Gated velocity (only works when low)
            "energy_cost": -0.05,  # PENALTY: Applied to sum(action^2)
            "survival_bonus": 0.5,  # BONUS: Applied every step alive
            "termination_penalty": -100.0,
        }
        self.foot_links = []

        self._init_spaces()

    def _init_spaces(self):
        n_joints = p.getNumJoints(self.humanoid_id)
        self.dof_per_joint = []
        for j in range(n_joints):
            jt = p.getLinkState(self.humanoid_id, j)
            jt = p.getJointInfo(self.humanoid_id, j)[2]
            self.dof_per_joint.append(
                3 if jt == p.JOINT_SPHERICAL else (1 if jt == p.JOINT_REVOLUTE else 0)
            )

        # Action space is normalized (-1 to 1)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(sum(self.dof_per_joint),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_joints * 2 + 7,), dtype=np.float32
        )
        self.foot_links = []
        for j in range(p.getNumJoints(self.humanoid_id)):
            info = p.getJointInfo(self.humanoid_id, j)
            link_name = info[12].decode("utf-8")
            if "foot" in link_name or "ankle" in link_name:
                self.foot_links.append(j)
        print(f"DEBUG: Found foot links at indices: {self.foot_links}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.steps_count = 0
        self.current_energy_cost = 0.0

        # Randomize friction slightly to improve robustness
        newFriction = random.uniform(0.8, 1.2)
        p.changeDynamics(self.plane_id, -1, lateralFriction=newFriction)
        print("Setting friction to: ", newFriction)

        resetJointMotorsAndState(self.humanoid_id)
        for _ in range(50):
            p.stepSimulation()
        return self._get_obs(), {}

    def step(self, action):
        printOn = True
        printStep = self.steps_count % 256 == 0
        if printStep and printOn:
            print(
                f"\n--- EPISODE {self.episode_count} | STEP {self.steps_count} | FORCE DIAGNOSTICS ---"
            )

        # 1. CALCULATE ENERGY COST (Normalized Actions)
        # Sum of squares of actions. Max value approx 17.0 (if all joints maxed).
        # Penalty = 17.0 * -0.05 = -0.85 per step.
        self.current_energy_cost = np.sum(np.square(action))

        action_idx = 0
        for j in range(p.getNumJoints(self.humanoid_id)):
            name = p.getJointInfo(self.humanoid_id, j)[1].decode("utf-8")
            if name not in MAX_TORQUE_MAP:
                continue

            max_f = np.array(MAX_TORQUE_MAP[name])

            if self.dof_per_joint[j] == 1:
                torque = action[action_idx] * max_f
                if printStep and printOn:
                    print(
                        f"Joint: {name:<15} | Action: {action[action_idx]:>6.2f} | Torque: {torque:>6.1f} Nm"
                    )
                p.setJointMotorControl2(
                    self.humanoid_id, j, p.TORQUE_CONTROL, force=torque
                )
                action_idx += 1

            elif self.dof_per_joint[j] == 3:
                raw_actions = action[action_idx : action_idx + 3]
                torques = raw_actions * max_f
                if printStep and printOn:
                    effort_pct = np.linalg.norm(raw_actions) / math.sqrt(3) * 100
                    print(
                        f"Joint: {name:<15} | Effort: {effort_pct:>5.1f}% | Torques: {np.round(torques, 1)}"
                    )
                p.setJointMotorControlMultiDof(
                    self.humanoid_id, j, p.TORQUE_CONTROL, force=list(torques)
                )
                action_idx += 3

        for _ in range(4):
            p.stepSimulation()

        obs = self._get_obs()
        reward, done, decomposition = self._get_reward()

        self.steps_count += 1
        truncated = self.steps_count >= self.max_steps
        info = {"decomposition": decomposition}

        return obs, reward, done, truncated, info

    def _get_reward(self):
        # 1. Get Physical States
        chest_state = p.getLinkState(self.humanoid_id, 1, computeLinkVelocity=1)
        root_state = p.getLinkState(self.humanoid_id, 0)

        chest_pos, chest_orn = chest_state[0], chest_state[1]
        chest_z = chest_pos[2]
        chest_vel_z = chest_state[6][2]  # Z-velocity in world space
        root_z = root_state[0][2]

        head_index = 2
        head_state = p.getLinkState(self.humanoid_id, head_index)

        head_pos, head_orn = head_state[0], head_state[1]
        head_z = head_pos[2]

        # 2. Orientation (Uprightness)
        rot_matrix = np.array(p.getMatrixFromQuaternion(chest_orn)).reshape(3, 3)
        chest_up_vector = rot_matrix[:, 2]  # The local Z-axis of the chest link
        uprightness = max(0, np.dot(chest_up_vector, [0, 0, 1]))

        head_rot_matrix = np.array(p.getMatrixFromQuaternion(head_orn)).reshape(3, 3)
        head_up_vector = head_rot_matrix[:, 2]
        head_uprightness = max(0, np.dot(head_up_vector, [0, 0, 1]))

        # CONTACT DETECTION (The Cure for Helicopter Legs)
        contact_points = 0
        for link_idx in self.foot_links:
            # Check if this link is touching the floor (plane_id)
            # p.getContactPoints returns a list; if not empty, we have contact
            contacts = p.getContactPoints(
                bodyA=self.humanoid_id, bodyB=self.plane_id, linkIndexA=link_idx
            )
            if len(contacts) > 0:
                contact_points += 1

        # Reward 1.0 per foot that is grounded.
        # This pays +2.0 for a stable stand, which is HUGE.
        feet_contact_raw = 1.0 * contact_points

        # 3. REWARD COMPONENTS

        # A. Height (The Goal)
        reward_chest = self.weights["chest_height"] * max(0, chest_z - 0.132)
        reward_root = self.weights["root_height"] * max(0, root_z - 0.108)

        # B. Uprightness (Scaled)
        reward_upright = self.weights["uprightness"] * (
            uprightness * max(0, chest_z - 0.132)
        )

        # C. GATED VELOCITY (Anti-Popcorn Logic)
        # Only reward upward velocity if we are ON THE FLOOR (< 0.6m).
        # Once standing, velocity reward is ZERO.
        if chest_z < 0.6:
            reward_vel = self.weights["chest_vel"] * chest_vel_z
        else:
            reward_vel = 0.0

        # NEW: NECK/HEAD REWARDS (Simplified)
        reward_neck_height = self.weights["neck_height"] * max(0, head_z - 0.123)

        # 2. Head Orientation
        # Gated by height so we don't reward looking at the ceiling while lying on back.
        reward_neck_orient = self.weights["neck_orientation"] * (
            head_uprightness * max(0, head_z - 0.123)
        )

        # D. ACTION PENALTY (New)
        # Penalize high action values to prevent flailing
        reward_energy = self.weights["energy_cost"] * self.current_energy_cost

        # E. SURVIVAL BONUS (New)
        # Constant reward for staying alive (not terminating)
        reward_survival = self.weights["survival_bonus"]

        # F. Feet Contact (New)
        # --- CRITICAL: BELLY START PROTECTION ---
        # Only grant this if the chest is reasonably high (>0.6m)
        # Otherwise it will just lie on the floor and tap its feet.
        if chest_z > 0.3:
            reward_feet = feet_contact_raw
        else:
            reward_feet = 0.0

        # 4. Termination Logic
        done = False
        reward_term = 0.0

        # Terminate if chest touches ground (0.25) or flies away (2.0)
        if chest_z < 0.1 or chest_z > 1.8:
            done = True
            reward_term = self.weights["termination_penalty"]
            reward_survival = 0.0  # No survival bonus on the death step

        total_reward = (
            reward_chest
            + reward_root
            + reward_upright
            + reward_vel
            + reward_energy
            + reward_survival
            + reward_feet
            + reward_term
            + reward_neck_height
            + reward_neck_orient
        )

        decomp = {
            "01_chest_height": reward_chest,
            "02_root_height": reward_root,
            "03_upright": reward_upright,
            "04_velocity": reward_vel,
            "05_energy": reward_energy,
            "06_survival": reward_survival,
            "07_feet": reward_feet,
            "08_term": reward_term,
            "09_neck_height": reward_neck_height,
            "10_neck_uprightness": reward_neck_orient,
            "z_TOTAL": total_reward,
        }

        return total_reward, done, decomp

    def _get_obs(self):
        angles, velocities = [], []
        for j in range(p.getNumJoints(self.humanoid_id)):
            js = p.getJointState(self.humanoid_id, j)
            angles.append(js[0])
            velocities.append(js[1])
        _, orn = p.getBasePositionAndOrientation(self.humanoid_id)
        _, ang_vel = p.getBaseVelocity(self.humanoid_id)
        return np.array(
            angles + velocities + list(orn) + list(ang_vel), dtype=np.float32
        )


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    with utils.PyBulletSim(gui=True) as client:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        plane_id = p.loadURDF("plane.urdf")
        humanoid_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            INITIAL_POSITION,
            START_ORIENTATION,
            flags=p.URDF_USE_SELF_COLLISION,
            globalScaling=0.3,
        )

        print("\n--- Humanoid Diagnostic Info ---")
        utils.print_joint_info(humanoid_id)
        utils.print_dynamics_info(humanoid_id)
        utils.print_link_states(humanoid_id)

        p.setTimeStep(1 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)

        env = HumanStandEnv(humanoid_id, plane_id)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=8)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

        # MODEL CONFIGURATION
        # ent_coef kept low (0.0) as requested.
        # learning_rate constant 5e-5 for stability.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-5,
            n_steps=4096,
            batch_size=512,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/",
        )

        print("--- Starting Training with Gated Velocity & Energy Penalty ---")
        model.learn(
            total_timesteps=10_000_000,
            callback=RewardLoggerCallback(),
            tb_log_name="V13_Run1_test",
        )

        model.save("humanoid_v13_final")
        env.save("vec_normalize_v13.pkl")
