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
import torch as th

# ==========================================
# GLOBAL VARS & CONFIG
# ==========================================
INITIAL_POSITION = [0, 0, 0.9]
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
    "chest": [500, 500, 500],
    "neck": [200, 200, 200],  # Fixed dangling neck
    "right_shoulder": [200, 200, 200],
    "left_shoulder": [200, 200, 200],
    "right_elbow": 100,
    "left_elbow": 100,
    "right_hip": [800, 800, 800],  # Massive hip power for 6m lever
    "left_hip": [800, 800, 800],
    "right_knee": 800,  # Massive knee power for crouching
    "left_knee": 800,
    "right_ankle": [400, 400, 400],  # Strong ankles to stop toppling
    "left_ankle": [400, 400, 400],
}

TARGET_HEAD = 6.06
TARGET_CHEST = 4.96
TARGET_ROOT = 3.82


def linear_schedule(
    initial_value: float, min_value: float = 1e-6
) -> Callable[[float], float]:
    """
    Decays the learning rate linearly from initial_value to min_value.
    """

    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 (start) and goes to 0.0 (end)

        # Calculate the drop range
        decay_range = initial_value - min_value

        # Current rate = Minimum Floor + (Amount left to decay)
        current_rate = min_value + (progress_remaining * decay_range)

        return current_rate

    return func


class RewardLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "decomposition" in info:
                for key, value in info["decomposition"].items():
                    self.logger.record(f"reward/{key}", value)
        return True


class PuppetMasterWrapper(gymnasium.Wrapper):
    def __init__(
        self, env, humanoid_id, total_timesteps=20_000_000
    ):  # Changed to 20M (Fail Fast)
        super().__init__(env)
        self.humanoid_id = humanoid_id
        self.total_timesteps = total_timesteps
        self.current_step = 0

        # Calculate Mass (Kept your logic)
        self.total_mass = sum(
            [
                p.getDynamicsInfo(humanoid_id, i)[0]
                for i in range(p.getNumJoints(humanoid_id))
            ]
        )
        self.total_mass += p.getDynamicsInfo(humanoid_id, -1)[0]

        # Calculate Weight (Gravity Force)
        self.robot_weight = self.total_mass * 9.81
        print("ROBOT WEIGHT", self.robot_weight)

    def step(self, action):
        self.current_step += 1

        # --- 1. CONFIGURATION: THE "MOON GRAVITY" SETUP ---
        # We attach the spring HIGH (10m) so it never goes slack.
        SPRING_ANCHOR = 10.0

        # Stiffness 0.16:
        #   - At Floor (Dist 10m): Pulls 1.6G (Strong Lift)
        #   - At Target 5m (Dist 5m): Pulls 0.8G (Perfect Anti-Gravity)
        start_kp = self.robot_weight * 0.16
        start_kd = self.total_mass * 0.1

        # --- 2. SCHEDULE: WARMUP -> DECAY -> REALITY ---
        WARMUP_END = 2_000_000  # Phase 1: Full help until 2M
        DECAY_END = 10_000_000  # Phase 2: Fade out until 10M

        if self.current_step < WARMUP_END:
            # Phase 1: Full Moon Gravity (No Decay)
            current_kp = start_kp
            current_kd = start_kd
            phase = "WARMUP"
        elif self.current_step < DECAY_END:
            # Phase 2: Linear Decay
            # 0.0 at warmup end -> 1.0 at decay end
            progress = (self.current_step - WARMUP_END) / (DECAY_END - WARMUP_END)
            current_kp = start_kp * (1.0 - progress)
            current_kd = start_kd * (1.0 - progress)
            phase = "DECAY"
        else:
            # Phase 3: Zero Assist (The Real World)
            current_kp = 0.0
            current_kd = 0.0
            phase = "REAL"

        # Logging (Every 50k steps)
        if self.current_step % 50_000 == 0:
            print(
                f"Step: {self.current_step} | Phase: {phase} | Kp: {current_kp:.2f} | Kd: {current_kd:.1f}"
            )

        # --- 3. CALCULATE FORCE ---
        try:
            # Only calculate if assist is active
            if current_kp > 0.001:
                link_state = p.getLinkState(self.humanoid_id, 1, computeLinkVelocity=1)
                current_z = link_state[0][2]
                current_vel_z = link_state[6][2]

                # CRITICAL: Pull towards the HIGH ANCHOR (10.0), not the chest target
                error_pos = SPRING_ANCHOR - current_z
                error_vel = 0.0 - current_vel_z

                spring_force_z = (current_kp * error_pos) + (current_kd * error_vel)

                # Safety Clips
                spring_force_z = max(0.0, spring_force_z)  # No pushing down
                spring_force_z = min(
                    spring_force_z, self.robot_weight * 3.0
                )  # Safety cap

                p.applyExternalForce(
                    self.humanoid_id,
                    1,
                    [0, 0, spring_force_z],
                    link_state[0],
                    p.WORLD_FRAME,
                )

        except Exception as e:
            pass

        return self.env.step(action)


class GravityCurriculumWrapper(gymnasium.Wrapper):
    def __init__(self, env, total_timesteps=25_000_000, start_g=-2.0, end_g=-9.81):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.start_g = start_g
        self.end_g = end_g
        self.current_step = 0

    def step(self, action):
        # 1. Update Step Count
        self.current_step += 1

        # 2. Calculate Gravity
        progress = min(1.0, self.current_step / self.total_timesteps)
        # current_gravity = 5
        current_gravity = self.start_g + (self.end_g - self.start_g) * progress

        # 3. Apply Gravity
        # Ensure '0' matches your PyBullet client ID if you have multiple sims.
        p.setGravity(0, 0, current_gravity)

        # 4. Standard Step (Gymnasium returns 5 values)
        # obs, reward, terminated, truncated, info
        step_result = self.env.step(action)

        # Safety Check: Handle both Old Gym (4 vals) and New Gymnasium (5 vals)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            info["gravity_z"] = current_gravity
            return obs, reward, terminated, truncated, info
        else:
            # Fallback if your specific env is still returning 4 values
            obs, reward, done, info = step_result
            info["gravity_z"] = current_gravity
            return obs, reward, done, info

    def reset(self, **kwargs):
        # FIX: Gymnasium requires passing 'seed' and 'options' down the chain.
        # We use **kwargs to catch everything SB3 throws at it.
        return self.env.reset(**kwargs)


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
        self.total_global_steps = 0

        self.current_energy_cost = 0.0

        self.total_mass = sum(
            [
                p.getDynamicsInfo(humanoid_id, i)[0]
                for i in range(p.getNumJoints(humanoid_id))
            ]
        )
        self.total_mass += p.getDynamicsInfo(humanoid_id, -1)[0]  # Add Base Mass
        self.robot_weight = self.total_mass * 9.81

        self.current_kp = 0.0
        self.current_kd = 0.0

        print(f"DEBUG: Robot Total Mass: {self.total_mass:.2f} kg")
        print(f"DEBUG: Robot Weight: {self.robot_weight:.2f} N")

        self.weights = {
            "chest_height": 5.0,  # Primary motivator
            "root_height": 2.0,  # Secondary motivator
            "neck_height": 1.5,  # High priority to encourage lifting the head
            "uprightness": 3.0,  # Orientation weight
            "feet_contact": 5.0,
            "neck_orientation": 1.0,  # Keeps the head looking forward/level
            "chest_vel": 1.0,  # Gated velocity (only works when low)
            "energy_cost": -0.001,  # PENALTY: Applied to sum(action^2)
            "survival_bonus": 0.5,  # BONUS: Applied every step alive
            "termination_penalty": -100.0,
        }
        self.foot_links = []

        self._init_spaces()

    def _init_spaces(self):
        n_joints = p.getNumJoints(self.humanoid_id)
        self.dof_per_joint = []
        obs_dim = 0

        for j in range(n_joints):
            jt = p.getJointInfo(self.humanoid_id, j)[2]
            if jt == p.JOINT_SPHERICAL:
                self.dof_per_joint.append(3)  # Action: 3 Torques
                obs_dim += 7  # Obs: 4 Quat + 3 Vel
            elif jt == p.JOINT_REVOLUTE:
                self.dof_per_joint.append(1)  # Action: 1 Torque
                obs_dim += 2  # Obs: 1 Angle + 1 Vel
            else:
                self.dof_per_joint.append(0)

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(sum(self.dof_per_joint),), dtype=np.float32
        )

        # Obs Space breakdown:
        # 1. Joint Data (obs_dim)
        # 2. Root Pos (3) + Root Orn (4) + Root AngVel (3) = 10
        # 3. Assist Factors (Kp, Kd) = 2
        # Total Extras = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim + 12,), dtype=np.float32
        )

        # Foot logic unchanged
        self.foot_links = []
        for j in range(n_joints):
            info = p.getJointInfo(self.humanoid_id, j)
            link_name = info[12].decode("utf-8")
            if "foot" in link_name or "ankle" in link_name:
                self.foot_links.append(j)
        print(f"DEBUG: Found foot links at indices: {self.foot_links}")

    # def _init_spaces(self):
    #     n_joints = p.getNumJoints(self.humanoid_id)
    #     self.dof_per_joint = []
    #     for j in range(n_joints):
    #         jt = p.getLinkState(self.humanoid_id, j)
    #         jt = p.getJointInfo(self.humanoid_id, j)[2]
    #         self.dof_per_joint.append(
    #             3 if jt == p.JOINT_SPHERICAL else (1 if jt == p.JOINT_REVOLUTE else 0)
    #         )

    #     # Action space is normalized (-1 to 1)
    #     self.action_space = spaces.Box(
    #         low=-1, high=1, shape=(sum(self.dof_per_joint),), dtype=np.float32
    #     )
    #     self.observation_space = spaces.Box(
    #         low=-np.inf, high=np.inf, shape=(n_joints * 2 + 7,), dtype=np.float32
    #     )
    #     self.foot_links = []
    #     for j in range(p.getNumJoints(self.humanoid_id)):
    #         info = p.getJointInfo(self.humanoid_id, j)
    #         link_name = info[12].decode("utf-8")
    #         if "foot" in link_name or "ankle" in link_name:
    #             self.foot_links.append(j)
    #     print(f"DEBUG: Found foot links at indices: {self.foot_links}")

    def _apply_spring_force(self):
        # CONFIG
        SPRING_ANCHOR = 10.0
        START_KP = self.robot_weight * 0.16
        START_KD = self.total_mass * 2.5

        # SCHEDULE
        WARMUP_END = 2_000_000
        DECAY_END = 10_000_000

        # LOGIC
        if self.total_global_steps < WARMUP_END:
            # Phase 1: Full Help
            self.current_kp = START_KP
            self.current_kd = START_KD
        elif self.total_global_steps < DECAY_END:
            # Phase 2: Linear Decay
            progress = (self.total_global_steps - WARMUP_END) / (DECAY_END - WARMUP_END)
            self.current_kp = START_KP * (1.0 - progress)
            self.current_kd = START_KD * (1.0 - progress)
        else:
            self.current_kp = 0.0
            self.current_kd = 0.0
            # Phase 3: Real World (Optimization)
            return

        # PHYSICS
        try:
            link_state = p.getLinkState(self.humanoid_id, 1, computeLinkVelocity=1)
            current_z = link_state[0][2]
            current_vel_z = link_state[6][2]

            error_pos = SPRING_ANCHOR - current_z
            error_vel = 0.0 - current_vel_z

            spring_force_z = (self.current_kp * error_pos) + (
                self.current_kd * error_vel
            )
            spring_force_z = max(0.0, min(spring_force_z, self.robot_weight * 3.0))

            p.applyExternalForce(
                self.humanoid_id,
                1,
                [0, 0, spring_force_z],
                link_state[0],
                p.WORLD_FRAME,
            )

            # Debug Print (Optional: Check first step)
            if self.total_global_steps == 1:
                print(f"!!! FORCE CHECK: Applied {spring_force_z:.1f}N !!!")
        except:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.steps_count = 0
        self.current_energy_cost = 0.0

        # Randomize friction slightly to improve robustness
        p.changeDynamics(self.plane_id, -1, lateralFriction=random.uniform(0.5, 1.2))

        resetJointMotorsAndState(self.humanoid_id)
        for _ in range(50):
            p.stepSimulation()
        return self._get_obs(), {}

    def step(self, action):
        self.total_global_steps += 1

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

        # --- POWER GOVERNOR ---
        # 0M Steps: 10% Strength (Baby)
        # 10M Steps: 100% Strength (Adult)
        # This matches the Spring Decay timeline.
        RAMP_STEPS = 10_000_000
        min_scale = 0.1
        if self.total_global_steps < RAMP_STEPS:
            progress = self.total_global_steps / RAMP_STEPS
            torque_scale = min_scale + (progress * (1.0 - min_scale))
        else:
            torque_scale = 1.0

        # action_idx = 0
        # for j in range(p.getNumJoints(self.humanoid_id)):
        #     name = p.getJointInfo(self.humanoid_id, j)[1].decode("utf-8")
        #     if name not in MAX_TORQUE_MAP:
        #         continue

        #     max_f = np.array(MAX_TORQUE_MAP[name]) * torque_scale

        #     if self.dof_per_joint[j] == 1:
        #         torque = action[action_idx] * max_f
        #         if printStep and printOn:
        #             print(
        #                 f"Joint: {name:<15} | Action: {action[action_idx]:>6.2f} | Torque: {torque:>6.1f} Nm"
        #             )
        #         p.setJointMotorControl2(
        #             self.humanoid_id, j, p.TORQUE_CONTROL, force=torque
        #         )
        #         action_idx += 1

        #     elif self.dof_per_joint[j] == 3:
        #         raw_actions = action[action_idx : action_idx + 3]
        #         torques = raw_actions * max_f
        #         if printStep and printOn:
        #             effort_pct = np.linalg.norm(raw_actions) / math.sqrt(3) * 100
        #             print(
        #                 f"Joint: {name:<15} | Effort: {effort_pct:>5.1f}% | Torques: {np.round(torques, 1)}"
        #             )
        #         p.setJointMotorControlMultiDof(
        #             self.humanoid_id, j, p.TORQUE_CONTROL, force=list(torques)
        #         )
        #         action_idx += 3
        action_idx = 0
        for j in range(p.getNumJoints(self.humanoid_id)):
            name = p.getJointInfo(self.humanoid_id, j)[1].decode("utf-8")
            if name not in MAX_TORQUE_MAP:
                continue

            max_f = np.array(MAX_TORQUE_MAP[name]) * torque_scale

            if self.dof_per_joint[j] == 1:
                # --- 1-DOF JOINT ---
                act_val = action[action_idx]
                torque = act_val * max_f

                if printStep and printOn:
                    # Effort for 1-DOF is just absolute percentage (0 to 100%)
                    effort_pct = abs(act_val) * 100

                    print(
                        f"Joint: {name:<15} | Action: {act_val:>18.2f}   | Effort: {effort_pct:>5.1f}% | Torque: {torque:>22.1f} Nm"
                    )

                p.setJointMotorControl2(
                    self.humanoid_id, j, p.TORQUE_CONTROL, force=torque
                )
                action_idx += 1

            elif self.dof_per_joint[j] == 3:
                # --- 3-DOF JOINT ---
                raw_actions = action[action_idx : action_idx + 3]
                torques = raw_actions * max_f

                if printStep and printOn:
                    # Effort for 3-DOF is Norm / Max_Norm
                    effort_pct = np.linalg.norm(raw_actions) / math.sqrt(3) * 100

                    # Formatting Vectors
                    act_str = f"[{raw_actions[0]:5.2f}, {raw_actions[1]:5.2f}, {raw_actions[2]:5.2f}]"
                    trq_str = (
                        f"[{torques[0]:6.1f}, {torques[1]:6.1f}, {torques[2]:6.1f}]"
                    )

                    print(
                        f"Joint: {name:<15} | Action: {act_str:>21} | Effort: {effort_pct:>5.1f}% | Torque: {trq_str:>25} Nm"
                    )

                p.setJointMotorControlMultiDof(
                    self.humanoid_id, j, p.TORQUE_CONTROL, force=list(torques)
                )
                action_idx += 3

        for _ in range(4):
            self._apply_spring_force()
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
        raw_chest_z = chest_z
        chest_vel_z = chest_state[6][2]  # Z-velocity in world space
        root_z = root_state[0][2]

        head_index = 2
        head_state = p.getLinkState(self.humanoid_id, head_index)

        head_pos, head_orn = head_state[0], head_state[1]
        head_z = head_pos[2]

        chest_z = min(chest_z, TARGET_CHEST)
        root_z = min(root_z, TARGET_ROOT)
        head_z = min(head_z, TARGET_HEAD)

        # 2. Orientation (Uprightness)
        rot_matrix = np.array(p.getMatrixFromQuaternion(chest_orn)).reshape(3, 3)
        chest_up_vector = rot_matrix[:, 2]  # The local Z-axis of the chest link
        uprightness = max(0, np.dot(chest_up_vector, [0, 0, 1]))

        head_rot_matrix = np.array(p.getMatrixFromQuaternion(head_orn)).reshape(3, 3)
        head_up_vector = head_rot_matrix[:, 2]
        head_uprightness = max(0, np.dot(head_up_vector, [0, 0, 1]))

        # CONTACT DETECTION (The Cure for Helicopter Legs)
        feet_contact_reward = 0.0
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
        feet_contact_raw = contact_points

        # 3. REWARD COMPONENTS

        # A. Height (The Goal)
        reward_chest = self.weights["chest_height"] * max(0, chest_z - 0.44)
        reward_root = self.weights["root_height"] * max(0, root_z - 0.36)

        # B. Uprightness (Scaled)
        reward_upright = self.weights["uprightness"] * (
            uprightness * max(0, chest_z - 0.44)
        )

        # C. GATED VELOCITY (Anti-Popcorn Logic)
        # Only reward upward velocity if we are ON THE FLOOR (< 0.6m).
        # Once standing, velocity reward is ZERO.
        # if chest_z < 0.6:
        #     reward_vel = self.weights["chest_vel"] * chest_vel_z
        # else:
        #     reward_vel = 0.0
        # No gate
        reward_vel = self.weights["chest_vel"] * chest_vel_z

        # NEW: NECK/HEAD REWARDS (Simplified)
        reward_neck_height = self.weights["neck_height"] * max(0, head_z - 0.41)
        # 2. Head Orientation
        # Gated by height so we don't reward looking at the ceiling while lying on back.
        reward_neck_orient = self.weights["neck_orientation"] * (
            head_uprightness * max(0, head_z - 0.41)
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
        if chest_z > 0.6:
            reward_feet = self.weights["feet_contact"] * feet_contact_raw
        else:
            reward_feet = 0.0

        # 4. Termination Logic
        done = False
        reward_term = 0.0

        # Terminate if chest touches ground (0.25) or flies away (6.0)
        # if chest_z < 0.25 or raw_chest_z > 6.0:
        #     done = True
        #     reward_term = self.weights["termination_penalty"]
        #     reward_survival = 0.0  # No survival bonus on the death step

        # 400 Steps = 1.6s grace period for start-up
        if self.steps_count > 400:
            if chest_z < 0.65:  # Must stand up
                done = True
                reward_term = self.weights["termination_penalty"]
                reward_survival = 0.0  # No survival bonus on the death step

        if raw_chest_z > 6.0:  # Ceiling Safety
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

    # def _get_obs(self):
    #     angles, velocities = [], []
    #     for j in range(p.getNumJoints(self.humanoid_id)):
    #         js = p.getJointState(self.humanoid_id, j)
    #         angles.append(js[0])
    #         velocities.append(js[1])
    #     _, orn = p.getBasePositionAndOrientation(self.humanoid_id)
    #     _, ang_vel = p.getBaseVelocity(self.humanoid_id)
    #     return np.array(
    #         angles + velocities + list(orn) + list(ang_vel), dtype=np.float32
    #     )

    def _get_obs(self):
        joint_obs = []
        debug_labels = []  # Store names for the printout

        for j in range(p.getNumJoints(self.humanoid_id)):
            info = p.getJointInfo(self.humanoid_id, j)
            jt = info[2]
            name = info[1].decode("utf-8")  # Joint Name

            if jt == p.JOINT_SPHERICAL:
                # 7 Values: 4 Quat + 3 Vel
                state = p.getJointStateMultiDof(self.humanoid_id, j)
                joint_obs.extend(state[0])
                joint_obs.extend(state[1])

                # Add labels
                debug_labels.extend(
                    [f"{name}_qx", f"{name}_qy", f"{name}_qz", f"{name}_qw"]
                )
                debug_labels.extend([f"{name}_vx", f"{name}_vy", f"{name}_vz"])

            elif jt == p.JOINT_REVOLUTE:
                # 2 Values: 1 Ang + 1 Vel
                state = p.getJointState(self.humanoid_id, j)
                joint_obs.append(state[0])
                joint_obs.append(state[1])

                # Add labels
                debug_labels.extend([f"{name}_ang", f"{name}_vel"])

        # Root State
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)
        _, ang_vel = p.getBaseVelocity(self.humanoid_id)

        # Assist Factors
        kp = getattr(self, "current_kp", 0.0)
        kd = getattr(self, "current_kd", 0.0)

        # Merge Data
        final_obs = joint_obs + list(pos) + list(orn) + list(ang_vel) + [kp, kd]

        # --- DIAGNOSTIC LOGGER (Every 2000 steps) ---
        if self.steps_count % 2000 == 0:
            # Add remaining labels
            debug_labels.extend(["Root_X", "Root_Y", "Root_Z"])
            debug_labels.extend(["Root_Qx", "Root_Qy", "Root_Qz", "Root_Qw"])
            debug_labels.extend(["Root_Wx", "Root_Wy", "Root_Wz"])
            debug_labels.extend(["Assist_Kp", "Assist_Kd"])

            print(f"\n--- OBS DEBUG (Step {self.steps_count}) ---")
            print(f"{'INDEX':<6} | {'LABEL':<25} | {'VALUE':<10}")
            print("-" * 45)
            for i, (label, val) in enumerate(zip(debug_labels, final_obs)):
                # Highlight Root Z and Kp visually
                marker = " <---" if label in ["Root_Z", "Assist_Kp"] else ""
                print(f"{i:<6} | {label:<25} | {val:>10.4f}{marker}")
            print("-" * 45 + "\n")

        return np.array(final_obs, dtype=np.float32)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    with utils.PyBulletSim(gui=False) as client:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        plane_id = p.loadURDF("plane.urdf")
        humanoid_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            INITIAL_POSITION,
            START_ORIENTATION,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        print("\n--- Humanoid Diagnostic Info ---")
        utils.print_joint_info(humanoid_id)
        utils.print_dynamics_info(humanoid_id)
        utils.print_link_states(humanoid_id)

        p.setTimeStep(1 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)

        TOTAL_TIMESTEPS = 20_000_000

        env = HumanStandEnv(humanoid_id, plane_id)
        # env = GravityCurriculumWrapper(
        #     env, total_timesteps=TOTAL_TIMESTEPS, start_g=-2.0, end_g=-9.81
        # )
        # env = PuppetMasterWrapper(env, humanoid_id, total_timesteps=TOTAL_TIMESTEPS)

        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=8)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

        # MODEL CONFIGURATION
        # Define the policy architecture
        policy_kwargs = dict(
            activation_fn=th.nn.Tanh,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-0.5,
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            use_sde=False,  # <--- Stops the flailing
            # sde_sample_freq=4,  # smooths noise every 4 steps
            verbose=1,
            # learning_rate=linear_schedule(1.0e-4, min_value=1.0e-6),
            learning_rate=1.0e-4,
            n_steps=4096,
            batch_size=1024,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.000,
            vf_coef=1.0,
            max_grad_norm=0.5,
            tensorboard_log="./logs/",
        )
        print(model.policy)
        print("--- Starting Training with Gated Velocity & Energy Penalty ---")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=RewardLoggerCallback(),
            tb_log_name="V12_Run37",
        )

        model.save("humanoid_v12_final")
        env.save("vec_normalize_v12.pkl")
