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

# ==========================================
# GLOBAL CONFIG
# ==========================================
INITIAL_POSITION = [0, 0, 0.28]
ROLL, PITCH, YAW = 0, math.pi / 2, 0
START_ORIENTATION = p.getQuaternionFromEuler([ROLL, PITCH, YAW])

# POSITION CONTROL SETTINGS
# PD logic: Force = Kp * (target - current) - Kd * velocity
UNIVERSAL_JOINT_LIMIT = 1.0  # +/- 1 Radian range for all joints
KP = 0.2
KD = 1.0

# FORCE CURRICULUM (Hand of God)
INITIAL_ASSIST_FORCE = 100.0
ASSIST_DECAY_STEPS = 2_000_000


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

    # Rubber Floor Friction
    for link in range(p.getNumJoints(humanoid_id)):
        p.changeDynamics(humanoid_id, link, lateralFriction=1.0)


class HumanStandEnv(gymnasium.Env):
    def __init__(self, humanoid_id, plane_id):
        super().__init__()
        self.humanoid_id = humanoid_id
        self.plane_id = plane_id
        self.max_steps = 1024
        self.steps_count = 0
        self.total_steps_trained = 0
        self.episode_count = 0

        self.weights = {
            "chest_height": 2.0,
            "root_height": 1.0,
            "neck_height": 1.5,
            "uprightness": 1.0,
            "neck_orientation": 1.0,
            "chest_vel": 0.1,
            "feet_contact": 1.0,
            "energy_cost": -0.05,
            "smoothness_cost": -0.1,
            "survival_bonus": 0.5,
            "termination_penalty": -100.0,
        }

        self._init_spaces()

    def _init_spaces(self):
        self.joint_config = []
        self.foot_links = []
        num_rev = 0
        num_sph = 0

        numJoints = p.getNumJoints(self.humanoid_id)

        for j in range(numJoints):
            info = p.getJointInfo(self.humanoid_id, j)
            name = info[1].decode("utf-8")
            jt = info[2]  # Joint Type

            if "foot" in info[12].decode("utf-8") or "ankle" in info[12].decode(
                "utf-8"
            ):
                self.foot_links.append(j)

            if jt in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_config.append({"id": j, "type": "REV", "name": name})
                num_rev += 1
            elif jt == p.JOINT_SPHERICAL:
                self.joint_config.append({"id": j, "type": "SPH", "name": name})
                num_sph += 1

        # 1. Action Space: Rev(1) + Sph(3)
        action_dim = (num_rev * 1) + (num_sph * 3)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.last_action = np.zeros(action_dim)

        # 2. Observation Space Math
        obs_joints = numJoints * 2
        obs_base = 7
        obs_memory = action_dim

        calculated_dim = obs_joints + obs_base + obs_memory

        # 3. Sanity Check
        actual_obs = self._get_obs()
        if actual_obs.shape[0] != calculated_dim:
            raise ValueError(
                f"Math Error! Calculated: {calculated_dim}, Actual: {actual_obs.shape[0]}"
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(calculated_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("RESETING")
        self.episode_count += 1
        self.steps_count = 0
        self.last_action = np.zeros(self.action_space.shape[0])

        p.changeDynamics(self.plane_id, -1, lateralFriction=random.uniform(0.8, 1.2))
        resetJointMotorsAndState(self.humanoid_id)
        for _ in range(50):
            p.stepSimulation()

        return self._get_obs(), {}

    def step(self, action):
        # 1. HAND OF GOD (Decaying Assist)
        assist_pct = max(0, 1.0 - (self.total_steps_trained / ASSIST_DECAY_STEPS))
        current_assist = INITIAL_ASSIST_FORCE * assist_pct
        if current_assist > 0:
            p.applyExternalForce(
                self.humanoid_id, 1, [0, 0, current_assist], [0, 0, 0], p.WORLD_FRAME
            )

        # 2. POSITION CONTROL (The Paper Strategy)
        ptr = 0
        for joint in self.joint_config:
            j_idx = joint["id"]
            if joint["type"] == "REV":
                target = action[ptr] * UNIVERSAL_JOINT_LIMIT
                ptr += 1
                p.setJointMotorControl2(
                    self.humanoid_id,
                    j_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=100,
                    positionGain=KP,
                    velocityGain=KD,
                )
            elif joint["type"] == "SPH":
                raws = action[ptr : ptr + 3]
                ptr += 3
                target_quat = p.getQuaternionFromEuler(raws * UNIVERSAL_JOINT_LIMIT)
                p.setJointMotorControlMultiDof(
                    self.humanoid_id,
                    j_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_quat,
                    force=[100, 100, 100],
                    positionGain=KP,
                    velocityGain=KD,
                )

        for _ in range(4):
            p.stepSimulation()

        # 3. COSTS & MEMORY
        self.current_energy_cost = np.sum(np.square(action))
        self.current_smoothness_cost = np.sum(np.square(action - self.last_action))
        self.last_action = action.copy()

        self.steps_count += 1
        self.total_steps_trained += 1

        obs = self._get_obs()
        reward, done, decomposition = self._get_reward()
        return (
            obs,
            reward,
            done,
            self.steps_count >= self.max_steps,
            {"decomposition": decomposition},
        )

    def _get_reward(self):
        chest = p.getLinkState(self.humanoid_id, 1, computeLinkVelocity=1)
        root = p.getLinkState(self.humanoid_id, 0)
        head = p.getLinkState(self.humanoid_id, 2)

        chest_z, chest_orn = chest[0][2], chest[1]
        root_z = root[0][2]
        head_z, head_orn = head[0][2], head[1]
        chest_vel_z = chest[6][2]

        # Uprightness Math
        def get_upright(orn):
            rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            return max(0, np.dot(rot[:, 2], [0, 0, 1]))

        upright = get_upright(chest_orn)
        h_upright = get_upright(head_orn)

        contacts = 0
        for l in self.foot_links:
            if (
                len(
                    p.getContactPoints(
                        bodyA=self.humanoid_id, bodyB=self.plane_id, linkIndexA=l
                    )
                )
                > 0
            ):
                contacts += 1

        r_chest = self.weights["chest_height"] * max(0, chest_z - 0.132)
        r_root = self.weights["root_height"] * max(0, root_z - 0.108)
        r_neck = self.weights["neck_height"] * max(0, head_z - 0.123)
        r_upright = self.weights["uprightness"] * (upright * max(0, chest_z - 0.132))
        r_neck_orn = self.weights["neck_orientation"] * (
            h_upright * max(0, head_z - 0.123)
        )

        r_vel = self.weights["chest_vel"] * chest_vel_z if chest_z < 0.6 else 0.0
        r_feet = self.weights["feet_contact"] * contacts if chest_z > 0.3 else 0.0

        r_energy = self.weights["energy_cost"] * self.current_energy_cost
        r_smooth = self.weights["smoothness_cost"] * self.current_smoothness_cost
        r_survival = self.weights["survival_bonus"]

        done = False
        r_term = 0.0
        if chest_z < 0.1 or chest_z > 1.8:
            done = True
            r_term = self.weights["termination_penalty"]
            r_survival = 0.0

        total = (
            r_chest
            + r_root
            + r_neck
            + r_upright
            + r_neck_orn
            + r_vel
            + r_feet
            + r_energy
            + r_smooth
            + r_survival
            + r_term
        )

        decomp = {
            "01_chest_height": r_chest,
            "02_root_height": r_root,
            "03_uprightness": r_upright,
            "04_chest_velocity": r_vel,
            "05_energy_penalty": r_energy,
            "06_survival_bonus": r_survival,
            "07_feet_contact": r_feet,
            "08_term_penalty": r_term,
            "09_neck_height": r_neck,
            "10_neck_orientation": r_neck_orn,
            "11_smoothness": r_smooth,
            "z_TOTAL": total,
        }
        return total, done, decomp

    def _get_obs(self):
        data = []
        for j in range(p.getNumJoints(self.humanoid_id)):
            js = p.getJointState(self.humanoid_id, j)
            # Flatten Pos/Vel
            if hasattr(js[0], "__len__"):
                data.extend(js[0])
            else:
                data.append(js[0])
            if hasattr(js[1], "__len__"):
                data.extend(js[1])
            else:
                data.append(js[1])

        _, base_orn = p.getBasePositionAndOrientation(self.humanoid_id)
        _, ang_vel = p.getBaseVelocity(self.humanoid_id)

        # Angles/Vels + BaseOrn(4) + BaseAngVel(3) + LastAction
        return np.concatenate(
            (data, list(base_orn), list(ang_vel), self.last_action)
        ).astype(np.float32)


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
            total_timesteps=100_000,
            callback=RewardLoggerCallback(),
            tb_log_name="V14_Run1_test",
        )

        model.save("humanoid_v14_final")
        env.save("vec_normalize_v14.pkl")
