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

max_torque_map = {
    "chest": [20, 20, 20],
    "neck": [3, 3, 3],
    "right_shoulder": [35, 35, 35],
    "left_shoulder": [35, 35, 35],
    "right_elbow": 20,
    "left_elbow": 20,
    "right_hip": [60, 60, 60],
    "left_hip": [60, 60, 60],
    "right_knee": 45,
    "left_knee": 45,
    "right_ankle": [5, 5, 5],
    "left_ankle": [5, 5, 5],
}


# helper
def resetJointMotorsAndState(humanoid_id):
    # --- 1. PHASE 1: RESET POSITIONS AND VELOCITIES (STATE) ---
    # A. Reset the Base (Root) Position and Orientation
    p.resetBasePositionAndOrientation(humanoid_id, initial_position, start_orientation)

    # Common neutral values for joint resets
    neutral_quat = [0.0, 0.0, 0.0, 1.0]
    zero_vel_3d = [0.0, 0.0, 0.0]

    # B. Reset All Joints' State (Position & Velocity)
    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        joint_type = info[2]

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.resetJointState(humanoid_id, j, targetValue=0.0, targetVelocity=0.0)

        elif joint_type == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(
                humanoid_id, j, targetValue=neutral_quat, targetVelocity=zero_vel_3d
            )

    # --- 2. PHASE 2: DISABLE INTERNAL MOTORS (CONTROL MODE) ---

    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        joint_type = info[2]

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            # 1-DoF joints
            p.setJointMotorControl2(
                humanoid_id, j, controlMode=p.VELOCITY_CONTROL, force=0
            )

        elif joint_type == p.JOINT_SPHERICAL:
            # 3-DoF joint
            p.setJointMotorControlMultiDof(
                humanoid_id,
                j,
                controlMode=p.POSITION_CONTROL,  # Use POSITION_CONTROL to clear constraints
                targetPosition=[0, 0, 0, 1],  # Quaternion for "no rotation"
                force=[0, 0, 0],  # Max force 0 = motor disabled
            )


# ----------------
# Gym environment
# ----------------
class HumanStandEnv(gymnasium.Env):
    def __init__(self, humanoid_id):
        super().__init__()
        self.humanoid_id = humanoid_id
        self.n_joints = None
        self.action_space = None
        self.observation_space = None
        self.dof_per_joint = None
        self.max_steps = 2048
        self.steps_count = 0

        self._init_action_space(humanoid_id)

    def _init_action_space(self, humanoid_id):
        self.n_joints = p.getNumJoints(humanoid_id)
        dof_per_joint = []

        for j in range(p.getNumJoints(humanoid_id)):
            info = p.getJointInfo(humanoid_id, j)
            joint_type = info[2]

            if joint_type == p.JOINT_REVOLUTE:
                dof_per_joint.append(1)
            elif joint_type == p.JOINT_SPHERICAL:
                dof_per_joint.append(3)
            else:
                dof_per_joint.append(0)  # fixed or prismatic
        self.dof_per_joint = dof_per_joint

        self.action_space = spaces.Box(
            low=-200, high=200, shape=(sum(self.dof_per_joint),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_joints * 2 + 7,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium compatibility
        self.steps_count = 0
        # Reset PyBullet humanoid
        p.resetBasePositionAndOrientation(
            self.humanoid_id, initial_position, start_orientation
        )

        resetJointMotorsAndState(self.humanoid_id)

        # Settle the robot
        print("Settling robot in RESET")
        for _ in range(100):
            p.stepSimulation()
        print("DONE Settling robot in RESET")

        obs = self._get_obs()  # get initial observation
        info = {}
        return obs, info

    def step(self, action):
        printStep = False
        if self.steps_count % 1024 == 0:
            printStep = True
            print("Step no: ", self.steps_count)

        action_idx = 0
        torque_vec = []

        force_scale_factor = 1.0
        # if self.steps_count < 512:
        #     force_scale_factor = 1.0
        # elif self.steps_count < 1536:
        #     force_scale_factor = 2.0
        # else:
        #     force_scale_factor = 1.0

        for j in range(self.n_joints):
            info = p.getJointInfo(self.humanoid_id, j)
            name = info[1].decode("utf-8")
            if name in max_torque_map:
                max_force = np.array(max_torque_map[name]) * force_scale_factor
            else:
                max_force = None

            if self.dof_per_joint[j] == 1:
                assert max_force is not None

                torque = action[action_idx] * max_force

                if printStep:
                    print("Single DOF Joint: ", j, name, torque)

                p.setJointMotorControl2(
                    self.humanoid_id, j, p.TORQUE_CONTROL, force=torque
                )
                action_idx += 1

            elif self.dof_per_joint[j] > 1:
                assert max_force is not None

                while len(torque_vec) < self.dof_per_joint[j]:
                    torqueComponent = action[action_idx]
                    torque_vec.append(float(torqueComponent))
                    action_idx += 1

                assert len(max_force) == 3 and len(torque_vec) == 3

                torque_vec = np.array(torque_vec)
                max_force = np.array(max_force)
                torque_vec *= max_force

                if printStep:
                    print("Multi DOF Joint: ", j, name, torque_vec)
                p.setJointMotorControlMultiDof(
                    self.humanoid_id, j, p.TORQUE_CONTROL, force=list(torque_vec)
                )
                torque_vec = []
        if printStep:
            print("\n")

        p.stepSimulation()
        # time.sleep(1 / 240.0)  # visualization
        obs = self._get_obs()
        reward = self._get_reward(action, printStep)
        done = self._is_done()
        info = {}
        truncated = False
        self.steps_count += 1

        return obs, reward, done, truncated, info

    def _get_obs(self):
        angles, velocities = [], []
        for j in range(p.getNumJoints(self.humanoid_id)):
            js = p.getJointState(self.humanoid_id, j)
            angles.append(js[0])
            velocities.append(js[1])
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.humanoid_id)
        return np.array(
            angles + velocities + list(orn) + list(ang_vel), dtype=np.float32
        )

    def _get_reward(self, action, printStep=False):
        # Chest Upright Score
        root_link_index = 0
        root_pos, root_orn = p.getLinkState(
            self.humanoid_id, root_link_index, computeLinkVelocity=1
        )[:2]

        root_lin_vel, root_ang_vel = p.getLinkState(
            self.humanoid_id, root_link_index, computeLinkVelocity=1
        )[6:8]

        root_rot_matrix = np.array(p.getMatrixFromQuaternion(root_orn)).reshape(3, 3)
        root_z = root_rot_matrix[:, 2]  # local Z-axis in world frame

        up_vector = np.array([0, 0, 1])
        if printStep:
            print("ROOT Z:", root_z)
        upright_score = max(0, np.dot(up_vector, root_z))

        # root_height_score
        current_root_height = root_pos[2]
        max_root_height = 1.5  # meters
        root_height_score = np.tanh(current_root_height / max_root_height)

        # Angular velocity penalty (prevent falling)
        root_ang_vel = np.array(root_ang_vel)
        root_max_ang_vel = 5.0  # rad/s, adjust to what you consider “fast/flailing”
        root_ang_vel_norm = np.linalg.norm(root_ang_vel) / root_max_ang_vel
        root_ang_vel_penalty = np.clip(root_ang_vel_norm, 0, 1)  # now between 0 and 1

        # COM Score
        left_ankle_index = 14
        right_ankle_index = 11

        left_ankle_pos, _ = p.getLinkState(self.humanoid_id, left_ankle_index)[:2]
        right_ankle_pos, _ = p.getLinkState(self.humanoid_id, right_ankle_index)[:2]

        ankle_center = np.mean([left_ankle_pos, right_ankle_pos], axis=0)

        ankle_center_xy = ankle_center[:2]
        chest_xy = root_pos[:2]

        ankle_center_chest_dist_xy = np.linalg.norm(ankle_center_xy - chest_xy)
        normalized_com_score = 1.0 - np.tanh(ankle_center_chest_dist_xy)

        # Foot Height Penalty
        avg_foot_height = (left_ankle_pos[2] + right_ankle_pos[2]) / 2.0
        max_foot_height = 0.5
        foot_height_penalty = np.tanh(avg_foot_height / max_foot_height)

        # Head height Score
        neck_link_index = 2
        neck_link_pos, _ = p.getLinkState(self.humanoid_id, neck_link_index)[:2]
        current_height = neck_link_pos[2]
        max_head_height = 3.0  # meters
        head_height_score = np.tanh(current_height / max_head_height)  # smooth, 0 -> 1

        # Head stability score
        neck_angular_velocity = p.getLinkState(
            self.humanoid_id, neck_link_index, computeLinkVelocity=1
        )[7]  # angular velocity
        neck_max_ang_vel = 3.0  # rad/s, adjust to what you consider “fast/flailing”
        neck_ang_vel_norm = np.linalg.norm(neck_angular_velocity) / neck_max_ang_vel
        neck_ang_vel_penalty = np.clip(neck_ang_vel_norm, 0, 1)  # now between 0 and 1

        # Total force penalty
        tau_penalty = np.mean(np.square(action))

        weighted_upright_score = 0.5 * upright_score
        weighted_com_score = 0.3 * normalized_com_score
        weighted_head_height_score = 0.1 * head_height_score
        weighted_root_height_score = 0.1 * root_height_score
        weighted_foot_height_penalty = -0.1 * foot_height_penalty

        reward = (
            weighted_upright_score
            + weighted_com_score
            + weighted_head_height_score
            + weighted_root_height_score
            + weighted_foot_height_penalty
        )

        if printStep:
            print(
                f"Upright score: {upright_score:.2f}. Weighted: {weighted_upright_score:.2f} \n"
                # f"Chest Ang Vel Pen: {chest_ang_vel_penalty:.2f}\n"
                f"Com Score: {normalized_com_score:.2f}. Weighted: {weighted_com_score:.2f}\n"
                f"Head Height Score: {head_height_score:.2f}. Weighted: {weighted_head_height_score:.2f}\n"
                f"Root Height Score: {root_height_score:.2f}. Weighted: {weighted_root_height_score:.2f}\n"
                f"Foot Height Penalty: {foot_height_penalty:.2f}. Weighted: {weighted_foot_height_penalty:.2f}\n"
                # f"Neck Ang Vel Pen: {neck_ang_vel_penalty:.2f}\n"
                # f"Tau Pen: {tau_penalty:.2f}\n"
                f"Total Score : {reward:.2f}"
            )

        return reward

    def _is_done(self):
        torso_height = p.getBasePositionAndOrientation(self.humanoid_id)[0][2]
        if torso_height < 0.01:
            return True
        if self.steps_count > self.max_steps:
            return True


if __name__ == "__main__":
    with utils.PyBulletSim(gui=False) as client:
        # --- Simulation Initialization ---
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)

        planeId = p.loadURDF("plane.urdf")

        # Enable self-collision
        flags = p.URDF_USE_SELF_COLLISION
        my_humanoid_id = p.loadURDF(
            "humanoid/humanoid.urdf", initial_position, start_orientation, flags=flags
        )

        # resetJointMotorsAndState(my_humanoid_id)

        utils.print_joint_info(my_humanoid_id)
        utils.print_link_states(my_humanoid_id)
        utils.print_dynamics_info(my_humanoid_id)

        p.setTimeStep(1 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)

        # resetJointMotorsAndState(my_humanoid_id)
        # for _ in range(10000):
        #     p.stepSimulation()
        # print("humanoid should be settled.")

        # ----------------
        # Train PPO
        # ----------------
        env = HumanStandEnv(my_humanoid_id)

        # policy_kwargs = dict(log_std_init=np.log(10))  # std ≈ 100 N·m

        # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1_000_000)
        model.save("humanoid_final.zip")
