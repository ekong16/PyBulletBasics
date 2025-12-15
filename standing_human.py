import pybullet as p
import pybullet_data
import time
import gymnasium
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import utils
import math
import random
from sb3_contrib import RecurrentPPO

###VARS
initial_position = [0, 0, 0.9]

roll = 0
pitch = math.pi / 2  # lying on belly
yaw = 0
start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

max_torque_map = {
    "chest": [25, 25, 25],
    "neck": [5, 5, 5],
    "right_shoulder": [60, 60, 60],
    "left_shoulder": [60, 60, 60],
    "right_elbow": 30,
    "left_elbow": 30,
    "right_hip": [100, 100, 100],
    "left_hip": [100, 100, 100],
    "right_knee": 80,
    "left_knee": 80,
    "right_ankle": [10, 10, 10],
    "left_ankle": [10, 10, 10],
}


# helper
def resetJointMotorsAndState(humanoid_id):
    # A. Reset the Base (Root) Position and Orientation
    p.resetBasePositionAndOrientation(humanoid_id, initial_position, start_orientation)

    # Common neutral values for joint resets
    neutral_quat = [0.0, 0.0, 0.0, 1.0]
    zero_vel_3d = [0.0, 0.0, 0.0]

    # Reset All Joints' State (Position & Velocity)
    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        joint_type = info[2]

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.resetJointState(humanoid_id, j, targetValue=0.0, targetVelocity=0.0)

        elif joint_type == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(
                humanoid_id, j, targetValue=neutral_quat, targetVelocity=zero_vel_3d
            )

    #  DISABLE INTERNAL MOTORS (CONTROL MODE)

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

    # Increase hand and feet friction
    # Replace these with your actual link indices from getJointInfo or URDF
    # Right wrist: 5. Left wrist: 8
    # Right ankle: 11. Left Ankle: 14
    wrist_links = [5, 8]
    ankle_links = [11, 14]

    # Increase friction for wrists and ankles
    for link in wrist_links:
        p.changeDynamics(
            bodyUniqueId=humanoid_id,
            linkIndex=link,
            lateralFriction=10.0,  # sticky enough for pushing
            spinningFriction=0.0,
            rollingFriction=0.0,
        )

    for link in ankle_links:
        p.changeDynamics(
            bodyUniqueId=humanoid_id,
            linkIndex=link,
            lateralFriction=3.0,  # sticky enough for pushing
            spinningFriction=0.0,
            rollingFriction=0.0,
        )


# ----------------
# Gym environment
# ----------------
class HumanStandEnv(gymnasium.Env):
    def __init__(self, humanoid_id, plame_id):
        super().__init__()
        self.humanoid_id = humanoid_id
        self.n_joints = None
        self.action_space = None
        self.observation_space = None
        self.dof_per_joint = None
        self.max_steps = 2048
        self.episode_count = 0
        self.steps_count = 0
        self.plane_id = plame_id

        self.prev_action = None
        self.action_idx_to_joint_name = []
        self.max_forces_flat = []

        self.cum_energy = 0
        self.cum_vel = 0
        self.prev_z_vel_chest = 0

        self.stage = 1

        self.force_factor = 3

        self._init_action_space(humanoid_id)
        self._get_max_forces_flat()

    def _get_max_forces_flat(self):
        for j in range(self.n_joints):
            info = p.getJointInfo(self.humanoid_id, j)
            name = info[1].decode("utf-8")
            if name in max_torque_map:
                max_force = max_torque_map[name]
            else:
                max_force = None

            if self.dof_per_joint[j] == 1:
                assert max_force is not None
                self.max_forces_flat.append(max_force)

            elif self.dof_per_joint[j] > 1:
                assert max_force is not None
                for x in range(self.dof_per_joint[j]):
                    self.max_forces_flat.append(max_force[x])

        print(
            "************",
            len(self.max_forces_flat),
            sum(self.dof_per_joint),
            self.max_forces_flat,
        )

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
        self.episode_count += 1
        self.stage = 1
        print("EPISODE COUNT: ", self.episode_count)
        self.steps_count = 0
        # self.cum_energy = 0
        self.cum_vel = 0
        # Reset PyBullet humanoid
        p.resetBasePositionAndOrientation(
            self.humanoid_id, initial_position, start_orientation
        )

        # Randomize friction of ground to learn more robust moves...
        sampled_mu = random.uniform(0.3, 1.5)
        print(
            f"************** SETTING FRICTION OF PLANE TO: {sampled_mu:.2f} *************"
        )
        p.changeDynamics(
            bodyUniqueId=self.plane_id,
            linkIndex=-1,  # Target the base (or only) body part of the plane
            lateralFriction=sampled_mu,
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
        if self.steps_count % 256 == 0:
            printStep = True
            print("Step no: ", self.steps_count)

        action_idx = 0
        torque_vec = []

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
                max_force = np.array(max_torque_map[name]) * self.force_factor
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
        obs = self._get_obs(printStep=printStep)
        reward = self._get_reward(action, printStep)
        done = self._is_done()
        info = {}
        truncated = False
        self.steps_count += 1
        self.prev_action = action

        return obs, reward, done, truncated, info

    def _get_obs(self, printStep=False):
        angles, velocities = [], []
        for j in range(p.getNumJoints(self.humanoid_id)):
            js = p.getJointState(self.humanoid_id, j)
            angles.append(js[0])
            velocities.append(js[1])

            # if self.dof_per_joint[j] == 1 and printStep:
            #     print(
            #         "**IN OBS",
            #         "Single DOF",
            #         "joint index",
            #         j,
            #         "angles:",
            #         js[0],
            #         "velocities: ",
            #         js[1],
            #     )
            # elif self.dof_per_joint[j] > 1 and printStep:
            #     print(
            #         "**IN OBS",
            #         "Multi DOF",
            #         "joint index",
            #         j,
            #         "angles:",
            #         js[0],
            #         "velocities: ",
            #         js[1],
            #     )
        pos, orn = p.getBasePositionAndOrientation(self.humanoid_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.humanoid_id)
        return np.array(
            angles + velocities + list(orn) + list(ang_vel), dtype=np.float32
        )

    """
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
    """

    def _get_reward(self, action, printStep=False):
        # Action smoothness score
        if self.prev_action is not None:
            prior_forces = self.prev_action * self.max_forces_flat * self.force_factor
        else:
            prior_forces = np.zeros(len(self.max_forces_flat))
            print(prior_forces)

        curr_forces = action * self.max_forces_flat * self.force_factor
        applied_forces_dist = np.linalg.norm(curr_forces - prior_forces)

        if printStep:
            print("curr_forces", curr_forces)
            print("Action", action)
            print("maxForces", self.max_forces_flat)
            print("Applied forces", curr_forces)
            print("curr vs prev applied forces dist", applied_forces_dist)

        # Total Energy Score
        energy_used = np.linalg.norm(curr_forces)
        self.cum_energy += energy_used

        # Chest orn Score
        chest_link_index = 1
        chest_link_state = p.getLinkState(
            self.humanoid_id, chest_link_index, computeLinkVelocity=1
        )
        chest_orn = chest_link_state[1]

        chest_rot_matrix = np.array(p.getMatrixFromQuaternion(chest_orn)).reshape(3, 3)
        chest_z = chest_rot_matrix[:, 2]  # local Z-axis in world frame

        chest_upright_score = np.dot(np.array([0, 0, 1]), chest_z)

        linear_velocity = chest_link_state[6]

        # Vertical Velocity
        vertical_score = linear_velocity[2]
        self.cum_vel += vertical_score

        # Vertical Acceleration
        vertical_acceleration = linear_velocity[2] - self.prev_z_vel_chest
        self.prev_z_vel_chest = vertical_score

        # Joint velocity penalty:
        joint_velocities = []
        for j in range(p.getNumJoints(self.humanoid_id)):
            js = p.getJointState(self.humanoid_id, j)
            joint_velocities.append(js[1])
        vel_norm = np.linalg.norm(joint_velocities)
        # vel_totalmax = math.sqrt(math.pow(10, 2) * len(joint_velocities))
        velocity_penalty = vel_norm

        # Foot velocity penalty
        right_ankle_index = 11
        left_ankle_index = 14
        link_state_right_ankle = p.getLinkState(
            self.humanoid_id,
            right_ankle_index,
            computeLinkVelocity=1,
        )
        right_ankle_linear_velocity = link_state_right_ankle[6]
        link_state_left_ankle = p.getLinkState(
            self.humanoid_id,
            left_ankle_index,
            computeLinkVelocity=1,
        )
        left_ankle_linear_velocity = link_state_left_ankle[6]
        total_ankle_vel = np.linalg.norm(right_ankle_linear_velocity) + np.linalg.norm(
            left_ankle_linear_velocity
        )

        # Self Collision Penalty
        self_contacts = p.getContactPoints(
            bodyA=self.humanoid_id,
            bodyB=self.humanoid_id,
        )
        num_contact_points = len(self_contacts)

        # Flying away penalty
        ground_contacts = p.getContactPoints(
            bodyA=self.humanoid_id, bodyB=self.plane_id
        )

        # Foot removal penalty
        ground_contact_right_foot = p.getContactPoints(
            bodyA=self.humanoid_id, bodyB=self.plane_id, linkIndexA=right_ankle_index
        )
        ground_contact_left_foot = p.getContactPoints(
            bodyA=self.humanoid_id, bodyB=self.plane_id, linkIndexA=left_ankle_index
        )
        foot_floating_penalty = 0
        if len(ground_contact_left_foot) == 0 and len(ground_contact_right_foot) == 0:
            foot_floating_penalty = 10

        # Stage 2: Root Velocity and Chest orientation...
        if self.stage > 1:
            root_link_index = 0
            root_link_state = p.getLinkState(
                self.humanoid_id, root_link_index, computeLinkVelocity=1
            )
            root_orn = root_link_state[1]
            root_rot_matrix = np.array(p.getMatrixFromQuaternion(root_orn)).reshape(
                3, 3
            )
            root_z = root_rot_matrix[:, 2]  # local Z-axis in world frame
            root_upright_score = np.dot(np.array([0, 0, 1]), root_z)
            root_linear_velocity = root_link_state[6][2]

        # Stage 3: Head Orientation and
        neck_link_index = 2
        neck_link_state = p.getLinkState(
            self.humanoid_id, neck_link_index, computeLinkVelocity=1
        )
        neck_orn = neck_link_state[1]
        neck_rot_matrix = np.array(p.getMatrixFromQuaternion(neck_orn)).reshape(3, 3)
        neck_z = neck_rot_matrix[:, 2]  # local Z-axis in world frame
        neck_upright_score = np.dot(np.array([0, 0, 1]), neck_z)

        fly_away_penalty = 0
        if len(ground_contacts) == 0:
            fly_away_penalty = 100

        reward = (
            self.cum_vel / energy_used
            + 1 * vertical_score / energy_used
            + 1 * vertical_acceleration / energy_used
            # + 7 * chest_upright_score
            - applied_forces_dist / 1000
            # - self.cum_energy / 100
            - 1 * energy_used
            - 1 * total_ankle_vel
            - 1 * velocity_penalty
            - 10 * num_contact_points
            - fly_away_penalty
            - foot_floating_penalty
        )

        if self.stage > 1:
            reward += 1 * chest_upright_score
            reward += 1 * root_linear_velocity / energy_used

        if self.stage > 2:
            reward += 1 * root_upright_score
            reward += 1 * neck_upright_score

        if self.stage == 1 and self.cum_vel > 150:
            print("**************** STAGE 2 ACHIEVED ***************")
            self.stage = 2
            reward += 1_000

        if self.stage > 1 and self.cum_vel < 125:
            reward -= 10
            if printStep:
                print("***** STAGE 2 PUNISHMENT FOR REGRESSING IN HEIGHT ********")

        if self.stage == 2 and self.cum_vel > 225:
            self.stage = 3
            reward += 1_000
            print("**** STAGE 3 Achieved *********")

        if printStep:
            print(f"Cum Vel: {self.cum_vel:.2f}. vel ={vertical_score:.2f} ")
            print(f"Vertical vel score: {vertical_score / energy_used:.2f}")
            print(f"Vertical acc score: {vertical_acceleration / energy_used:.2f}")
            # print(
            #     f"Chest local z: {chest_z}. Chest Reward: {chest_upright_score: .2f}",
            # )
            print(f"Smoothness: {applied_forces_dist: .2f}.")
            print(f"energy penalty: {energy_used: .2f}.")
            print(f"Ankle movement: {total_ankle_vel: .2f}.")
            print(f"velocity penalty: {velocity_penalty: .2f}.")
            print(f"Self Contact points: {num_contact_points: .2f}.")
            print(f"Fly away penalty: {fly_away_penalty: .2f}.")
            print(f"Foot floating penalty: {foot_floating_penalty: .2f}.")

            if self.stage > 1:
                print(f"Chest Upright score {chest_upright_score:.2f}.")
                print(f"Root lin vel: {root_linear_velocity:.2f}")

            if self.stage > 2:
                print(f"Root Upright score {root_upright_score:.2f}.")
                print(f"Head Upright score {neck_upright_score:.2f}.")

            print(f"Cum energy: {self.cum_energy: .2f}.")
            print(f"Total reward: {reward: .2f}.")

        return reward

    def _is_done(self):
        torso_height = p.getBasePositionAndOrientation(self.humanoid_id)[0][2]
        if torso_height < 0.01:
            return True
        if self.steps_count > self.max_steps:
            return True
        # if self.cum_energy > 100_000:
        #     return True


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
        env = HumanStandEnv(my_humanoid_id, planeId)

        # policy_kwargs = dict(log_std_init=np.log(10))  # std ≈ 100 N·m

        # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
        print("MODEL POLICY ******** ", model.policy)
        model.learn(total_timesteps=204_800)
        model.save("humanoid_final.zip")
