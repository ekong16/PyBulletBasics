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
import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn.functional as F
from transformers import AutoVideoProcessor, AutoModel
from PIL import Image

# ==========================================
# GLOBAL VARS & CONFIG
# ==========================================
INITIAL_POSITION = utils.SimConfig.INITIAL_POS
ROLL, PITCH, YAW = (
    utils.SimConfig.START_ORI[0],
    utils.SimConfig.START_ORI[1],
    utils.SimConfig.START_ORI[2],
)
START_ORIENTATION = p.getQuaternionFromEuler([ROLL, PITCH, YAW])

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

TARGET_HEAD = 1.94
TARGET_CHEST = 1.60
TARGET_ROOT = 1.23

# ==========================================
# 1. CONFIG & V-JEPA LOADING
# ==========================================
# HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
# device = "cuda" if th.cuda.is_available() else "cpu"

# print(f"🧠 Loading V-JEPA 2 on {device}...")
# processor = AutoVideoProcessor.from_pretrained(HF_REPO)
# vjepa_model = AutoModel.from_pretrained(HF_REPO).to(device).eval()


# def get_target_latent():
#     target_path = "poses/standing_pose.jpg"
#     img = Image.open(target_path).convert("RGB")
#     video_clip = np.asarray([img] * 16)
#     inputs = processor(video_clip, return_tensors="pt").to(device)
#     with th.no_grad():
#         return vjepa_model(**inputs).last_hidden_state

my_jepa_model = utils.JEPAEngine()
JEPA_VERBOSE = True
Target_Video = None
No_Robot_Video = None


def get_target_latent():
    global Target_Video
    target_path = "poses/standing_pose.jpg"
    img = Image.open(target_path).convert("RGB")
    video_clip = np.asarray([img] * 16)
    Target_Video = video_clip
    ret = my_jepa_model.get_latent(video_clip, verbose=JEPA_VERBOSE)
    return ret


def get_no_robot_test():
    global No_Robot_Video
    target_path = "poses/no_robot.jpg"
    img = Image.open(target_path).convert("RGB")
    video_clip = np.asarray([img] * 16)
    No_Robot_Video = video_clip
    ret = my_jepa_model.get_latent(video_clip, verbose=JEPA_VERBOSE)
    return ret


TARGET_LATENT = get_target_latent()
assert Target_Video is not None

# NO_ROBOT_LATENT = get_no_robot_test()


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


def save_labeled_video_OLD(
    video_buffer, Target_Video, mse, episode, folder="recordings"
):
    """
    Saves the 16-frame buffer to disk with the MSE burned into the corner.
    """
    assert video_buffer is not None
    assert Target_Video is not None
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"ep_{episode:03d}_mse_{mse:.4f}.mp4")

    # Define the codec and create VideoWriter object (H.264)
    # 256x256 is your current resolution
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(filepath, fourcc, 8.0, (512, 256))

    def to_uint8(buf):
        if buf.dtype != np.uint8:
            return (
                (buf * 255).astype(np.uint8)
                if buf.max() <= 1.0
                else buf.astype(np.uint8)
            )
        return buf

    buf_a = to_uint8(video_buffer)
    buf_b = to_uint8(Target_Video)

    for i in range(16):
        # 1. Grab frames and ensure they are contiguous for C++
        frame_l = np.ascontiguousarray(buf_a[i])
        frame_r = np.ascontiguousarray(buf_b[i])

        # 2. Convert both from RGB to BGR for OpenCV
        bgr_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
        bgr_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

        # 3. Horizontal Stack (RL Result on Left, Target on Right)
        canvas = np.hstack((bgr_l, bgr_r))

        # 4. Burn-in Info (Top Left of the left frame)
        label_top = "6x Slow Motion (8 FPS)"
        label_bot = f"Ep: {episode} | MSE: {mse:.4f}"

        # Smooth, high-contrast text settings
        font, scale, thick, line = cv2.FONT_HERSHEY_DUPLEX, 0.45, 1, cv2.LINE_AA

        for text, pos in [(label_top, (10, 25)), (label_bot, (10, 50))]:
            # Outline for legibility
            cv2.putText(canvas, text, pos, font, scale, (0, 0, 0), thick + 2, line)
            # Main white text
            cv2.putText(canvas, text, pos, font, scale, (255, 255, 255), thick, line)

        out.write(canvas)

    out.release()
    print(f"🎬 Video saved to {filepath}")


import cv2
import numpy as np
import os


def save_labeled_video(
    video_buffer, Target_Video, mse, episode, step, folder="recordings"
):
    """
    Saves the 16-frame buffer side-by-side with a border, target label, and step count.
    """
    assert video_buffer is not None
    assert Target_Video is not None
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"ep_{episode:03d}_step_{step}_mse_{mse:.4f}.mp4")

    # Define codec (avc1 for H.264)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    # DIMENSIONS: 256 (Left) + 4 (Border) + 256 (Right) = 516 Width
    video_width = 516
    video_height = 256
    out = cv2.VideoWriter(filepath, fourcc, 8.0, (video_width, video_height))

    def to_uint8(buf):
        if buf.dtype != np.uint8:
            return (
                (buf * 255).astype(np.uint8)
                if buf.max() <= 1.0
                else buf.astype(np.uint8)
            )
        return buf

    buf_a = to_uint8(video_buffer)
    buf_b = to_uint8(Target_Video)

    # Styling settings
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.42  # Slightly smaller to accommodate the third line
    thick = 1
    line_type = cv2.LINE_AA

    for i in range(16):
        frame_l_rgb = np.ascontiguousarray(buf_a[i])
        frame_r_rgb = np.ascontiguousarray(buf_b[i])

        # Convert to BGR for OpenCV
        bgr_l = cv2.cvtColor(frame_l_rgb, cv2.COLOR_RGB2BGR)
        bgr_r = cv2.cvtColor(frame_r_rgb, cv2.COLOR_RGB2BGR)

        # 1. THE BORDER ASSEMBLY
        border_width = 4
        spacer = np.full((video_height, border_width, 3), 50, dtype=np.uint8)
        canvas = np.hstack((bgr_l, spacer, bgr_r))

        # 2. THE TEXT OVERLAYS
        # Left Side Labels
        labels_l = [
            (f"Episode: {episode}", 22),
            (f"Step: {step}", 42),
            (f"Latent MSE: {mse:.4f}", 62),
        ]

        # Right Side Label
        label_target = "Target Pose"
        x_right = bgr_l.shape[1] + border_width + 10
        target_color = (0, 215, 255)  # Gold/Yellow in BGR

        # DRAW LEFT LABELS
        for text, y_pos in labels_l:
            cv2.putText(
                canvas, text, (10, y_pos), font, scale, (0, 0, 0), thick + 2, line_type
            )
            cv2.putText(
                canvas,
                text,
                (10, y_pos),
                font,
                scale,
                (255, 255, 255),
                thick,
                line_type,
            )

        # DRAW RIGHT LABEL
        cv2.putText(
            canvas,
            label_target,
            (x_right, 22),
            font,
            scale,
            (0, 0, 0),
            thick + 2,
            line_type,
        )
        cv2.putText(
            canvas,
            label_target,
            (x_right, 22),
            font,
            scale,
            target_color,
            thick,
            line_type,
        )

        out.write(canvas)

    out.release()
    print(f"🎬 Enhanced Video saved: {filepath}")


def Apply128kgMasses(humanoid_id):
    """
    The 'Power of Two' Build.
    Target Total Mass: EXACTLY 128.0 kg.

    Distribution Strategy:
    - Root (33kg) acts as the primary CoM anchor.
    - Legs (57kg) are kept heavy to prevent 'stilts' effect.
    - Chest (20kg) is lightened to reduce the load on your 400Nm ankles.
    """
    mass_map = {
        # --- THE ANCHOR (33.0 kg) ---
        "root": 33.0,
        # --- THE UPPER BODY (38.0 kg Total) ---
        # Chest reduced to 20kg to help stability.
        "chest": 20.0,
        "neck": 4.0,  # Heavy Head (4kg)
        "right_shoulder": 3.5,
        "left_shoulder": 3.5,
        "right_elbow": 2.5,
        "left_elbow": 2.5,
        "right_wrist": 1.0,
        "left_wrist": 1.0,
        # --- THE BASE (57.0 kg Total) ---
        # Legs are ~45% of total mass. Good for stability.
        "right_hip": 16.0,
        "left_hip": 16.0,
        "right_knee": 10.0,
        "left_knee": 10.0,
        "right_ankle": 2.5,
        "left_ankle": 2.5,
    }

    print("\n--- APPLYING 128kg MASS DISTRIBUTION ---")
    total_mass = 0.0

    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        link_name = info[12].decode("utf-8")

        # Default fallback (very light)
        target_mass = 0.1

        for key, mass in mass_map.items():
            if key in link_name:
                target_mass = mass
                break

        p.changeDynamics(humanoid_id, j, mass=target_mass)
        total_mass += target_mass

    # Base (-1) - The final rounding error handler
    # We set it to 0.0 to keep the sum clean, or 0.1 if bullet complains.
    # (Physics engines usually prefer non-zero mass, so we'll use a tiny epsilon elsewhere
    # but for your 128kg goal, the links above sum to 128.0 exactly).
    p.changeDynamics(humanoid_id, -1, mass=1e-3)

    print(f"--- TOTAL MASS: {total_mass:.1f} kg ---")
    print(f"--- 128.0 KG LOCKED IN. ---")
    return total_mass


def resetJointMotorsAndState(humanoid_id):
    p.resetBasePositionAndOrientation(humanoid_id, INITIAL_POSITION, START_ORIENTATION)
    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        jt = info[2]

        # --- APPLY DAMPING ---
        # 1.0 is a good starting point. It eats up kinetic energy.
        # This allows high torque (strength) but prevents high velocity (flailing).
        p.changeDynamics(
            humanoid_id,
            j,
            jointDamping=0.5,
            angularDamping=0.1,  # Resists the link's tendency to spin wildly
            # Set to a very high number to stop the engine from 'clamping'
            # and causing the 'flying' teleportation glitch.
            # maxJointVelocity=50.0,
        )

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
        p.changeDynamics(humanoid_id, link, lateralFriction=5.0, rollingFriction=1.0)

    for link in [11, 14]:
        p.changeDynamics(
            humanoid_id,
            link,
            lateralFriction=10.0,
            contactStiffness=30000,  # Prevents the "infinite hardness" bounce
            contactDamping=1000,  # Absorbs the impact energy at the foot-floor interface
        )


class HumanStandEnv(gymnasium.Env):
    def __init__(self, humanoid_id, plane_id, video_dir):
        super().__init__()
        self.humanoid_id = humanoid_id
        self.plane_id = plane_id
        self.video_dir = video_dir
        self.max_steps = 9  # Increased slightly to allow for stability testing
        self.steps_count = 0
        self.episode_count = 0
        self.camera = utils.PyBulletCamera()  # Your existing utility
        self.last_action = None

        self.total_mass = Apply128kgMasses(self.humanoid_id)
        self.robot_weight = self.total_mass * 9.81

        print(f"DEBUG: Robot Total Mass: {self.total_mass:.2f} kg")
        print(f"DEBUG: Robot Weight: {self.robot_weight:.2f} N")

        self._init_spaces()

    def _init_spaces(self):
        n_joints = p.getNumJoints(self.humanoid_id)
        self.joint_indices = []
        self.dof_per_joint = []
        obs_dim = 0

        for j in range(n_joints):
            self.joint_indices.append(j)
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
        self.last_action = np.zeros(sum(self.dof_per_joint), dtype=np.float32)

        # Obs Space breakdown:
        # 1. Joint Data (obs_dim)
        # 2. Root Pos (3) + Root Orn (4) + Root AngVel (3) = 10
        # 3. Assist Factors (Kp, Kd) = 2 -- Nuked to 0
        # Total Extras = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim + 10 + sum(self.dof_per_joint),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.steps_count = 0
        self.current_energy_cost = 0.0
        self.last_action = np.zeros(sum(self.dof_per_joint), dtype=np.float32)

        # Randomize friction slightly to improve robustness
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)
        # p.changeDynamics(self.plane_id, -1, lateralFriction=random.uniform(0.5, 1.2))

        resetJointMotorsAndState(self.humanoid_id)

        for _ in range(256):
            p.stepSimulation()
        return self._get_obs(), {}

    def step(self, action):
        printOn = True
        printStep = self.steps_count % (self.max_steps / 2) == 0
        if printStep and printOn:
            print(
                f"\n--- EPISODE {self.episode_count} | STEP {self.steps_count} | FORCE DIAGNOSTICS ---"
            )

        torque_scale = 0.25
        # --- 2. PRE-CALCULATE TORQUES ---
        # We calculate the target torques ONCE per policy step
        # but apply them multiple times in the physics loop.
        prepared_torques = []
        action_idx = 0

        if printStep and printOn:
            print("LAST ACTION:", self.last_action)

        for j in self.joint_indices:
            name = p.getJointInfo(self.humanoid_id, j)[1].decode("utf-8")
            if name not in MAX_TORQUE_MAP:
                continue
            max_f = np.array(MAX_TORQUE_MAP[name]) * torque_scale

            if self.dof_per_joint[j] == 1:
                act_val = action[action_idx]
                torque = act_val * max_f
                if printStep and printOn:
                    effort_pct = abs(act_val) * 100
                    print(
                        f"Joint: {name:<15} | Action: {act_val:>18.2f}   | Effort: {effort_pct:>5.1f}% | Torque: {torque:>22.1f} Nm"
                    )
                prepared_torques.append((j, 1, torque))
                action_idx += 1
            elif self.dof_per_joint[j] == 3:
                raw_actions = action[action_idx : action_idx + 3]
                torques = raw_actions * max_f
                if printStep and printOn:
                    effort_pct = np.linalg.norm(raw_actions) / math.sqrt(3) * 100
                    act_str = f"[{raw_actions[0]:5.2f}, {raw_actions[1]:5.2f}, {raw_actions[2]:5.2f}]"
                    trq_str = (
                        f"[{torques[0]:6.1f}, {torques[1]:6.1f}, {torques[2]:6.1f}]"
                    )
                    print(
                        f"Joint: {name:<15} | Action: {act_str:>21} | Effort: {effort_pct:>5.1f}% | Torque: {trq_str:>25} Nm"
                    )
                prepared_torques.append((j, 3, list(torques)))
                action_idx += 3

        video_buffer = []
        for tick in range(160):
            # Apply Motor Torques at every simulation tick (240Hz)
            for j_idx, dof, val in prepared_torques:
                if dof == 1:
                    p.setJointMotorControl2(
                        self.humanoid_id, j_idx, p.TORQUE_CONTROL, force=val
                    )
                else:
                    p.setJointMotorControlMultiDof(
                        self.humanoid_id, j_idx, p.TORQUE_CONTROL, force=val
                    )

            p.stepSimulation()

            if tick % 10 == 0:
                self.camera.update()
                img = self.camera.get_last_image()
                video_buffer.append(img)

        video_buffer = np.asarray(video_buffer)
        # print("shape of image", np.asarray(img).shape)
        # print("Shape of video", video_buffer.shape)
        reward, done, decomposition = self._get_reward(action, video_buffer)
        obs = self._get_obs()

        self.steps_count += 1
        truncated = self.steps_count >= self.max_steps
        info = {"decomposition": decomposition}

        return obs, reward, done, truncated, info

    def _get_reward(self, action, video_buffer):
        done = False

        # Check if robot too far away...
        # 1. Get the robot's current physical state
        base_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        distance_from_origin = np.linalg.norm(
            np.array(base_pos[:2])
        )  # Euclidean distance (X, Y)

        # 2. Hard Termination if the robot escapes the 1m radius
        if distance_from_origin > 1.0:
            done = True
            print(
                f"⚠️ Episode Terminated: Robot drifted {distance_from_origin:.2f}m from origin."
            )
        # D. ACTION PENALTY (New)
        action_diff = action - self.last_action
        action_rate_cost = np.sum(np.square(action_diff)) * -0.1
        self.last_action = action.copy()

        # 4. V-JEPA REWARD
        # inputs = processor(video_buffer, return_tensors="pt")
        # with th.no_grad():
        #     current_latent = vjepa_model(**inputs).last_hidden_state
        current_latent = my_jepa_model.get_latent(video_buffer, verbose=JEPA_VERBOSE)

        # Perceptual Distance (Reward is negative MSE)
        mse_dist = F.mse_loss(TARGET_LATENT, current_latent).item()
        mse_dist = my_jepa_model.compute_mse(current_latent, TARGET_LATENT)

        vjepa_reward = -mse_dist

        total_reward = vjepa_reward  # + action_rate_cost

        # if self.steps_count == self.max_steps:  # and self.episode_count % 2 == 0:
        #     save_labeled_video(
        #         video_buffer, mse_dist, self.episode_count, folder=self.video_dir
        #     )

        assert Target_Video is not None
        save_labeled_video(
            video_buffer,
            Target_Video,
            mse_dist,
            self.episode_count,
            self.steps_count,
            folder=self.video_dir,
        )

        decomp = {
            "01_vjepa_reward": vjepa_reward,
            # "02_action_rate_penalty": action_rate_cost,
            "z_TOTAL": total_reward,
        }

        return total_reward, done, decomp  # done is false...

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

        # Merge Data
        final_obs = (
            joint_obs + list(pos) + list(orn) + list(ang_vel) + list(self.last_action)
        )

        # --- DIAGNOSTIC LOGGER  ---
        if self.steps_count % (self.max_steps / 2) == 0:
            # Add remaining labels
            debug_labels.extend(["Root_X", "Root_Y", "Root_Z"])
            debug_labels.extend(["Root_Qx", "Root_Qy", "Root_Qz", "Root_Qw"])
            debug_labels.extend(["Root_Wx", "Root_Wy", "Root_Wz"])

            # Add labels for Last Action
            # (We use indices since 'joint names' map awkwardly to raw action indices)
            for k in range(len(self.last_action)):
                debug_labels.append(f"LastAct_{k}")

            print(f"\n--- OBS DEBUG (Step {self.steps_count}) ---")
            print(f"{'INDEX':<6} | {'LABEL':<25} | {'VALUE':<10}")
            print("-" * 45)
            for i, (label, val) in enumerate(zip(debug_labels, final_obs)):
                # Highlight Root Z and Kp visually
                marker = (
                    " <---" if label in ["Root_Z", "Assist_Kp", "LastAct_0"] else ""
                )
                print(f"{i:<6} | {label:<25} | {val:>10.4f}{marker}")
            print("-" * 45 + "\n")

        return np.array(final_obs, dtype=np.float32)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    with utils.PyBulletSim(gui=True, disableRender=True) as client:
        humanoid_id, plane_id = utils.setup_humanoid_scene(p)

        TOTAL_TIMESTEPS = 90
        RUN_NAME = "V1_Run3_TEST"
        VIDEO_DIR = "videos/" + RUN_NAME
        env = HumanStandEnv(humanoid_id, plane_id, VIDEO_DIR)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=88.8)

        utils.print_joint_info(humanoid_id)
        utils.print_dynamics_info(humanoid_id)
        utils.print_link_states(humanoid_id)

        # MODEL CONFIGURATION
        # Define the policy architecture
        policy_kwargs = dict(
            activation_fn=th.nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            log_std_init=-2.0,
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            use_sde=True,  # <--- Stops the flailing
            sde_sample_freq=4,  # smooths noise every 4 steps
            verbose=1,
            learning_rate=linear_schedule(5.0e-5, min_value=0),
            # learning_rate=5.0e-5,
            n_steps=9,  # buffer of training data
            batch_size=9,  # Batch size passed at once to NN
            n_epochs=2,  # number of times entire buffer passed to NN
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=1.0,
            max_grad_norm=0.5,
            tensorboard_log="./logs/",
        )
        print(model.policy)
        print("--- Starting Training with Gated Velocity & Energy Penalty ---")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=RewardLoggerCallback(),
            tb_log_name=RUN_NAME,
        )

        model.save("jepa_humanoid_v1_final")
        env.save("jepa_humanoid_vecnormalize.pkl")
