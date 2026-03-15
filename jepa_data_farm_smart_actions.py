import os
import math
import random
import numpy as np
import torch
import cv2
import pybullet as p
import utils  # Your custom utils

# ==========================================
# GLOBAL VARS & CONFIG
# ==========================================
# 128kg Power-of-Two Physical Constraints
MAX_TORQUE_MAP = {
    "chest": [500, 500, 500],
    "neck": [200, 200, 200],
    "right_shoulder": [200, 200, 200],
    "left_shoulder": [200, 200, 200],
    "right_elbow": 100,
    "left_elbow": 100,
    "right_hip": [800, 800, 800],
    "left_hip": [800, 800, 800],
    "right_knee": 800,
    "left_knee": 800,
    "right_ankle": [400, 400, 400],
    "left_ankle": [400, 400, 400],
}

# Data Collection Parameters
START_EPISODE = 1000  # Start at 1000 to append to existing data
NUM_EPISODES = 1000  # Generate 1000 new smart episodes
CHAIN_LENGTH = 10
DATASET_DIR = "world_model_dataset"


# ==========================================
# HELPER FUNCTIONS (Physics & Storage)
# ==========================================
def get_smart_action(dim):
    """
    Generates sparse, normally distributed actions to teach the world model
    both fine motor skills and passive gravity (Zero State).
    """
    # 1. Roll for 'Whole-Body Rest' (10% chance)
    if random.random() < 0.10:
        return np.zeros(dim, dtype=np.float32)

    # 2. Sample from Normal Distribution (centered at 0, std 0.5)
    action = np.random.normal(loc=0.0, scale=0.5, size=dim).astype(np.float32)

    # 3. Inject 'Joint Sparsity' (Randomly zero out 30-70% of joints)
    sparsity_factor = random.uniform(0.3, 0.7)
    mask = (np.random.rand(dim) > sparsity_factor).astype(np.float32)
    action *= mask

    # 4. Final Clamp to stay within the simulator's torque limits [-1, 1]
    return np.clip(action, -1.0, 1.0)


def resetJointMotorsAndState(humanoid_id, target_pos, target_orn):
    """
    Resets the robot to a specific position/orientation, clears all residual
    velocities from the last episode, and reapplies custom damping and friction.
    """
    p.resetBasePositionAndOrientation(humanoid_id, target_pos, target_orn)

    for j in range(p.getNumJoints(humanoid_id)):
        info = p.getJointInfo(humanoid_id, j)
        jt = info[2]

        # --- APPLY DAMPING ---
        p.changeDynamics(
            humanoid_id,
            j,
            jointDamping=0.5,
            angularDamping=0.1,
        )

        # --- CLEAR RESIDUAL MOTORS & VELOCITIES ---
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

    # --- CUSTOM FRICTION & STIFFNESS ---
    for link in [5, 8]:
        p.changeDynamics(humanoid_id, link, lateralFriction=5.0, rollingFriction=1.0)

    for link in [11, 14]:
        p.changeDynamics(
            humanoid_id,
            link,
            lateralFriction=10.0,
            contactStiffness=30000,
            contactDamping=1000,
        )


def save_debug_mp4(video_0, action, video_1, episode, step, folder):
    """Saves a side-by-side MP4 of the Before and After states for visual auditing."""
    os.makedirs(folder, exist_ok=True)
    # Renamed to perfectly match the .pt file in the same folder
    filepath = os.path.join(folder, f"transition_{step:03d}_debug.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    # Dimensions: 256 (Left) + 4 (Border) + 256 (Right) = 516 Width
    out = cv2.VideoWriter(filepath, fourcc, 8.0, (516, 256))

    # --- ACTION SUMMARY ---
    # Summarize the 28-dim array so it doesn't clutter the screen
    act_norm = np.linalg.norm(action)
    act_max = np.max(np.abs(action))
    action_text = f"Action Norm: {act_norm:.2f} | Max Action: {act_max:.2f} | FPS: 8.0"

    # --- TEXT RENDERER ---
    def draw_text(img, text, pos, color=(255, 255, 255), scale=0.45):
        """Draws high-contrast text with a thick black outline."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 1. Thick Black Outline
        cv2.putText(img, text, pos, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        # 2. Bright Inner Text
        cv2.putText(img, text, pos, font, scale, color, 1, cv2.LINE_AA)

    for i in range(16):
        # video_0 and video_1 are already in (H, W, C) format natively now!
        frame_l = video_0[i]
        frame_r = video_1[i]

        bgr_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
        bgr_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

        spacer = np.full((256, 4, 3), 50, dtype=np.uint8)
        canvas = np.hstack((bgr_l, spacer, bgr_r))

        # --- APPLY HUD TEXT ---
        # Top Left: State 0
        draw_text(canvas, "State 0 (Before)", (10, 22), color=(255, 255, 255))
        # Bottom Left: The Action Summary (in a bright green for visibility)
        draw_text(canvas, action_text, (10, 245), color=(100, 255, 100))

        # Top Right: State 1 & Step Counter
        draw_text(
            canvas, f"State 1 (After) | Step: {step}", (266, 22), color=(0, 215, 255)
        )
        # Bottom Right: Episode Counter
        draw_text(canvas, f"Episode: {episode:03d}", (266, 245), color=(255, 255, 255))

        out.write(canvas)

    out.release()


class DataCollector:
    def __init__(self, humanoid_id):
        self.humanoid_id = humanoid_id
        self.camera = utils.PyBulletCamera()

        # 1. Map Joint Action Space
        n_joints = p.getNumJoints(self.humanoid_id)
        self.joint_indices = []
        self.dof_per_joint = []
        for j in range(n_joints):
            self.joint_indices.append(j)
            jt = p.getJointInfo(self.humanoid_id, j)[2]
            if jt == p.JOINT_SPHERICAL:
                self.dof_per_joint.append(3)
            elif jt == p.JOINT_REVOLUTE:
                self.dof_per_joint.append(1)
            else:
                self.dof_per_joint.append(0)

        self.action_dim = sum(self.dof_per_joint)

    def execute_and_record(self, action_array):
        """Applies torques for 160 steps, snaps 16 frames, returns formatted Tensor."""
        torque_scale = 0.25
        prepared_torques = []
        action_idx = 0

        # Calculate torques once
        for j in self.joint_indices:
            name = p.getJointInfo(self.humanoid_id, j)[1].decode("utf-8")
            if name not in MAX_TORQUE_MAP:
                continue
            max_f = np.array(MAX_TORQUE_MAP[name]) * torque_scale

            if self.dof_per_joint[j] == 1:
                torque = action_array[action_idx] * max_f
                prepared_torques.append((j, 1, torque))
                action_idx += 1
            elif self.dof_per_joint[j] == 3:
                torques = action_array[action_idx : action_idx + 3] * max_f
                prepared_torques.append((j, 3, list(torques)))
                action_idx += 3

        video_buffer = []

        # The 160-Step Physics Loop
        for tick in range(160):
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

            # The Camera Snapping Cadence
            if tick % 10 == 0:
                self.camera.update()
                img = self.camera.get_last_image()  # Returns uint8 (H, W, 3)
                video_buffer.append(img)

        # Raw (Time, Height, Width, Channels) - EXACTLY what your JEPA wrapper wants
        video_numpy = np.asarray(video_buffer, dtype=np.uint8)
        return video_numpy


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)

    with utils.PyBulletSim(gui=True, disableRender=True) as client:
        humanoid_id, plane_id = utils.setup_humanoid_scene(p)
        collector = DataCollector(humanoid_id)

        print("🚀 Starting LeCun World Model Data Farm...")

        for ep in range(START_EPISODE, START_EPISODE + NUM_EPISODES):
            ep_dir = os.path.join(DATASET_DIR, f"episode_{ep:05d}")
            os.makedirs(ep_dir, exist_ok=True)

            # --- THE SKYDIVING INITIALIZATION ---
            # 1. Random Orientation
            rand_roll = random.uniform(0, 3.14)
            rand_pitch = random.uniform(0, 3.14)
            rand_yaw = random.uniform(0, 3.14)
            rand_quat = p.getQuaternionFromEuler([rand_roll, rand_pitch, rand_yaw])

            # 2. Spawn 1.5m in the air
            resetJointMotorsAndState(
                humanoid_id, target_pos=[0, 0, 1.5], target_orn=rand_quat
            )

            # 3. Random Settle Window (Gravity takes over)
            settle_steps = random.randint(20, 150)
            for _ in range(settle_steps):
                p.stepSimulation()

            # --- THE BASELINE STATE (Video 0) ---
            # Record the momentum of the crash by applying ZERO torque for 160 steps
            initial_action = np.random.uniform(-1, 1, size=collector.action_dim).astype(
                np.float32
            )
            current_state_video = collector.execute_and_record(initial_action)

            # --- THE CONTINUOUS CHAIN ---
            for step in range(CHAIN_LENGTH):
                # ---> THE 1.2m LEASH <---
                base_pos, _ = p.getBasePositionAndOrientation(humanoid_id)
                # Calculate Euclidean distance on the X-Y plane
                drift_dist = math.hypot(base_pos[0], base_pos[1])

                if drift_dist > 1.2:
                    print(
                        f"⚠️ Robot drifted {drift_dist:.2f}m. Terminating chain {ep:05d} early."
                    )
                    break  # Exits the CHAIN_LENGTH loop, moves to next episode

                # 1. Generate Random Exploration Action
                # Using uniform [-1, 1] - PyBullet damping handles the rest
                random_action = get_smart_action(collector.action_dim)

                # 2. Apply and Record Next State
                next_state_video = collector.execute_and_record(random_action)

                # 3. Construct the .pt Dictionary
                transition_data = {
                    "state_0": current_state_video,  # uint8, shape: (16, 256, 256, 3)
                    "action": random_action,  # float32, shape: (Action_Dim,)
                    "state_1": next_state_video,  # uint8, shape: (16, 256, 256, 3)
                }

                # 4. Save to Disk
                save_path = os.path.join(ep_dir, f"transition_{step:03d}.pt")
                torch.save(transition_data, save_path)

                # 5. Sanity Check Video
                if ep % 10 == 0:
                    save_debug_mp4(
                        current_state_video,
                        random_action,
                        next_state_video,
                        ep,
                        step,
                        folder=ep_dir,
                    )

                # 6. Shift the chain
                current_state_video = next_state_video

            print(f"✅ Episode {ep:05d} complete. Saved {step} transitions.")
