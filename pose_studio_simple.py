import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
from utils import PyBulletCamera, SimConfig, enable_headless_opengl


# --- 1. SETUP & SAFETY ---
def safe_read(parameter_id, default=0.0):
    """Prevents 'Failed to read parameter' errors from crashing the UI."""
    try:
        return p.readUserDebugParameter(parameter_id)
    except:
        return default


os.makedirs("poses", exist_ok=True)

if p.isConnected():
    p.disconnect()
physicsClient = p.connect(p.GUI)
enable_headless_opengl(physicsClient)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

DT = 1.0 / 240.0
p.setTimeStep(DT)

# MODIFICATION 1: Start with ZERO gravity
p.setGravity(0, 0, 0)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

# --- STATIC POSE CONFIGURATION ---
POSE_CONFIG_PLANK = {
    "chest": (0, 0, 0),
    "neck": (0, 0, 0),
    "right_shoulder": (0, 0, 80),
    "right_elbow": 50,
    "right_wrist": (0, 0, 0),
    "left_shoulder": (0, 0, 80),
    "left_elbow": 50,
    "left_wrist": (0, 0, 0),
    "right_hip": (0, 20, 80),
    "right_knee": -80,
    "right_ankle": (0, 0, 0),
    "left_hip": (0, -20, 80),
    "left_knee": -80,
    "left_ankle": (0, 0, 0),
}

POSE_CONFIG_DEFAULT = {
    "chest": (0, 0, 0),
    "neck": (0, 0, 0),
    "right_shoulder": (0, 0, 0),
    "right_elbow": 0,
    "right_wrist": (0, 0, 0),
    "left_shoulder": (0, 0, 0),
    "left_elbow": 0,
    "left_wrist": (0, 0, 0),
    "right_hip": (0, 0, 0),
    "right_knee": 0,
    "right_ankle": (0, 0, 0),
    "left_hip": (0, 0, 0),
    "left_knee": 0,
    "left_ankle": (0, 0, 0),
}

POSE_CONFIG_KNEETUCK = {
    "chest": (0, 0, 0),
    "neck": (0, 0, 0),
    "right_shoulder": (0, 0, 0),
    "right_elbow": 0,
    "right_wrist": (0, 0, 0),
    "left_shoulder": (0, 0, 0),
    "left_elbow": 0,
    "left_wrist": (0, 0, 0),
    "right_hip": (0, 30, 150),
    "right_knee": -150,
    "right_ankle": (0, 0, 0),
    "left_hip": (0, -30, 150),
    "left_knee": -150,
    "left_ankle": (0, 0, 0),
}

# Athletic low stand
POSE_CONFIG = {
    "chest": (0, 0, 0),
    "neck": (0, 0, 0),
    "right_shoulder": (-20, 0, 30),
    "right_elbow": 0,
    "right_wrist": (0, 0, 0),
    "left_shoulder": (20, 0, 30),
    "left_elbow": 0,
    "left_wrist": (0, 0, 0),
    "right_hip": (-20, 0, 60),
    "right_knee": -60,
    "right_ankle": (0, 0, 20),
    "left_hip": (20, 0, 60),
    "left_knee": -60,
    "left_ankle": (0, 0, 20),
}

# --- 2. LOAD ASSETS ---
p.loadURDF("plane.urdf")
standing_ori = [math.pi / 2, 0, math.pi / 2]
# start_quat = p.getQuaternionFromEuler(SimConfig.START_ORI)
start_quat = p.getQuaternionFromEuler(standing_ori)
flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

# MODIFICATION 2: Spawn higher in the air (0.8 instead of 0.16) so legs can swing
robot_id = p.loadURDF(
    "humanoid/humanoid.urdf",
    [0, 0, 1.8],
    start_quat,
    globalScaling=SimConfig.GLOBAL_SCALE,
    flags=flags,
)
camera = PyBulletCamera(width=256, height=256)

# --- 3. MAP JOINTS ---
num_joints = p.getNumJoints(robot_id)
joint_info_map = {}

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    joint_info_map[i] = {"name": joint_name, "type": joint_type}

snap_btn = p.addUserDebugParameter("📸 Snap Image", 1, 0, 0)

# --- 4. MAIN LOOP ---
step_counter = 0
CAMERA_FREQ = 8
prev_btn_val = 0
pose_counter = 1

print(f"🎮 POSE STUDIO ACTIVE @ {int(1 / DT)}Hz")

while True:
    p.stepSimulation()
    step_counter += 1

    # MODIFICATION 3: Engage gravity after 0.5 seconds (120 steps)
    if step_counter == 240:
        p.setGravity(0, 0, -9.8)
        print("🌍 Assembly complete. Gravity engaged!")

    if step_counter % CAMERA_FREQ == 0:
        camera.update(client_id=physicsClient)

    # Process Joints
    for joint_index, info in joint_info_map.items():
        name = info["name"]

        if name in POSE_CONFIG:
            val = POSE_CONFIG[name]
            joint_type = info["type"]

            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                v = val[0] if isinstance(val, (list, tuple)) else val
                target_rad = math.radians(v)
                p.setJointMotorControl2(
                    robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_rad
                )

            elif joint_type == p.JOINT_SPHERICAL:
                if not isinstance(val, (list, tuple)):
                    val = (val, 0, 0)
                r_rad = math.radians(val[0])
                p_rad = math.radians(val[1])
                y_rad = math.radians(val[2])

                quat = p.getQuaternionFromEuler([r_rad, p_rad, y_rad])
                p.setJointMotorControlMultiDof(
                    robot_id, joint_index, p.POSITION_CONTROL, targetPosition=quat
                )

    # Snap Logic
    btn_val = safe_read(snap_btn)
    if btn_val > prev_btn_val:
        prev_btn_val = btn_val
        img = camera.get_last_image()
        if img:
            print("SHAPE:", np.asarray(img).shape)
            path = f"poses/pose_{pose_counter}.jpg"
            img.save(path)
            print(f"✅ Saved: {path}")
            pose_counter += 1

    # time.sleep(DT)
