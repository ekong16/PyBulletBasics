import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from utils import PyBulletCamera, SimConfig


# --- 1. SETUP & SAFETY ---
def safe_read(parameter_id, default=0.0):
    """Prevents 'Failed to read parameter' errors from crashing the UI."""
    try:
        # We use the full 'p' name here; just ensure 'p' isn't overwritten in loops
        return p.readUserDebugParameter(parameter_id)
    except:
        return default


os.makedirs("poses", exist_ok=True)

# Connect and clean up previous sessions
if p.isConnected():
    p.disconnect()
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# EXPLICIT PHYSICS CONFIG
DT = 1.0 / 240.0
p.setTimeStep(DT)
# p.setGravity(0, 0, -9.8)

# PREVIEW CONFIG (Shadows + Color)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

# --- 2. LOAD ASSETS ---
p.loadURDF("plane.urdf")
# Load humanoid slightly in the air so it doesn't clip the floor
start_quat = p.getQuaternionFromEuler(SimConfig.START_ORI)
flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
robot_id = p.loadURDF(
    "humanoid/humanoid.urdf",
    [0, 0, 0.16],
    start_quat,
    globalScaling=SimConfig.GLOBAL_SCALE,
    flags=flags,
)
camera = PyBulletCamera(width=256, height=256)

# --- 3. DYNAMIC UI SLIDERS ---
num_joints = p.getNumJoints(robot_id)
joint_controls = {}

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    safe_name = f"J{i}_{joint_name}"

    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        lower, upper = info[8], info[9]
        if lower >= upper:
            lower, upper = -3.14, 3.14
        slider_id = p.addUserDebugParameter(safe_name, lower, upper, 0)
        joint_controls[i] = {"type": "1dof", "slider": slider_id}

    elif joint_type == p.JOINT_SPHERICAL:
        # 3-DOF joints need 3 sliders
        r_id = p.addUserDebugParameter(f"{safe_name}_R", -3.14, 3.14, 0)
        p_id = p.addUserDebugParameter(f"{safe_name}_P", -3.14, 3.14, 0)
        y_id = p.addUserDebugParameter(f"{safe_name}_Y", -3.14, 3.14, 0)
        joint_controls[i] = {"type": "3dof", "sliders": (r_id, p_id, y_id)}

snap_btn = p.addUserDebugParameter("📸 Snap Image", 1, 0, 0)

# --- 4. MAIN LOOP ---
step_counter = 0
CAMERA_FREQ = 8  # Update vision every 8 physics steps (~30 FPS)
prev_btn_val = 0
pose_counter = 1

print(f"🎮 POSE STUDIO ACTIVE @ {int(1 / DT)}Hz")

while True:
    p.stepSimulation()
    step_counter += 1

    # Update visual preview at a lower frequency for performance
    # if step_counter % CAMERA_FREQ == 0:
    camera.update(client_id=physicsClient)

    # Process Joints
    for joint_index, control in joint_controls.items():
        if control["type"] == "1dof":
            target = safe_read(control["slider"])
            p.setJointMotorControl2(
                robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target
            )

        elif control["type"] == "3dof":
            # Using specific _val names so we don't overwrite 'p' (the library)
            r_val = safe_read(control["sliders"][0])
            p_val = safe_read(control["sliders"][1])
            y_val = safe_read(control["sliders"][2])

            quat = p.getQuaternionFromEuler([r_val, p_val, y_val])
            p.setJointMotorControlMultiDof(
                robot_id, joint_index, p.POSITION_CONTROL, targetPosition=quat
            )

    # Snap Logic
    btn_val = safe_read(snap_btn)
    if btn_val > prev_btn_val:
        prev_btn_val = btn_val
        img = camera.get_last_image()
        if img:
            path = f"poses/pose_{pose_counter}.jpg"
            img.save(path)
            print(f"✅ Saved: {path}")
            pose_counter += 1

    # Sync loop speed to real-time (240 iterations per second)
    time.sleep(DT)
