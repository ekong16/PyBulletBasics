import pybullet as p
import time
import pybullet_data
import numpy as np
from tabulate import tabulate

# --- Safe cleanup ---
if p.isConnected():
    p.disconnect()  # only disconnect if already connected

# --- Utility Functions ---
def print_dynamics_info(body_id):
    """
    Prints mass, local inertia, friction, damping, etc. for each link (and base).
    """
    rows = []

    # Base (link index = -1)
    base_dyn = p.getDynamicsInfo(body_id, -1)
    rows.append([
        -1,
        "base",
        base_dyn[0],                  # mass
        np.round(base_dyn[2], 3),     # local inertia pos
        np.round(base_dyn[3], 3),     # local inertia orn
        base_dyn[1],                  # lateral friction
        np.round(base_dyn[4], 3),                  # restitution
        base_dyn[5],                  # rolling friction
        base_dyn[6],                  # spinning friction
        base_dyn[7],                  # contact damping
        base_dyn[8]                   # contact stiffness
    ])

    # Links
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        name = info[12].decode("utf-8")
        dyn = p.getDynamicsInfo(body_id, i)
        rows.append([
            i,
            name,
            dyn[0],
            np.round(dyn[2], 3),
            np.round(dyn[3], 3),
            dyn[1],
            np.round(dyn[4], 3),
            dyn[5],
            dyn[6],
            dyn[7],
            dyn[8]
        ])

    headers = [
        "ID", "Link", "Mass", "Inertia_Pos", "Inertia_Orn",
        "Friction", "Restitution", "Roll_Fric", "Spin_Fric",
        "Contact_Damp", "Contact_Stiff"
    ]

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", floatfmt=".3f"))

# --- Simulation Initialization ---
physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])

r2d2_id = p.loadURDF("r2d2.urdf",startPos, startOrientation)
# --- Inspect joints to find head link ---
print_dynamics_info(r2d2_id)
print("\nR2D2 joints:")
for i in range(p.getNumJoints(r2d2_id)):
    joint_info = p.getJointInfo(r2d2_id, i)
    link_state = p.getLinkState(r2d2_id, i)

    pos = np.round(link_state[0], 2)
    orn = np.round(link_state[1], 2)
    local_inertia_pos = np.round(link_state[2], 2)
    local_inertia_orn = np.round(link_state[3], 2)
    mass = joint_info[10]
    name = joint_info[12].decode('utf-8')
    print(f"{i:2d} | {name:18s} | mass={mass:7.3f} | pos={pos} | inertia={local_inertia_pos}")

    #print(f"id={i}, name={joint_info[12].decode('utf-8')}, LinkXYZ={np.round(link_state[0],2)}")

HEAD_LINK = 14  # Change this if your printout differs

# --- Camera parameters and Get Camera View Func ---
width, height = 160, 120
fov = 60
aspect = width / height
near, far = 0.01, 100

def get_head_camera_view(robot_id, link_index):
    init_fwd_vec = [0, 1, 0]
    up_vec = [0, 0, 1]
    # Camera offset in the head's local frame
    cam_offset = [0.0, 0.05, 0.1]  # Slightly above and forward

    ret = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    link_pos, link_orn = ret[0], ret[1]

    orn_mat = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)

    # Camera in world coordinates
    cam_pos = np.array(link_pos) +  orn_mat @ cam_offset
    cam_target = cam_pos + orn_mat @ init_fwd_vec    
    cam_up = orn_mat @ up_vec

   
    view = p.computeViewMatrix(cam_pos, cam_target, cam_up)
    proj = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    return view, proj

# --- Run Simulation ---
move_speed = 0.05  #forward/backward step
turn_speed = 0.05  #radians per step 
frame=0
for i in range (10000):
    frame += 1

    keys = p.getKeyboardEvents()
    pos, orn = p.getBasePositionAndOrientation(r2d2_id)
    pos_x, pos_y, pos_z = pos

    if (frame % 10 == 0):
        print(f"robotId: {r2d2_id}, robotBaseXYZ={np.round(pos,2)} robotBaseOrnQuat: {np.round(orn, 2)}") 

    # Turning (J/L)
    moved = False
    if ord('j') in keys:  # turn left
        turn_quat = p.getQuaternionFromEuler([0, 0, turn_speed])
        orn = p.multiplyTransforms([0,0,0], turn_quat, [0,0,0], orn)[1]
        moved = True

    if ord('l') in keys:  # turn right
        turn_quat = p.getQuaternionFromEuler([0, 0, -turn_speed])
        orn = p.multiplyTransforms([0,0,0], turn_quat, [0,0,0], orn)[1]
        moved = True

    # Forward vector from quaternion (local Y is forward)
    # this is from the qauternion to euler matrix
    forward_x = 2*(orn[0]*orn[1] - orn[3]*orn[2])
    forward_y = 1 - 2*(orn[0]**2 + orn[2]**2)

    # Foward and backward movement E/D
    if ord('e') in keys:  # forward
        pos_x += move_speed * forward_x
        pos_y += move_speed * forward_y
        moved = True

    if ord('d') in keys:  # backward
        pos_x -= move_speed * forward_x
        pos_y -= move_speed * forward_y
        moved = True

    # Update position and orientation
    if moved:
        p.resetBasePositionAndOrientation(r2d2_id, [pos_x, pos_y, pos_z], orn)
    
    if frame % 3 == 0:  # every 10th frame → ~24 FPS equivalent
        view, proj = get_head_camera_view(r2d2_id, HEAD_LINK)
        _, _, rgb, depth, seg = p.getCameraImage(width, height, view, proj)
        rgb_array = np.reshape(rgb, (height, width, 4))
        #print(f"Head camera frame: mean RGB={rgb_array[..., :3].mean():.2f}")

    #STEP
    p.stepSimulation()
    time.sleep(1./240.)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(r2d2_id)
print(cubePos,cubeOrn)
p.disconnect()