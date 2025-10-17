import pybullet as p
import time
import pybullet_data
import numpy as np

# --- Safe cleanup ---
if p.isConnected():
    p.disconnect()  # only disconnect if already connected

physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])

r2d2_id = p.loadURDF("r2d2.urdf",startPos, startOrientation)
# --- Inspect joints to find head link ---
print("\nR2D2 joints:")
for i in range(p.getNumJoints(r2d2_id)):
    info = p.getJointInfo(r2d2_id, i)
    print(f"  id={i}, name={info[12].decode('utf-8')}")

HEAD_LINK = 14  # Change this if your printout differs


 # --- Camera parameters ---
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


move_speed = 0.05  #forward/backward step
turn_speed = 0.05  #radians per step 
frame=0
for i in range (10000):
    frame += 1

    keys = p.getKeyboardEvents()
    pos, orn = p.getBasePositionAndOrientation(r2d2_id)
    pos_x, pos_y, pos_z = pos 

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
        print(f"Head camera frame: mean RGB={rgb_array[..., :3].mean():.2f}")

    #STEP
    p.stepSimulation()
    time.sleep(1./240.)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(r2d2_id)
print(cubePos,cubeOrn)
p.disconnect()