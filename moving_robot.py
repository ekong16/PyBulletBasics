import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")

startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
r2d2_id = p.loadURDF("r2d2.urdf",startPos, startOrientation)

move_speed = 0.05  #forward/backward step
turn_speed = 0.05  #radians per step 

for i in range (10000):
    keys = p.getKeyboardEvents()
    pos, orn = p.getBasePositionAndOrientation(r2d2_id)
    pos_x, pos_y, pos_z = pos 

    # --- Turning (J/L) ---
    if ord('j') in keys:  # turn left
        turn_quat = p.getQuaternionFromEuler([0, 0, turn_speed])
        orn = p.multiplyTransforms([0,0,0], turn_quat, [0,0,0], orn)[1]
        p.resetBasePositionAndOrientation(r2d2_id, [pos_x, pos_y, pos_z], orn)


    if ord('l') in keys:  # turn right
        turn_quat = p.getQuaternionFromEuler([0, 0, -turn_speed])
        orn = p.multiplyTransforms([0,0,0], turn_quat, [0,0,0], orn)[1]
        p.resetBasePositionAndOrientation(r2d2_id, [pos_x, pos_y, pos_z], orn)

    
    # --- Forward vector from quaternion (local Y is forward) ---
    # this is from the qauternion to euler matrix
    forward_x = 2*(orn[0]*orn[1] - orn[3]*orn[2])
    forward_y = 1 - 2*(orn[0]**2 + orn[2]**2)

    # --- WASD movement ---
    if ord('e') in keys:  # forward
        pos_x += move_speed * forward_x
        pos_y += move_speed * forward_y
        p.resetBasePositionAndOrientation(r2d2_id, [pos_x, pos_y, pos_z], orn)

    if ord('d') in keys:  # backward
        pos_x -= move_speed * forward_x
        pos_y -= move_speed * forward_y
        p.resetBasePositionAndOrientation(r2d2_id, [pos_x, pos_y, pos_z], orn)

    
    # --- Update position and orientation ---
   
    p.stepSimulation()
    time.sleep(1./240.)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(r2d2_id)
print(cubePos,cubeOrn)
p.disconnect()