import pybullet as p
import time
import pybullet_data
import numpy as np
from tabulate import tabulate
import random


class PyBulletSim:
    def __init__(self, gui=True):
        self.gui = gui

    def __enter__(self):
        # Disconnect any leftover session
        if p.isConnected():
            p.disconnect()
        # Start a new connection
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up
        print("DISCONNECTING CLEANLY...")
        if p.isConnected():
            p.removeAllUserDebugItems()  # optional, clears debug lines
            p.disconnect()
        # Let exceptions propagate (don’t suppress them)
        return False


# --- Utility Functions ---
def spawn_random_object():
    # Choose a random shape type: box, sphere, cylinder
    shape_type = random.choice(["box", "sphere", "cylinder"])

    # Random size
    if shape_type == "box":
        half_extents = [random.uniform(0.05, 0.2) for _ in range(3)]
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[random.random(), random.random(), random.random(), 1],
        )
    elif shape_type == "sphere":
        radius = random.uniform(0.05, 0.2)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[random.random(), random.random(), random.random(), 1],
        )
    else:  # cylinder
        radius = random.uniform(0.05, 0.15)
        height = random.uniform(0.05, 0.3)
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[random.random(), random.random(), random.random(), 1],
        )

    # Random position on the floor
    pos = [random.uniform(-10, 10), random.uniform(-3, 3), 0.2]
    orientation = p.getQuaternionFromEuler([0, 0, random.uniform(0, 3.14)])

    # Mass > 0 to make it dynamic
    mass = random.uniform(0.5, 2)

    # Create the object
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos,
        baseOrientation=orientation,
    )

    return body_id


def print_dynamics_info(body_id):
    """
    Prints mass, local inertia, friction, damping, etc. for each link (and base).
    """
    rows = []

    # Base (link index = -1)
    base_dyn = p.getDynamicsInfo(body_id, -1)
    rows.append(
        [
            -1,
            "base",
            base_dyn[0],  # mass
            np.round(base_dyn[2], 3),  # local inertia pos
            np.round(base_dyn[3], 3),  # local inertia orn
            base_dyn[1],  # lateral friction
            np.round(base_dyn[4], 3),  # restitution
            base_dyn[5],  # rolling friction
            base_dyn[6],  # spinning friction
            base_dyn[7],  # contact damping
            base_dyn[8],  # contact stiffness
        ]
    )

    # Links
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        name = info[12].decode("utf-8")
        dyn = p.getDynamicsInfo(body_id, i)
        rows.append(
            [
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
                dyn[8],
            ]
        )

    headers = [
        "ID",
        "Link",
        "Mass",
        "Inertia_Pos",
        "Inertia_Orn",
        "Friction",
        "Restitution",
        "Roll_Fric",
        "Spin_Fric",
        "Contact_Damp",
        "Contact_Stiff",
    ]

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", floatfmt=".3f"))


with PyBulletSim(gui=True) as client:
    # --- Simulation Initialization ---
    # physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)

    planeId = p.loadURDF("plane.urdf")

    startPos = [0, 0, 0.5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    r2d2_id = p.loadURDF("r2d2.urdf", startPos, startOrientation, useFixedBase=False)
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
        name = joint_info[12].decode("utf-8")
        print(
            f"{i:2d} | {name:18s} | mass={mass:7.3f} | pos={pos} | inertia={local_inertia_pos}"
        )

        # print(f"id={i}, name={joint_info[12].decode('utf-8')}, LinkXYZ={np.round(link_state[0],2)}")

    HEAD_BOX_LINK = 14  # Change this if your printout differs
    RIGHT_FRONT_WHEEL = 2
    RIGHT_BACK_WHEEL = 3
    LEFT_FRONT_WHEEL = 6
    LEFT_BACK_WHEEL = 7

    for _ in range(5):
        spawn_random_object()

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
        cam_pos = np.array(link_pos) + orn_mat @ cam_offset
        cam_target = cam_pos + orn_mat @ init_fwd_vec
        cam_up = orn_mat @ up_vec

        view = p.computeViewMatrix(cam_pos, cam_target, cam_up)
        proj = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        return view, proj

    # --- Run Simulation ---
    move_force_newtons = 1000.0  # forward/backward step
    turn_speed = 0.05  # radians per step
    frame = 0
    wheel_current_velocities = [0, 0, 0, 0]
    wheel_target_velocities = [0, 0, 0, 0]  # Radians / S
    for i in range(10000):
        frame += 1

        wheel_target_velocities = [0, 0, 0, 0]

        keys = p.getKeyboardEvents()
        pos, orn = p.getBasePositionAndOrientation(r2d2_id)
        pos_x, pos_y, pos_z = pos

        if frame % 50 == 0:
            print(
                f"robotId: {r2d2_id}, robotBaseXYZ={np.round(pos, 2)} robotBaseOrnQuat: {np.round(orn, 2)}"
            )

        # Turning (J/L)
        moved = False
        if ord("j") in keys:  # turn left
            turn_quat = p.getQuaternionFromEuler([0, 0, turn_speed])
            orn = p.multiplyTransforms([0, 0, 0], turn_quat, [0, 0, 0], orn)[1]
            moved = True

        if ord("l") in keys:  # turn right
            turn_quat = p.getQuaternionFromEuler([0, 0, -turn_speed])
            orn = p.multiplyTransforms([0, 0, 0], turn_quat, [0, 0, 0], orn)[1]
            moved = True

        # Forward vector from quaternion (local Y is forward)
        # this is from the qauternion to euler matrix
        orn_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # forward_x = 2*(orn[0]*orn[1] - orn[3]*orn[2])
        # forward_y = 1 - 2*(orn[0]**2 + orn[2]**2)

        # Foward and backward movement E/D
        if ord("e") in keys and keys[ord("e")] & p.KEY_IS_DOWN:  # forward
            wheel_target_velocities = [-30, -30, -30, -30]
            forces = [8, 8, 8, 8]
            # print("FORCE", force, orn_mat[:, 1], p.LINK_FRAME)
            # p.applyExternalForce(
            #     r2d2_id, -1, forceObj=force, posObj=pos, flags=p.WORLD_FRAME
            # )
            # pos_x += move_speed * forward_x
            # pos_y += move_speed * forward_y
            moved = True

        elif ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:  # backward
            wheel_target_velocities = [20, 20, 20, 20]
            forces = [8, 8, 8, 8]
            moved = True
        else:
            wheel_target_velocities = [0, 0, 0, 0]
            forces = [2, 2, 2, 2]
        #     pos_x -= move_speed * forward_x
        #     pos_y -= move_speed * forward_y
        #     moved = True

        # Update position and orientation
        # print("Velocity", linear_velocity)
        # if (moved):
        #     p.applyExternalForce(r2d2_id, -1, forceObj=[F_x, F_y, F_z], posObj=[0,0,0], flags=p.WORLD_FRAME)
        # if moved:
        #     p.resetBasePositionAndOrientation(r2d2_id, [pos_x, pos_y, pos_z], orn)

        if wheel_target_velocities != wheel_current_velocities:
            p.setJointMotorControlArray(
                r2d2_id,
                [
                    RIGHT_FRONT_WHEEL,
                    RIGHT_BACK_WHEEL,
                    LEFT_FRONT_WHEEL,
                    LEFT_BACK_WHEEL,
                ],
                p.VELOCITY_CONTROL,
                targetVelocities=wheel_target_velocities,
                forces=forces,
            )
            wheel_current_velocities = wheel_target_velocities

        if frame % 3 == 0:  # every 10th frame → ~24 FPS equivalent
            view, proj = get_head_camera_view(r2d2_id, HEAD_BOX_LINK)
            _, _, rgb, depth, seg = p.getCameraImage(width, height, view, proj)
            rgb_array = np.reshape(rgb, (height, width, 4))
            # print(f"Head camera frame: mean RGB={rgb_array[..., :3].mean():.2f}")

        # STEP
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(r2d2_id)
    print(cubePos, cubeOrn)
    p.disconnect()
