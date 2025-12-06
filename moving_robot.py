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


def spawn_pickable_object(shapetype):
    half_height_box = 0.3
    half_extents_box = [0.15, 0.15, half_height_box]
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_box)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents_box,
        rgbaColor=[0.36, 0.25, 0.20, 1],  # Dark Brown
    )

    pos_box = [0, 1, half_height_box]
    box_mass = 10

    body_id_box = p.createMultiBody(
        baseMass=box_mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos_box,
        baseOrientation=[0, 0, 0],
    )

    if shapetype == "box":
        half_height_obj = 0.05
        pos_obj = pos_box.copy()
        pos_obj[2] += half_height_box + half_height_obj
        obj_mass = 1.5
        half_extents_obj = [0.03, 0.03, half_height_obj]
        collision_shape_obj = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents_obj
        )
        visual_shape_obj = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_obj,
            rgbaColor=[0.25, 0.8, 0.3, 1],  # Leaf green
        )
        body_id_box = p.createMultiBody(
            baseMass=obj_mass,
            baseCollisionShapeIndex=collision_shape_obj,
            baseVisualShapeIndex=visual_shape_obj,
            basePosition=pos_obj,
            baseOrientation=[0, 0, 0],
        )

    return body_id_box


# --- Utility Functions ---
def spawn_random_object():
    # Choose a random shape type: box, sphere, cylinder
    shape_type = random.choice(["box", "sphere", "cylinder"])

    # Random size
    if shape_type == "box":
        half_extents = [random.uniform(0.2, 0.3) for _ in range(3)]
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[random.random(), random.random(), random.random(), 1],
        )
    elif shape_type == "sphere":
        radius = random.uniform(0.2, 0.3)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[random.random(), random.random(), random.random(), 1],
        )
    else:  # cylinder
        radius = random.uniform(0.1, 0.2)
        height = random.uniform(0.2, 0.3)
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
    pos = [random.uniform(-5, 5), random.uniform(-5, 5), 0.25]
    orientation = p.getQuaternionFromEuler([0, 0, random.uniform(0, 3.14)])

    # Mass > 0 to make it dynamic
    mass = random.uniform(1, 5)

    # Create the object
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos,
        baseOrientation=orientation,
    )

    return body_id


def print_joint_info(body_id):
    JOINT_TYPES = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_SPHERICAL: "SPHERICAL",
        p.JOINT_PLANAR: "PLANAR",
        p.JOINT_FIXED: "FIXED",
    }

    headers = [
        "JointIdx",
        "JointName",
        "JointType",
        "qIndex",
        "uIndex",
        "Flags",
        "JointDamping",
        "JointFriction",
        "LowerLimit",
        "UpperLimit",
        "MaxForce",
        "MaxVelocity",
        "LinkName",
        "JointAxis",
        "ParentFramePos",
        "ParentFrameOrn",
        "ParentIndex",
    ]

    rows = []

    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        joint_axis = np.round(info[13], 3)  # vec3
        parent_pos = np.round(info[14], 3)  # vec3
        parent_orn = np.round(info[15], 3)  # vec4

        rows.append(
            [
                info[0],  # jointIndex
                info[1].decode("utf-8"),  # jointName
                JOINT_TYPES.get(info[2], info[2]),  # jointType
                info[3],  # qIndex
                info[4],  # uIndex
                info[5],  # flags
                round(info[6], 3),  # jointDamping
                round(info[7], 3),  # jointFriction
                round(info[8], 3),  # jointLowerLimit
                round(info[9], 3),  # jointUpperLimit
                round(info[10], 3),  # jointMaxForce
                round(info[11], 3),  # jointMaxVelocity
                info[12].decode("utf-8"),  # linkName
                joint_axis,
                parent_pos,
                parent_orn,
                info[16],  # parentIndex
            ]
        )

    print("\n=== Joint Info for Body ID {} ===\n".format(body_id))
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", floatfmt=".3f"))


def print_link_states(body_id):
    """
    Prints all fields returned by getLinkState() using the exact PyBullet names
    and ordering from the official docs.
    """

    headers = [
        "ID",
        "JointName",
        "linkWorldPosition",
        "linkWorldOrientation",
        "localInertialFramePosition",
        "localInertialFrameOrientation",
        "worldLinkFramePosition",
        "worldLinkFrameOrientation",
        "worldLinearVelocity",
        "worldAngularVelocity",
    ]

    rows = []

    # BASE LINK (index = -1)
    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    base_lin, base_ang = p.getBaseVelocity(body_id)

    rows.append(
        [
            -1,
            "base",
            base_pos,
            base_orn,
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            base_lin,
            base_ang,
        ]
    )

    # CHILD LINKS
    for i in range(p.getNumJoints(body_id)):
        state = p.getLinkState(
            body_id,
            i,
            computeForwardKinematics=1,
            computeLinkVelocity=1,
        )

        (
            linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinearVelocity,
            worldAngularVelocity,
        ) = state

        joint_name = p.getJointInfo(body_id, i)[1].decode("utf-8")

        rows.append(
            [
                i,
                joint_name,
                np.round(linkWorldPosition, 3),
                np.round(linkWorldOrientation, 3),
                np.round(localInertialFramePosition, 3),
                np.round(localInertialFrameOrientation, 3),
                np.round(worldLinkFramePosition, 3),
                np.round(worldLinkFrameOrientation, 3),
                np.round(worldLinearVelocity, 3),
                np.round(worldAngularVelocity, 3),
            ]
        )

    print("\n=== Link States for Body ID {} ===\n".format(body_id))
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def print_dynamics_info(body_id):
    """
    Prints full dynamics information (mass, inertia, friction, restitution, etc.)
    for the base and each link of a PyBullet body.
    """
    headers = [
        "ID",
        "Link",
        "Mass",
        "Lateral_Fric",
        "Inertia_Diag",
        "Inertia_Pos",
        "Inertia_Orn",
        "Restitution",
        "Rolling_Fric",
        "Spinning_Fric",
        "Contact_Damp",
        "Contact_Stiff",
        "Body_Type",
        "Collision_Margin",
    ]

    rows = []

    # Base (link index = -1)
    base_dyn = p.getDynamicsInfo(body_id, -1)
    rows.append(
        [
            -1,
            "base",
            base_dyn[0],  # mass
            base_dyn[1],  # lateral friction
            np.round(base_dyn[2], 3),  # local inertia diagonal
            np.round(base_dyn[3], 3),  # local inertial position
            np.round(base_dyn[4], 3),  # local inertial orientation
            base_dyn[5],  # restitution
            base_dyn[6],  # rolling friction
            base_dyn[7],  # spinning friction
            base_dyn[8],  # contact damping
            base_dyn[9],  # contact stiffness
            base_dyn[10],  # body type
            base_dyn[11],  # collision margin
        ]
    )

    # Links
    for i in range(p.getNumJoints(body_id)):
        info = p.getDynamicsInfo(body_id, i)
        name = p.getJointInfo(body_id, i)[12].decode(
            "utf-8"
        )  # link name from joint info
        rows.append(
            [
                i,
                name,
                info[0],  # mass
                info[1],  # lateral friction
                np.round(info[2], 3),  # local inertia diagonal
                np.round(info[3], 3),  # local inertial position
                np.round(info[4], 3),  # local inertial orientation
                info[5],  # restitution
                info[6],  # rolling friction
                info[7],  # spinning friction
                info[8],  # contact damping
                info[9],  # contact stiffness
                info[10],  # body type
                info[11],  # collision margin
            ]
        )

    # Print title
    print("\n=== Dynamics Info for Body ID {} ===\n".format(body_id))
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

    print_joint_info(r2d2_id)
    print_link_states(r2d2_id)
    print_dynamics_info(r2d2_id)

    HEAD_LINK = 13
    HEAD_BOX_LINK = 14
    RIGHT_FRONT_WHEEL = 2
    RIGHT_BACK_WHEEL = 3
    LEFT_FRONT_WHEEL = 6
    LEFT_BACK_WHEEL = 7
    LEFT_GRIPPER = 9
    RIGHT_GRIPPER = 11
    GRIPPER_POLE = 8

    # for _ in range(5):
    #     spawn_random_object()

    spawn_pickable_object("box")

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
    wheel_current_target_velocities = [0, 0, 0, 0]
    wheel_target_velocities = [0, 0, 0, 0]  # Radians / S
    head_current_velocity = 0
    head_current_target_velocity = 0
    for i in range(10000):
        frame += 1

        wheel_target_velocities = [0, 0, 0, 0]

        keys = p.getKeyboardEvents()
        pos, orn = p.getBasePositionAndOrientation(r2d2_id)
        pos_x, pos_y, pos_z = pos

        if frame % 50 == 0:
            print(
                f"{frame}, robotId: {r2d2_id}, robotBaseXYZ={np.round(pos, 2)} robotBaseOrnQuat: {np.round(orn, 2)}"
            )

        if ord("e") in keys and keys[ord("e")] & p.KEY_IS_DOWN:  # forward
            wheel_target_velocities = [-25, -25, -25, -25]
            forces = [5, 5, 5, 5]

        elif ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:  # backward
            wheel_target_velocities = [25, 25, 25, 25]
            forces = [5, 5, 5, 5]

        elif ord("j") in keys and keys[ord("j")] & p.KEY_IS_DOWN:
            wheel_target_velocities = [-50, -50, 50, 50]
            forces = [20, 20, 20, 20]

        elif ord("l") in keys and keys[ord("l")] & p.KEY_IS_DOWN:
            wheel_target_velocities = [50, 50, -50, -50]
            forces = [20, 20, 20, 20]
        else:
            wheel_target_velocities = [0, 0, 0, 0]
            forces = [2, 2, 2, 2]

        if wheel_target_velocities != wheel_current_target_velocities:
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
            wheel_current_target_velocities = wheel_target_velocities

        # if ord("i") in keys and keys[ord("i")] & p.KEY_IS_DOWN:
        #     head_current_target_velocity = 5
        # elif ord("k") in keys and keys[ord("k")] & p.KEY_IS_DOWN:
        #     head_current_target_velocity = -5
        # else:
        #     head_current_target_velocity = 0

        if head_current_target_velocity != head_current_velocity:
            p.setJointMotorControl2(
                r2d2_id,
                HEAD_LINK,
                p.VELOCITY_CONTROL,
                targetVelocity=head_current_target_velocity,
            )
            head_current_target_velocity = wheel_target_velocities

        if ord("o") in keys and keys[ord("o")] & p.KEY_WAS_TRIGGERED:
            p.setJointMotorControl2(
                r2d2_id,
                LEFT_GRIPPER,
                p.POSITION_CONTROL,
                targetPosition=0.548,
                force=50,
            )
            p.setJointMotorControl2(
                r2d2_id,
                RIGHT_GRIPPER,
                p.POSITION_CONTROL,
                targetPosition=0.548,
                force=50,
            )

        if ord("p") in keys and keys[ord("p")] & p.KEY_WAS_TRIGGERED:
            p.setJointMotorControl2(
                r2d2_id,
                LEFT_GRIPPER,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=50,
            )
            p.setJointMotorControl2(
                r2d2_id,
                RIGHT_GRIPPER,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=50,
            )

        if ord("u") in keys and keys[ord("u")] & p.KEY_WAS_TRIGGERED:
            p.setJointMotorControl2(
                r2d2_id,
                GRIPPER_POLE,
                p.POSITION_CONTROL,
                targetPosition=-0.300,
                force=50,
            )
        if ord("i") in keys and keys[ord("i")] & p.KEY_WAS_TRIGGERED:
            p.setJointMotorControl2(
                r2d2_id,
                GRIPPER_POLE,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=50,
            )

        if frame % 3 == 0:  # every 10th frame → ~24 FPS equivalent
            view, proj = get_head_camera_view(r2d2_id, HEAD_BOX_LINK)
            _, _, rgb, depth, seg = p.getCameraImage(width, height, view, proj)
            rgb_array = np.reshape(rgb, (height, width, 4))
            # print(f"Head camera frame: mean RGB={rgb_array[..., :3].mean():.2f}")

        # STEP
        p.stepSimulation()
        # time.sleep(1.0 / 240.0)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(r2d2_id)
    print(cubePos, cubeOrn)
    p.disconnect()
