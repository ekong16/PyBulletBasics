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
