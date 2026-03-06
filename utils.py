import pybullet as p
import time
import pybullet_data
import numpy as np
from tabulate import tabulate
import math
import pkgutil

from PIL import Image

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoVideoProcessor, AutoModel

# Global cache to prevent redundant loading across scripts
_JEPA_CACHE = {"model": None, "processor": None}


def enable_headless_opengl(client_id):
    """
    Enables hardware OpenGL rendering for headless (p.DIRECT) sessions.
    Safe to call even in GUI mode; it will just skip if EGL isn't needed.
    """
    conn_info = p.getConnectionInfo(physicsClientId=client_id)
    if conn_info["connectionMethod"] == p.DIRECT:
        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            plugin_id = p.loadPlugin(
                egl.get_filename(), "_eglRendererPlugin", physicsClientId=client_id
            )
            if plugin_id >= 0:
                print(
                    "🚀 EGL Render Plugin loaded (Hardware OpenGL enabled for headless)"
                )
            return plugin_id
        else:
            print("EGL not found!!!!")
    return -1


class JEPAEngine:
    def __init__(self, model_id="facebook/vjepa2-vitl-fpc16-256-ssv2", device="mps"):
        self.device = device

        # Hardware Overclock: Only set if not already configured ... not use it for now
        # if not torch.get_num_threads() > 1:
        #     cores = multiprocessing.cpu_count()
        #     torch.set_num_threads(cores)
        #     torch.set_num_interop_threads(cores)

        if _JEPA_CACHE["model"] is None:
            print(f"🧠 Loading {model_id} into RAM...")
            _JEPA_CACHE["processor"] = AutoVideoProcessor.from_pretrained(model_id)
            _JEPA_CACHE["model"] = AutoModel.from_pretrained(model_id).to(self.device)
            _JEPA_CACHE["model"].eval()

        self.model = _JEPA_CACHE["model"]
        self.processor = _JEPA_CACHE["processor"]

    def get_latent(self, video_frames, verbose=False):
        """
        Extracts latents with optional profiling and dtype inspection.
        video_frames: List or Array of 16 frames (uint8).
        """
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        inputs = self.processor(video_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Standard V-JEPA 2 output is the last hidden state
            raw_latent = outputs.last_hidden_state

        if verbose:
            wall_time = time.perf_counter() - t0_wall
            cpu_time = time.process_time() - t0_cpu

            # The "Inspector" logic
            input_dtype = inputs[next(iter(inputs))].dtype
            model_dtype = next(self.model.parameters()).dtype

            print(f"\n--- [JEPA DEBUG PROFILE] ---")
            print(
                f"Inputs: {input_dtype} | Model: {model_dtype} | Latent: {raw_latent.dtype}"
            )
            print(f"Input Shape: {video_frames.shape}")
            print(
                f"Raw Latent Shape: {raw_latent.shape} (Tokens: {raw_latent.shape[1]})"
            )
            print(f"⏱️ Wall Time: {wall_time:.2f}s | CPU Time: {cpu_time:.2f}s")
            print(f"----------------------------")

        return raw_latent.to(torch.float32)

    def compute_mse(self, current, target):
        return F.mse_loss(current, target).item()


class PyBulletCamera:
    def __init__(
        self,
        width=256,
        height=256,
        target_pos=[0, 0, 0.5],
        distance=3.6,
        yaw=45,
        pitch=-30,
    ):
        self.width = width
        self.height = height
        self.target_pos = target_pos
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.roll = 0
        self.fov = 60
        self.near = 0.1
        self.far = 100.0
        self.latest_frame = None

    def update(self, client_id=0):
        """Calls the camera once. Feeds the PiP preview AND caches the image."""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target_pos,
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            upAxisIndex=2,
            physicsClientId=client_id,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=client_id,
        )

        # This single call automatically updates the PyBullet GUI preview window
        _, _, rgb_img, _, _ = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=client_id,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            shadow=1,  # Turn shadows ON
            lightDirection=[1, 1, 1],  # Point the "Sun" diagonally down
        )

        # Slice RGB and cache as 8-bit integers (retains color for PIL!)
        img_arr = np.reshape(rgb_img, (self.height, self.width, 4))[:, :, :3]
        self.latest_frame = np.uint8(img_arr)

    def get_last_image(self):
        """Instantly returns the cached frame as a PIL Image."""
        if self.latest_frame is not None:
            return Image.fromarray(self.latest_frame)
        return None


class PyBulletSim:
    def __init__(self, gui=True, disableRender=False):
        self.gui = gui
        self.disableRender = disableRender

    def __enter__(self):
        # Disconnect any leftover session
        if p.isConnected():
            p.disconnect()
        # Start a new connection
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        if self.gui and self.disableRender:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
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


# --- 1. CONFIGURATION CLASS ---
# Central place for all physics constants.
# Import this in your main script if you need to access SCALE.
class SimConfig:
    # Scale
    GLOBAL_SCALE = 0.32
    INITIAL_POS = [0, 0, 0.9]
    START_ORI = [0, math.pi / 2, 0]  # Euler (Roll, Pitch, Yaw)

    # Physics Engine
    PHYSICS_FREQ = 480
    ENGINE_PARAMS = {
        "numSubSteps": 4,  # Accuracy
        "frictionERP": 0.2,  # Grip
        "numSolverIterations": 150,  # Stability
        "erp": 0.2,  # Joint stiffness
        "contactSlop": 0.001,  # Anti-bounce
    }


# --- 2. SETUP FUNCTION ---
def setup_humanoid_scene(p_module):
    """
    Applies the standard physics config and loads the robot/plane.
    Args:
        p_module: The pybullet module (usually 'p')
    Returns:
        humanoid_id, plane_id
    """
    # A. Basic World Setup
    p_module.setAdditionalSearchPath(pybullet_data.getDataPath())
    p_module.setRealTimeSimulation(0)

    # B. Apply Physics Constants
    p_module.setTimeStep(1.0 / SimConfig.PHYSICS_FREQ)
    p_module.setPhysicsEngineParameter(**SimConfig.ENGINE_PARAMS)

    # C. Load Plane
    plane_id = p_module.loadURDF("plane.urdf")
    # Fix floor friction so the robot can actually push off
    p_module.changeDynamics(plane_id, -1, lateralFriction=1.0)

    # D. Load Robot with Correct Flags
    # USE_SELF_COLLISION: Arms can hit chest
    # EXCLUDE_PARENT: Thighs won't explode against Shins
    flags = (
        p_module.URDF_USE_SELF_COLLISION
        | p_module.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    )

    # (Optional) Fix Inertia if you are using custom masses
    # flags |= p_module.URDF_COMPUTE_FULL_INERTIA

    start_quat = p_module.getQuaternionFromEuler(SimConfig.START_ORI)

    humanoid_id = p_module.loadURDF(
        "humanoid/humanoid.urdf",
        SimConfig.INITIAL_POS,
        start_quat,
        globalScaling=SimConfig.GLOBAL_SCALE,
        flags=flags,
    )

    print(f"--- Loaded Humanoid (Scale: {SimConfig.GLOBAL_SCALE}) ---")
    return humanoid_id, plane_id


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
            np.round(base_pos, 3),
            np.round(base_orn, 3),
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            np.round(base_lin, 3),
            np.round(base_ang, 3),
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
