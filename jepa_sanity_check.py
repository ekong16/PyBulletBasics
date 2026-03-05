import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import torch
import torch.nn.functional as F
from transformers import AutoVideoProcessor, AutoModel
from PIL import Image
import numpy as np

# Suppress duplicate library warnings on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# 1. SETUP THE V-JEPA 2 ARCHITECTURE
# ==============================================================================
HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"

print(f"🧠 Loading Meta's {HF_REPO}...")
processor = AutoVideoProcessor.from_pretrained(HF_REPO)
# AutoModel gives us the raw hidden states (the math space)
model = AutoModel.from_pretrained(HF_REPO)
model.eval()


def get_raw_latent_grid(image_path):
    """Extracts the raw 16x16 spatial tokens from a static image."""
    if not os.path.exists(image_path):
        print(f"⚠️ Error: Could not find {image_path}")
        return None

    img = Image.open(image_path).convert("RGB")

    # 'Stillness Trick': Repeat image to create a 16-frame 'static video'
    video_clip = np.asarray([img] * 16)
    print("video clip shape: ", video_clip.shape)
    inputs = processor(video_clip, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # raw_latent shape: [Batch, Tokens, Dim] -> [1, 256, 1024]
        raw_latent = outputs.last_hidden_state

    return raw_latent


# ==============================================================================
# 2. DEFINE THE TARGET AND TEST POSES
# ==============================================================================
TARGET_POSE = "poses/standing_pose.jpg"
TEST_POSES = {
    "The Goal (Standing)": "poses/standing_pose.jpg",
    "Knee Tuck": "poses/knee_tuck.jpg",
    "Plank": "poses/plank_pose.jpg",
    "Starting Line (Lying Down)": "poses/lying_pose.jpg",
}

# ==============================================================================
# 3. COMPUTE ENERGY DISTANCES (MSE)
# ==============================================================================
print(f"🎯 Extracting Target Grid from: {TARGET_POSE}")
target_grid = get_raw_latent_grid(TARGET_POSE)

if target_grid is None:
    print("❌ Aborting: Cannot find target image.")
    exit()

# Printing the shape as requested
print(f"📐 Encoded Target Shape: {target_grid.shape}")
print(f"🧩 Interpretation: [Batch: 1, Tokens: 256 (16x16), Latent_Dim: 1024]\n")

print("📊 --- V-JEPA 2 RAW LATENT LEADERBOARD ---")
print("Using Mean Squared Error (MSE) - Lower is closer to Target...\n")

results = {}

for pose_name, filepath in TEST_POSES.items():
    print("Predicting: ", pose_name)
    current_grid = get_raw_latent_grid(filepath)
    print("...done!")
    print("THE SHAPE IS: ", current_grid.shape)

    if current_grid is not None:
        # With raw grids, MSE is the standard metric to measure distance
        mse_dist = F.mse_loss(target_grid, current_grid).item()
        results[pose_name] = mse_dist

# Sort results from lowest MSE (best) to highest
sorted_results = sorted(results.items(), key=lambda item: item[1])

print("\n" + "=" * 60)
for rank, (name, mse) in enumerate(sorted_results, 1):
    print(f"{rank}. {name}")
    print(f"   MSE Energy Distance: {mse:.6f}")
print("=" * 60 + "\n")
