import os
from PIL import Image
import numpy as np
from utils import JEPAEngine

# Force PyTorch to use ALL available CPU cores
# cores = multiprocessing.cpu_count()
# torch.set_num_threads(cores)
# torch.set_num_interop_threads(cores)
# print(f"⚡ CPU Overclock: Utilizing all {cores} cores.")

# ==============================================================================
# 1. SETUP THE V-JEPA 2 ARCHITECTURE
# ==============================================================================

my_jepa_model = JEPAEngine()

# HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"

# print(f"🧠 Loading Meta's {HF_REPO}...")
# processor = AutoVideoProcessor.from_pretrained(HF_REPO)
# model = AutoModel.from_pretrained(HF_REPO)
# model.eval()


# def get_raw_latent_grid(image_path):
#     """Extracts the raw 16x16 spatial tokens from a static image."""
#     if not os.path.exists(image_path):
#         print(f"⚠️ Error: Could not find {image_path}")
#         return None

#     # Start the timers
#     t0_wall = time.perf_counter()
#     t0_cpu = time.process_time()

#     img = Image.open(image_path).convert("RGB")
#     video_clip = np.asarray([img] * 16)

#     inputs = processor(video_clip, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(**inputs)
#         raw_latent = outputs.last_hidden_state

#     # Stop the timers
#     t1_wall = time.perf_counter()
#     t1_cpu = time.process_time()

#     wall_time = t1_wall - t0_wall
#     cpu_time = t1_cpu - t0_cpu

#     print(f"   ⏱️ Profiler -> Wall Time: {wall_time:.2f}s | CPU Time: {cpu_time:.2f}s")

#     return raw_latent.to(torch.float32)


# ==============================================================================
# 2. DEFINE THE TARGET AND TEST POSES
# ==============================================================================
TARGET_POSE = "poses/standing_pose.jpg"

# Removed "The Goal" from here so we don't calculate it twice!
TEST_POSES = {
    "Knee Tuck": "poses/knee_tuck.jpg",
    "Plank": "poses/plank_pose.jpg",
    "Starting Line (Lying Down)": "poses/lying_pose.jpg",
    "No ROBOT": "poses/no_robot.jpg",
}

# ==============================================================================
# 3. COMPUTE ENERGY DISTANCES (MSE)
# ==============================================================================
print(f"\nExtracting Latent Encoding from: {TARGET_POSE}")
img = Image.open(TARGET_POSE).convert("RGB")
video_clip = np.asarray([img] * 16)
target_grid = my_jepa_model.get_latent(video_clip, verbose=True)

if target_grid is None:
    print("❌ Aborting: Cannot find target image.")
    exit()

print(f"Encoded Target Shape: {target_grid.shape}")

print("📊 --- V-JEPA 2 RAW LATENT LEADERBOARD ---")
print("Using Mean Squared Error (MSE) - Lower is closer to Target...\n")

results = {}

# Manually inject the target score so it still shows up on the leaderboard
results["The Goal (Standing)"] = 0.0

for pose_name, filepath in TEST_POSES.items():
    print(f"Predicting: {pose_name}...")
    img = Image.open(filepath).convert("RGB")
    video_clip = np.asarray([img] * 16)
    current_grid = my_jepa_model.get_latent(video_clip, verbose=True)

    if current_grid is not None:
        mse_dist = my_jepa_model.compute_mse(target_grid, current_grid)
        results[pose_name] = mse_dist

sorted_results = sorted(results.items(), key=lambda item: item[1])

print("\n" + "=" * 60)
for rank, (name, mse) in enumerate(sorted_results, 1):
    print(f"{rank}. {name}")
    print(f"   MSE Energy Distance: {mse:.6f}")
print("=" * 60 + "\n")
