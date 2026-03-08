import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# 1. Create a dummy "Visual State" (1 Batch, 3 Tokens, 4 Features)
# Let's pretend these are 3 patches of your robot (Head, Torso, Legs)
state = torch.zeros((1, 3, 4))
state[0, 0, :] = 1.0  # Head is all 1s
state[0, 1, :] = 2.0  # Torso is all 2s
state[0, 2, :] = 3.0  # Legs are all 3s

print("--- ORIGINAL STATE ---")
print(state)

# 2. Create a dummy "Action" (1 Batch, 1 Token, 4 Features)
# Let's say the action vector is [0.1, 0.5, 0.9, 0.0]
action = torch.tensor([[[0.1, 0.5, 0.9, 0.0]]])

# 3. THE BROADCAST ADDITION
combined = action + state

print("\n--- AFTER BROADCAST ADDITION ---")
print(combined)
