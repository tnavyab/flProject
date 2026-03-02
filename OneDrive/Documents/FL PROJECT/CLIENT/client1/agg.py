import os
import re
import torch
import numpy as np
from collections import OrderedDict

# =====================================================
# CONFIG
# =====================================================
MODEL_DIR = "server_models"
OUTPUT_MODEL = "final_global_model.pth"
DEVICE = "cpu"

# =====================================================
# SORT MODELS NUMERICALLY (Fix round_10 issue)
# =====================================================
def extract_round(filename):
    match = re.search(r"round_(\d+)", filename)
    return int(match.group(1)) if match else 0

model_files = [
    os.path.join(MODEL_DIR, f)
    for f in os.listdir(MODEL_DIR)
    if f.endswith(".pth")
]

model_files.sort(key=extract_round)

assert len(model_files) > 0, "No global models found!"

print(f"Found {len(model_files)} global models:")
for f in model_files:
    print(" -", f)

# =====================================================
# LOAD FIRST MODEL TO CHECK FORMAT
# =====================================================
first_model = torch.load(
    model_files[0],
    map_location=DEVICE,
    weights_only=False
)

# =====================================================
# CASE 1: Flower format (List of numpy arrays)
# =====================================================
if isinstance(first_model, list):

    print("\nDetected Flower-style list format. Performing layer-wise averaging...\n")

    all_models = []

    for path in model_files:
        weights = torch.load(path, map_location=DEVICE, weights_only=False)
        all_models.append(weights)

    num_models = len(all_models)
    num_layers = len(all_models[0])

    averaged_weights = []

    for layer_idx in range(num_layers):
        layer_sum = np.zeros_like(all_models[0][layer_idx])

        for model in all_models:
            layer_sum += model[layer_idx]

        layer_avg = layer_sum / num_models
        averaged_weights.append(layer_avg)

    # Save averaged list (Flower-compatible)
    torch.save(averaged_weights, OUTPUT_MODEL)

# =====================================================
# CASE 2: Proper PyTorch state_dict format
# =====================================================
elif isinstance(first_model, dict):

    print("\nDetected PyTorch state_dict format. Performing key-wise averaging...\n")

    avg_state_dict = OrderedDict()

    for i, model_path in enumerate(model_files):
        state_dict = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False
        )

        if i == 0:
            for k, v in state_dict.items():
                avg_state_dict[k] = v.clone().float()
        else:
            for k, v in state_dict.items():
                avg_state_dict[k] += v.float()

    num_models = len(model_files)

    for k in avg_state_dict:
        avg_state_dict[k] /= num_models

    torch.save(avg_state_dict, OUTPUT_MODEL)

else:
    raise TypeError("Unsupported model format!")

# =====================================================
# DONE
# =====================================================
print(f"\n✅ Final aggregated global model saved as: {OUTPUT_MODEL}")
