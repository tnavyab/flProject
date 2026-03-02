# utils.py
import cv2
import numpy as np
import torch

def weinmed_preprocess(img_tensor):
    """
    img_tensor: torch.Tensor (C, H, W) with values in [0,1] or [0,255]
    returns: torch.Tensor (3, H, W) normalized
    """

    # ---- Torch → NumPy ----
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    # ---- Convert to uint8 safely ----
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # ---- Convert to grayscale ----
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---- Histogram Equalization (WEINMED core) ----
    gray_eq = cv2.equalizeHist(gray)

    # ---- Back to 3 channels ----
    img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)

    # ---- Normalize to [0,1] ----
    img_eq = img_eq.astype(np.float32) / 255.0

    # ---- NumPy → Torch ----
    img_eq = torch.from_numpy(img_eq).permute(2, 0, 1)

    return img_eq
