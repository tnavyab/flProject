# utils.py
import cv2
import torch
import numpy as np

def weinmed_preprocess(img_tensor):
    """
    Input : torch.Tensor (C, H, W) in range [0,1]
    Output: torch.Tensor (C, H, W)
    """

    # Tensor → NumPy (H, W, C)
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Convert to grayscale
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Histogram Equalization
    img_eq = cv2.equalizeHist(img_gray)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_eq)

    # Back to 3 channels
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # Normalize & Tensor
    img_clahe = img_clahe.astype(np.float32) / 255.0
    img_clahe = torch.from_numpy(img_clahe).permute(2, 0, 1)

    return img_clahe
