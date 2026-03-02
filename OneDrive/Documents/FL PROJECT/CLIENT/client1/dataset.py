import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# OPTIONAL: comment this if you don't want WeinMed
try:
    from utils import weinmed_preprocess
    USE_WEINMED = True
except:
    USE_WEINMED = False


class IDRiDDataset(Dataset):
    """
    Multi-task Dataset
    Tasks:
      - DR classification (binary or multi-class)
      - DME classification (0,1,2)
    """

    def __init__(
        self,
        csv_path,
        img_dir,
        is_train=True,
        dr_stage="binary",   # "binary" | "multi"
    ):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()  # 🔥 FIX column issues
        self.img_dir = img_dir
        self.is_train = is_train
        self.dr_stage = dr_stage

        # ---------- REQUIRED COLUMNS ----------
        required_cols = ["id_code", "diagnosis", "Risk of macular edema"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column: {c}")

        # ---------- CLEAN ----------
        self.df = self.df.dropna(subset=["diagnosis", "Risk of macular edema"])
        self.df["diagnosis"] = self.df["diagnosis"].astype(int)
        self.df["Risk of macular edema"] = self.df["Risk of macular edema"].astype(int)
        self.df.reset_index(drop=True, inplace=True)

        # ---------- LABELS ----------
        # DR
        if self.dr_stage == "binary":
            self.df["dr_label"] = self.df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
        else:
            self.df["dr_label"] = self.df["diagnosis"]

        # DME (0,1,2)
        self.df["dme_label"] = self.df["Risk of macular edema"]

        # ---------- TRANSFORMS ----------
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(
            f"✅ Dataset loaded | samples={len(self.df)} | "
            f"DR={self.dr_stage} | Multi-task"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.jpg")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if USE_WEINMED:
            image = weinmed_preprocess(image)

        image = self.normalize(image)

        dr_label = torch.tensor(row["dr_label"], dtype=torch.long)
        dme_label = torch.tensor(row["dme_label"], dtype=torch.long)

        return image, dr_label, dme_label
