# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class IDRiDDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        is_train=True,
        dr_stage="binary",   # "binary" or "multi"
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.is_train = is_train

        # --------------------------------------------------
        # KEEP ONLY GRADABLE IMAGES
        # --------------------------------------------------
        if "adjudicated_gradable" in self.df.columns:
            self.df = self.df[self.df["adjudicated_gradable"] == 1]

        # --------------------------------------------------
        # DR LABEL
        # --------------------------------------------------
        self.df["diagnosis"] = self.df["diagnosis"].astype(int)

        if dr_stage == "binary":
            self.df["dr_label"] = self.df["diagnosis"].apply(
                lambda x: 0 if x == 0 else 1
            )
        else:
            self.df["dr_label"] = self.df["diagnosis"]

        # --------------------------------------------------
        # DME LABEL
        # --------------------------------------------------
        if "adjudicated_dme" in self.df.columns:
            self.df["dme_label"] = self.df["adjudicated_dme"].astype(int)
        else:
            raise ValueError("❌ No DME column found")

        # --------------------------------------------------
        # FILTER VALID IMAGES
        # --------------------------------------------------
        valid_rows = []
        missing = 0

        for _, row in self.df.iterrows():
            img_name = str(row["id_code"])
            img_path = os.path.join(self.img_dir, img_name)

            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                missing += 1

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # --------------------------------------------------
        # TRANSFORMS
        # --------------------------------------------------
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # --------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------
        if len(self.df) == 0:
            raise RuntimeError(
                "❌ Dataset EMPTY!\n"
                "Check:\n"
                "1) CSV id_code\n"
                "2) Image path\n"
                "3) File extensions"
            )

        print("✅ Dataset loaded")
        print(f"Samples : {len(self.df)}")
        print(f"Missing : {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["id_code"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        dr = torch.tensor(row["dr_label"], dtype=torch.long)
        dme = torch.tensor(row["dme_label"], dtype=torch.long)

        return image, dr, dme
