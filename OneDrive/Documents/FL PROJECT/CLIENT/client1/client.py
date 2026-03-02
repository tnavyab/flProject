# ============================================================
#                IDRiD CLIENT (FEDPROX + SAFE CM)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from dataset import IDRiDDataset

# ============================================================
#                       CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

CSV_PATH = r"C:\Users\tnavy\OneDrive\Documents\FL PROJECT\CLIENT\DATA\idrid_labels.csv"
IMG_DIR  = r"C:\Users\tnavy\OneDrive\Documents\FL PROJECT\CLIENT\DATA\images"

DR_CLASSES  = 2
DME_CLASSES = 3

BATCH_SIZE   = 8
LOCAL_EPOCHS = 5
LR           = 1e-4
MU           = 0.01   # FedProx coefficient


# ============================================================
#                    ATTENTION MODULES
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.se(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * s


# ============================================================
#                    MULTI-TASK MODEL
# ============================================================

class DenseNet_MultiTask(nn.Module):
    def __init__(self, dr_classes, dme_classes):
        super().__init__()
        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.att = CBAM(1024)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dr_head = nn.Linear(1024, dr_classes)
        self.dme_head = nn.Linear(1024, dme_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.att(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.dr_head(x), self.dme_head(x)


model = DenseNet_MultiTask(DR_CLASSES, DME_CLASSES).to(DEVICE)

# Freeze backbone except denseblock4
for name, p in model.features.named_parameters():
    p.requires_grad = "denseblock4" in name


# ============================================================
#                          DATA
# ============================================================

dataset = IDRiDDataset(
    CSV_PATH,
    IMG_DIR,
    is_train=True,
    dr_stage="binary"
)

indices = torch.randperm(len(dataset)).tolist()
split = int(0.8 * len(indices))

train_set = Subset(dataset, indices[:split])
val_set   = Subset(dataset, indices[split:])

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, BATCH_SIZE, shuffle=False)


# ============================================================
#                    LOSS & OPTIMIZER
# ============================================================

dr_loss_fn  = nn.CrossEntropyLoss()
dme_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# ============================================================
#                    FEDPROX TRAINING
# ============================================================

def train_fedprox(global_params):
    model.train()
    total_loss = 0.0

    for imgs, dr, dme in train_loader:
        imgs, dr, dme = imgs.to(DEVICE), dr.to(DEVICE), dme.to(DEVICE)

        optimizer.zero_grad()
        dr_out, dme_out = model(imgs)

        loss = dr_loss_fn(dr_out, dr) + dme_loss_fn(dme_out, dme)

        prox_term = 0.0
        for p, g in zip(model.parameters(), global_params):
            prox_term += torch.norm(p - g) ** 2

        loss += (MU / 2) * prox_term

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ============================================================
#                       EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model():
    model.eval()

    dr_true, dr_pred = [], []
    dme_true, dme_pred = [], []

    total_loss = 0.0

    for imgs, dr, dme in val_loader:
        imgs, dr, dme = imgs.to(DEVICE), dr.to(DEVICE), dme.to(DEVICE)

        dr_out, dme_out = model(imgs)

        loss = dr_loss_fn(dr_out, dr) + dme_loss_fn(dme_out, dme)
        total_loss += loss.item()

        dr_predictions = torch.argmax(dr_out, dim=1)
        dme_predictions = torch.argmax(dme_out, dim=1)

        dr_true.extend(dr.cpu().numpy())
        dr_pred.extend(dr_predictions.cpu().numpy())

        dme_true.extend(dme.cpu().numpy())
        dme_pred.extend(dme_predictions.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    dr_cm = confusion_matrix(dr_true, dr_pred, labels=[0, 1])
    dme_cm = confusion_matrix(dme_true, dme_pred, labels=[0, 1, 2])

    metrics = {
        "dr_acc": float(accuracy_score(dr_true, dr_pred)),
        "dr_f1": float(f1_score(dr_true, dr_pred, average="weighted")),
        "dme_acc": float(accuracy_score(dme_true, dme_pred)),
        "dme_f1": float(f1_score(dme_true, dme_pred, average="weighted")),

        "dr_00": int(dr_cm[0][0]),
        "dr_01": int(dr_cm[0][1]),
        "dr_10": int(dr_cm[1][0]),
        "dr_11": int(dr_cm[1][1]),

        "dme_00": int(dme_cm[0][0]),
        "dme_01": int(dme_cm[0][1]),
        "dme_02": int(dme_cm[0][2]),
        "dme_10": int(dme_cm[1][0]),
        "dme_11": int(dme_cm[1][1]),
        "dme_12": int(dme_cm[1][2]),
        "dme_20": int(dme_cm[2][0]),
        "dme_21": int(dme_cm[2][1]),
        "dme_22": int(dme_cm[2][2]),
    }

    return avg_loss, metrics


# ============================================================
#                      FLOWER CLIENT
# ============================================================

class IDRiDClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(model.parameters(), parameters):
            p.data = torch.tensor(new_p, device=DEVICE, dtype=p.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in model.parameters()]

        loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            loss += train_fedprox(global_params)

        loss /= LOCAL_EPOCHS

        return self.get_parameters(config), len(train_set), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        avg_loss, metrics = evaluate_model()
        return float(avg_loss), len(val_set), metrics


# ============================================================
#                        START CLIENT
# ============================================================

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=IDRiDClient()
    )
