# ============================================================
#      FLOWER SERVER (GLOBAL CM + MODEL SAVE)
# ============================================================

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

# ============================================================
#                 CREATE SAVE DIRECTORIES
# ============================================================

CM_DIR = "server_confusion_matrices"
MODEL_DIR = "server_models"

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
#            FUNCTION TO SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ============================================================
#                 CUSTOM STRATEGY
# ============================================================

class FedAvgWithCMAndSave(fl.server.strategy.FedAvg):

    # 🔹 Save global model after aggregation
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            model_path = os.path.join(
                MODEL_DIR, f"global_model_round_{rnd}.pth"
            )

            torch.save(ndarrays, model_path)
            print(f"✅ Global model saved: {model_path}")

        return aggregated_parameters, aggregated_metrics


    # 🔹 Aggregate evaluation + plot confusion matrix
    def aggregate_evaluate(self, rnd, results, failures):

        if not results:
            return None, {}

        losses = [r.loss for _, r in results]
        avg_loss = float(np.mean(losses))

        # -----------------------------
        # DR Confusion Matrix
        # -----------------------------
        dr_cm_global = np.zeros((2, 2), dtype=int)
        for _, r in results:
            dr_cm = np.array([
                [r.metrics["dr_00"], r.metrics["dr_01"]],
                [r.metrics["dr_10"], r.metrics["dr_11"]],
            ])
            dr_cm_global += dr_cm

        # -----------------------------
        # DME Confusion Matrix
        # -----------------------------
        dme_cm_global = np.zeros((3, 3), dtype=int)
        for _, r in results:
            dme_cm = np.array([
                [r.metrics["dme_00"], r.metrics["dme_01"], r.metrics["dme_02"]],
                [r.metrics["dme_10"], r.metrics["dme_11"], r.metrics["dme_12"]],
                [r.metrics["dme_20"], r.metrics["dme_21"], r.metrics["dme_22"]],
            ])
            dme_cm_global += dme_cm

        # -----------------------------
        # Compute Accuracy
        # -----------------------------
        dr_acc = np.trace(dr_cm_global) / np.sum(dr_cm_global)
        dme_acc = np.trace(dme_cm_global) / np.sum(dme_cm_global)

        # -----------------------------
        # Print Results
        # -----------------------------
        print("\n" + "="*60)
        print(f"🌟 ROUND {rnd} GLOBAL RESULTS")
        print("="*60)

        print(f"\nAverage Loss : {avg_loss:.4f}")

        print("\nDR Confusion Matrix:")
        print(dr_cm_global)
        print(f"DR Accuracy : {dr_acc:.4f}")

        print("\nDME Confusion Matrix:")
        print(dme_cm_global)
        print(f"DME Accuracy : {dme_acc:.4f}")
        print("="*60 + "\n")

        # -----------------------------
        # SAVE CONFUSION MATRIX IMAGES
        # -----------------------------

        save_confusion_matrix(
            dr_cm_global,
            classes=["No DR", "DR"],
            title=f"DR Confusion Matrix - Round {rnd}",
            filename=os.path.join(CM_DIR, f"DR_CM_Round_{rnd}.png")
        )

        save_confusion_matrix(
            dme_cm_global,
            classes=["Grade 0", "Grade 1", "Grade 2"],
            title=f"DME Confusion Matrix - Round {rnd}",
            filename=os.path.join(CM_DIR, f"DME_CM_Round_{rnd}.png")
        )

        return avg_loss, {
            "dr_acc": float(dr_acc),
            "dme_acc": float(dme_acc),
        }


# ============================================================
#                    STRATEGY CONFIG
# ============================================================

strategy = FedAvgWithCMAndSave(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

# ============================================================
#                    START SERVER
# ============================================================

if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
