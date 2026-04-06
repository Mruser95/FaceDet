import matplotlib.pyplot as plt
import numpy as np
from FaceNetPack.data_processor import Process
from FaceNetPack.Model.Backbone import CowResNet
from FaceNetPack.Model.VisionTransformer import ViT
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

STATE_PATH = Path(__file__).resolve().parents[1] / "State" / "cloud_model.pth"

class Margin_cal:
    def __init__(self, beta=0.005):
        self.beta = beta
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0

    def paint(self):
        TPR = [g[0][0] for g in self.roc]
        FPR = [g[0][1] for g in self.roc]
        plt.figure(figsize=(6,6))
        plt.plot(FPR, TPR, label="ROC curve")
        plt.plot([0,1], [0,1], "--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.grid()
        plt.legend()
        plt.show()

    def margin(self, default=0.3, sim=None, by=None):
        if sim is None or by is None:
            if STATE_PATH.exists():
                ckpt = torch.load(STATE_PATH, map_location='cpu')
                if isinstance(ckpt, dict):
                    return float(ckpt.get("margin", default))
            return float(default)

        self.roc = []
        best_beta, best_acc = float(default), -1.0
        sim = torch.as_tensor(sim).detach().cpu().numpy()
        by = torch.as_tensor(by).detach().cpu().numpy()

        for beta in np.arange(-1.0, 1.0 + self.beta, self.beta):
            TP = ((sim >= beta) & (by == 1)).sum()
            FP = ((sim >= beta) & (by == 0)).sum()
            TN = ((sim < beta) & (by == 0)).sum()
            FN = ((sim < beta) & (by == 1)).sum()
            TPR = TP / np.clip((TP + FN), a_min=1e-9, a_max=np.inf)
            FPR = FP / np.clip((FP + TN), a_min=1e-9, a_max=np.inf)
            acc = (TP + TN) / np.clip(len(sim), a_min=1, a_max=np.inf)
            self.roc.append(((TPR, FPR), beta))
            if acc > best_acc:
                best_acc = acc
                best_beta = beta

        return float(best_beta)


def load_margin_model(state_path=STATE_PATH):
    ckpt = torch.load(state_path, map_location='cpu')
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = nn.Sequential(CowResNet(), ViT(count=112, emb_dim=512))
    model.load_state_dict(state_dict)

    model.eval()
    return model


if __name__ == "__main__":
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {STATE_PATH}")

    margin_cal = Margin_cal()
    sims, bys = [], []

    _, val_ld = Process(train_num=100, val_num=100, train_size=0.5).loader(
        world_size=1, rank=0, batch_size=1, num_worker=1
    )
    model = load_margin_model(STATE_PATH)

    for bx1, bx2, by in val_ld:
        bx = torch.concat([bx1, bx2], dim=0)
        emb  = model(bx)
        emb1, emb2 = torch.chunk(emb, 2, dim=0)
        sim = F.cosine_similarity(emb1, emb2)
        sims.append(sim.detach().cpu())
        bys.append(by.detach().cpu())
    sims, bys = torch.cat(sims), torch.cat(bys)

    margin_cal.margin(sim=sims, by=bys)
    margin_cal.paint()