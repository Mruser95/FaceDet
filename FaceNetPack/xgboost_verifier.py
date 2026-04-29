"""
XGBoost 验证器：在神经网络 embedding 之上训练 XGBoost，
利用多维特征（全局/局部余弦相似度、L2距离、元素级统计量等）
学习非线性决策面，提升验证准确率。

用法:
    # 训练完神经网络后，提取特征并训练 XGBoost
    python -m FaceNetPack.xgboost_verifier --train

    # 加载已训练的 XGBoost 进行推理
    python -m FaceNetPack.xgboost_verifier --eval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse

from FaceNetPack.Model.Backbone import CowResNet
from FaceNetPack.Model.VisionTransformer import ViT
from FaceNetPack.Model.ArcFace import LocalSplitArcFaceLoss
from FaceNetPack.data_processor import Process, dataset as FaceDataset
STATE_PATH = Path(__file__).resolve().parent / "State" / "cloud_model.pth"
XGB_STATE_PATH = Path(__file__).resolve().parent / "State" / "xgb_verifier.pkl"


def load_model_and_local_criterion(state_path=STATE_PATH, device="cpu"):
    from FaceNetPack.cloud_train import FaceSequential

    ckpt = torch.load(state_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = FaceSequential(CowResNet(), ViT(count=112, emb_dim=512))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    local_criterion = None
    if isinstance(ckpt, dict) and "local_criterion" in ckpt:
        num_train = ckpt["local_criterion"]["loss_funcs.0.W"].shape[0]
        local_criterion = LocalSplitArcFaceLoss(512, num_train, splits=2)
        local_criterion.load_state_dict(ckpt["local_criterion"])
        local_criterion = local_criterion.to(device).eval()

    return model, local_criterion


def extract_pair_features(emb1, emb2, local_feat1=None, local_feat2=None,
                          local_module=None):
    """从一对 embedding 提取丰富的特征向量用于 XGBoost。

    返回 shape: [B, num_features] 的 numpy array。
    """
    emb1_n = F.normalize(emb1, p=2, dim=1)
    emb2_n = F.normalize(emb2, p=2, dim=1)

    cos_sim = F.cosine_similarity(emb1_n, emb2_n)                   # [B]
    l2_dist = (emb1_n - emb2_n).norm(p=2, dim=1)                    # [B]

    diff = (emb1_n - emb2_n).abs()                                  # [B, D]
    diff_mean = diff.mean(dim=1)
    diff_std = diff.std(dim=1)
    diff_max = diff.max(dim=1).values
    diff_q75 = diff.quantile(0.75, dim=1)

    prod = emb1_n * emb2_n                                          # [B, D]
    prod_mean = prod.mean(dim=1)
    prod_std = prod.std(dim=1)
    prod_min = prod.min(dim=1).values

    feats = [cos_sim, l2_dist,
             diff_mean, diff_std, diff_max, diff_q75,
             prod_mean, prod_std, prod_min]

    if local_module is not None and local_feat1 is not None and local_feat2 is not None:
        splits = local_module.splits
        chunks1 = torch.chunk(local_feat1, splits, dim=2)
        chunks2 = torch.chunk(local_feat2, splits, dim=2)
        for i in range(splits):
            le1 = F.normalize(local_module.bns[i](chunks1[i].mean(dim=[2, 3])), p=2, dim=1)
            le2 = F.normalize(local_module.bns[i](chunks2[i].mean(dim=[2, 3])), p=2, dim=1)
            local_cos = F.cosine_similarity(le1, le2)
            local_l2 = (le1 - le2).norm(p=2, dim=1)
            feats.extend([local_cos, local_l2])

    # 留在 GPU 上返回，由 caller 决定何时一次性同步到 CPU
    return torch.stack(feats, dim=1).detach()


FEATURE_NAMES = [
    "cos_sim", "l2_dist",
    "diff_mean", "diff_std", "diff_max", "diff_q75",
    "prod_mean", "prod_std", "prod_min",
    "local0_cos", "local0_l2",
    "local1_cos", "local1_l2",
]


@torch.no_grad()
def collect_features_and_labels(model, local_criterion, dataloader, device):
    """遍历 pair dataloader，收集所有特征和标签。"""
    model.eval()
    if local_criterion is not None:
        local_criterion.eval()

    local_module = None
    if local_criterion is not None:
        local_module = (local_criterion.module
                        if hasattr(local_criterion, "module") else local_criterion)

    all_feats, all_labels = [], []
    for bx1, bx2, by in tqdm(dataloader, desc="extracting features"):
        bx1 = bx1.to(device, non_blocking=True)
        bx2 = bx2.to(device, non_blocking=True)
        bx = torch.cat([bx1, bx2], dim=0)

        if local_module is not None:
            global_emb, local_feat = model(bx, return_local=True)
            emb1, emb2 = torch.chunk(global_emb, 2, dim=0)
            lf1, lf2 = torch.chunk(local_feat, 2, dim=0)
            feats = extract_pair_features(emb1, emb2, lf1, lf2, local_module)
        else:
            emb = model(bx)
            emb1, emb2 = torch.chunk(emb, 2, dim=0)
            feats = extract_pair_features(emb1, emb2)

        all_feats.append(feats)
        all_labels.append(by)

    # 单次 cat + 一次性同步到 CPU，避免每个 batch 都触发 GPU→CPU 拷贝同步
    X = torch.cat(all_feats, dim=0).cpu().numpy()
    y = torch.cat(all_labels, dim=0).numpy().astype(np.int32)
    return X, y


def train_xgboost(X_train, y_train, X_val, y_val):
    """训练 XGBoost 分类器。"""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES[:X_train.shape[1]])
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES[:X_val.shape[1]])

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["error", "auc", "logloss"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "seed": 42,
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=25,
    )

    return bst


def evaluate_xgboost(bst, X, y):
    """评估 XGBoost 分类器并打印详细指标。"""
    import xgboost as xgb

    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES[:X.shape[1]])
    probs = bst.predict(dmat)
    preds = (probs >= 0.5).astype(np.int32)

    acc = (preds == y).mean()
    mask_pos = y == 1
    mask_neg = y == 0

    tp = (preds[mask_pos] == 1).sum()
    fn = (preds[mask_pos] == 0).sum()
    fp = (preds[mask_neg] == 1).sum()
    tn = (preds[mask_neg] == 0).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    if mask_pos.sum() > 0:
        print(f"  True  prob: min={probs[mask_pos].min():.4f}, "
              f"max={probs[mask_pos].max():.4f}, mean={probs[mask_pos].mean():.4f}")
    if mask_neg.sum() > 0:
        print(f"  False prob: min={probs[mask_neg].min():.4f}, "
              f"max={probs[mask_neg].max():.4f}, mean={probs[mask_neg].mean():.4f}")

    return acc


def print_feature_importance(bst):
    """打印特征重要性排名。"""
    importance = bst.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importance (gain):")
    for fname, score in sorted_imp:
        print(f"    {fname:18s} {score:.2f}")


class XGBVerifier:
    """封装好的 XGBoost 验证器，可在推理时直接调用。"""

    def __init__(self, model, local_criterion=None, xgb_path=XGB_STATE_PATH, device="cpu"):
        self.model = model.eval()
        self.local_criterion = local_criterion
        self.device = device
        if local_criterion is not None:
            local_criterion.eval()
        self.local_module = None
        if local_criterion is not None:
            self.local_module = (local_criterion.module
                                 if hasattr(local_criterion, "module") else local_criterion)

        with open(xgb_path, "rb") as f:
            self.bst = pickle.load(f)

    @torch.no_grad()
    def verify(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple:
        """
        输入: 两个 [B, 4, H, W] 的图像 tensor（已预处理）
        返回: (probs, preds)  probs: [B] 概率, preds: [B] 0/1 预测
        """
        import xgboost as xgb

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        bx = torch.cat([img1, img2], dim=0)

        if self.local_module is not None:
            global_emb, local_feat = self.model(bx, return_local=True)
            emb1, emb2 = torch.chunk(global_emb, 2, dim=0)
            lf1, lf2 = torch.chunk(local_feat, 2, dim=0)
            feats = extract_pair_features(emb1, emb2, lf1, lf2, self.local_module)
        else:
            emb = self.model(bx)
            emb1, emb2 = torch.chunk(emb, 2, dim=0)
            feats = extract_pair_features(emb1, emb2)

        feats_np = feats.cpu().numpy()
        dmat = xgb.DMatrix(feats_np, feature_names=FEATURE_NAMES[:feats_np.shape[1]])
        probs = self.bst.predict(dmat)
        preds = (probs >= 0.5).astype(np.int32)
        return probs, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="训练 XGBoost")
    parser.add_argument("--eval", action="store_true", help="仅评估已训练的 XGBoost")
    parser.add_argument("--train-pairs", type=int, default=20000, help="训练对数量")
    parser.add_argument("--val-pairs", type=int, default=5000, help="验证对数量")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.train and not args.eval:
        args.train = True

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[xgb] Device: {device}")

    print("[xgb] Loading neural network model...")
    model, local_criterion = load_model_and_local_criterion(STATE_PATH, device)

    print("[xgb] Loading dataset...")
    data_process = Process(train_num=args.train_pairs, val_num=args.val_pairs, device=device)
    train_ds = FaceDataset(data_process.train_ps, train=False,
                           train_num=args.train_pairs, img_size=data_process.img_size)
    val_ds = FaceDataset(data_process.val_ps, train=False,
                         train_num=args.val_pairs, img_size=data_process.img_size)

    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available())
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available())

    print("[xgb] Extracting train features...")
    X_train, y_train = collect_features_and_labels(model, local_criterion, train_ld, device)
    print(f"  Train: {X_train.shape}, pos={y_train.sum()}, neg={(1-y_train).sum()}")

    print("[xgb] Extracting val features...")
    X_val, y_val = collect_features_and_labels(model, local_criterion, val_ld, device)
    print(f"  Val:   {X_val.shape}, pos={y_val.sum()}, neg={(1-y_val).sum()}")

    if args.train:
        print("\n[xgb] ===== Baseline: cosine threshold =====")
        from FaceNetPack.Model.MarginModel import Margin_cal
        mc = Margin_cal()
        train_sims = torch.tensor(X_train[:, 0])
        val_sims = torch.tensor(X_val[:, 0])

        margin = mc.margin(sim=train_sims, by=torch.tensor(y_train))
        print(f"  Best margin (train): {margin:.4f}")
        baseline_pred = (X_val[:, 0] >= margin).astype(np.int32)
        baseline_acc = (baseline_pred == y_val).mean()
        print(f"  Baseline val accuracy: {baseline_acc:.4f}")

        print("\n[xgb] ===== Training XGBoost =====")
        bst = train_xgboost(X_train, y_train, X_val, y_val)

        print("\n[xgb] ===== Train set evaluation =====")
        evaluate_xgboost(bst, X_train, y_train)

        print("\n[xgb] ===== Val set evaluation =====")
        xgb_acc = evaluate_xgboost(bst, X_val, y_val)

        print_feature_importance(bst)

        print(f"\n[xgb] Improvement: {baseline_acc:.4f} -> {xgb_acc:.4f} "
              f"({(xgb_acc - baseline_acc)*100:+.2f}%)")

        XGB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(XGB_STATE_PATH, "wb") as f:
            pickle.dump(bst, f)
        print(f"[xgb] Model saved to {XGB_STATE_PATH}")

    elif args.eval:
        if not XGB_STATE_PATH.exists():
            print(f"[xgb] ERROR: {XGB_STATE_PATH} not found. Run --train first.")
            return
        with open(XGB_STATE_PATH, "rb") as f:
            bst = pickle.load(f)

        print("\n[xgb] ===== Val set evaluation =====")
        evaluate_xgboost(bst, X_val, y_val)
        print_feature_importance(bst)


if __name__ == "__main__":
    main()
