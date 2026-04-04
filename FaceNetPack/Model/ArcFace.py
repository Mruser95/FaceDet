import torch
import torch.nn as nn
import torch.nn.functional as F


def build_optim(model:nn.Module, lr, weight_decay):
    decay, undecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim == 1 or "bn" in name.lower():
            undecay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay}, 
        {"params": undecay, "weight_decay": 0.0}
    ], lr=lr)
    return optimizer


class ArcFaceLoss(nn.Module):
    def __init__(self, emb_dim, classes, margin=0.6, scale=64, label_smoothing=0.0):
        super().__init__()
        self.classes = classes
        self.m = margin
        self.s = scale
        self.label_smoothing = label_smoothing
        self.W = nn.Parameter(torch.randn(classes, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb:torch.Tensor, label:torch.Tensor):
        with torch.amp.autocast("cuda", enabled=False):
            emb = emb.float()
            emb_norm, W_norm = F.normalize(emb, p=2, dim=1), F.normalize(self.W, p=2, dim=1)
            cos = F.linear(emb_norm, W_norm)
            cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)

            # 避免使用 acos，采用官方 ArcFace 余弦展开公式以保证数值稳定
            import math
            cos_m = math.cos(self.m)
            sin_m = math.sin(self.m)
            th = math.cos(math.pi - self.m)
            mm = math.sin(math.pi - self.m) * self.m
            
            sine = torch.sqrt(1.0 - torch.pow(cos, 2)).clamp(min=1e-7)
            phi = cos * cos_m - sine * sin_m
            target_logits = torch.where(cos > th, phi, cos - mm)
            
            one_hot = F.one_hot(label.long(), num_classes=self.classes).float().to(emb.device)

            logits = cos * (1.0 - one_hot) + target_logits * one_hot
            logits = logits * self.s
            loss = F.cross_entropy(logits, label.long(), label_smoothing=self.label_smoothing)
            
            # 预测用的原始 logits，不加 margin 惩罚
            original_logits = cos * self.s
        return loss, original_logits

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)
    
    在同一 batch 内，将相同 label 的 embedding 拉近，不同 label 的推远。
    与 ArcFace 的分类信号互补：ArcFace 让 embedding 靠近各自的类中心，
    SupCon 则直接优化样本间的相对距离结构。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, label: torch.Tensor):
        with torch.amp.autocast("cuda", enabled=False):
            emb = F.normalize(emb.float(), p=2, dim=1)
            B = emb.size(0)
            
            sim = emb @ emb.t() / self.temperature          # [B, B]
            
            label = label.long().view(-1)
            mask_pos = label.unsqueeze(0) == label.unsqueeze(1)  # [B, B]
            mask_self = torch.eye(B, dtype=torch.bool, device=emb.device)
            mask_pos = mask_pos & ~mask_self                     # 排除自身

            num_pos = mask_pos.float().sum(dim=1)                # 每个样本的正例数
            has_pos = num_pos > 0

            sim_max = sim.detach().max(dim=1, keepdim=True).values
            logits = sim - sim_max                               # 数值稳定
            exp_logits = torch.exp(logits)

            # 分母：除自身外所有样本的 exp 之和
            denom = (exp_logits * ~mask_self).sum(dim=1, keepdim=True)
            log_prob = logits - torch.log(denom.clamp(min=1e-12))

            # 对每个样本取所有正例的平均 log 概率
            mean_log_prob = (mask_pos.float() * log_prob).sum(dim=1)
            mean_log_prob = mean_log_prob / num_pos.clamp(min=1.0)

            loss = -mean_log_prob[has_pos].mean()
            return loss


class LocalSplitArcFaceLoss(nn.Module):
    def __init__(self, emb_dim, classes, splits=3, margin=0.5, scale=64, label_smoothing=0.0):
        """
        局部特征竖向拆分ArcFace损失计算
        """
        super().__init__()
        self.splits = splits
        
        # 为每个拆分块创建独立的ArcFaceLoss和BatchNorm
        self.loss_funcs = nn.ModuleList([
            ArcFaceLoss(emb_dim, classes, margin, scale, label_smoothing) for _ in range(splits)
        ])
        # 加入 BatchNorm1d 防止局部特征塌缩
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(emb_dim) for _ in range(splits)
        ])

    def forward(self, local_feat: torch.Tensor, label: torch.Tensor):
        """
        local_feat: [B, C, H, W] 恢复后的局部特征图
        label: [B]
        """
        # 沿着高度(H)方向竖向拆分为指定份数
        chunks = torch.chunk(local_feat, self.splits, dim=2)
        
        total_loss = 0.0
        for i, chunk in enumerate(chunks):
            # 对每个局部块进行空间全局平均池化得到特征向量: [B, C]
            chunk_emb = chunk.mean(dim=[2, 3])
            # 对局部特征进行 BN 归一化
            chunk_emb = self.bns[i](chunk_emb)
            
            # 分别计算ArcFace损失并累加
            loss, _ = self.loss_funcs[i](chunk_emb, label)
            total_loss += loss
            
        return total_loss / self.splits  # 平均损失，防止局部损失过大导致梯度爆炸