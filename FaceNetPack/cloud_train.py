from FaceNetPack.Model.Backbone import CowResNet
from FaceNetPack.Model.ArcFace import ArcFaceLoss, LocalSplitArcFaceLoss, SupConLoss, build_optim
from FaceNetPack.Model.VisionTransformer import ViT
from FaceNetPack.data_processor import Process, dataset as FaceDataset, SingleImageDataset
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from FaceNetPack.Model.MarginModel import Margin_cal
from FaceNetPack.xgboost_verifier import (
    collect_features_and_labels, train_xgboost, evaluate_xgboost,
    print_feature_importance, XGB_STATE_PATH, FEATURE_NAMES,
)
import pickle

all_state = Path(__file__).resolve().parent / "State" / "cloud_model.pth"
log_path = Path(__file__).resolve().parents[1] / "tf-logs" / "cloud_model" #modified
writer = None
LABEL_SMOOTHING = 0.0
VAL_EXTREME_TOPK = (1, 5, 10)
VAL_EXTREME_SAMPLE_SEED = 0


def _select_amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


AMP_DTYPE = _select_amp_dtype()
USE_GRAD_SCALER = AMP_DTYPE == torch.float16
INPUT_MEM_FMT = torch.channels_last if torch.cuda.is_available() else torch.contiguous_format

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")
world_size = 1


def build_pair_eval_loader(source_ld, template_ld):
    eval_ds = FaceDataset(
        source_ld.dataset.persons,
        train=False,
        train_num=len(template_ld.dataset),
        img_size=source_ld.dataset.out_size,
    )
    if isinstance(template_ld.sampler, DistributedSampler):
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=template_ld.sampler.num_replicas,
            rank=template_ld.sampler.rank,
            shuffle=False,
        )
    else:
        eval_sampler = None

    loader_args = {
        "batch_size": template_ld.batch_size,
        "drop_last": True,
        "num_workers": template_ld.num_workers,
        "pin_memory": template_ld.pin_memory,
    }
    if eval_sampler is None:
        loader_args["shuffle"] = False
    else:
        loader_args["sampler"] = eval_sampler
    if template_ld.num_workers > 0:
        loader_args["prefetch_factor"] = template_ld.prefetch_factor
        loader_args["persistent_workers"] = template_ld.persistent_workers
    return DataLoader(eval_ds, **loader_args)


def retrieval_topk_accuracy(embeddings, pids, topk=VAL_EXTREME_TOPK,
                            seed=VAL_EXTREME_SAMPLE_SEED,
                            paths=None, report_k=None):
    """检索式 Recall@K（gallery / query 分离）。

    每个身份随机抽一张图作 gallery，剩余图片作 query。
    对每个 query 计算与所有 gallery 图片的余弦相似度，按降序排序，
    若 top-k 中包含与 query 同身份的 gallery 图片则视为命中。
    最终返回各 k 值的 Recall@K。
    """
    embs = torch.as_tensor(embeddings).detach().cpu().float()
    pids = torch.as_tensor(pids).detach().cpu().long().flatten()
    n = embs.size(0)
    if n < 2:
        return [0.0 for _ in topk], []

    embs = F.normalize(embs, p=2, dim=1)
    if report_k is None:
        report_k = min(topk)

    rng = torch.Generator()
    rng.manual_seed(seed)

    unique_pids = pids.unique()
    gallery_idx_list, query_idx_list = [], []
    for pid in unique_pids:
        indices = (pids == pid).nonzero(as_tuple=True)[0]
        perm = torch.randperm(indices.numel(), generator=rng)
        gallery_idx_list.append(indices[perm[0]].item())
        query_idx_list.extend(indices[perm[1:]].tolist())

    if not query_idx_list:
        return [0.0 for _ in topk], []

    gallery_idx = torch.tensor(gallery_idx_list, dtype=torch.long)
    query_idx = torch.tensor(query_idx_list, dtype=torch.long)

    gallery_embs = embs[gallery_idx]          # (G, dim)
    query_embs = embs[query_idx]              # (Q, dim)
    gallery_pids = pids[gallery_idx]          # (G,)
    query_pids = pids[query_idx]              # (Q,)

    sim_matrix = query_embs @ gallery_embs.t()  # (Q, G)

    _, sorted_gallery = sim_matrix.sort(dim=1, descending=True)
    sorted_pids = gallery_pids[sorted_gallery]  # (Q, G)
    hits = sorted_pids == query_pids.unsqueeze(1)  # (Q, G)

    num_queries = query_idx.numel()
    num_gallery = gallery_idx.numel()
    scores = []
    for k in topk:
        ek = min(int(k), num_gallery)
        recall = hits[:, :ek].any(dim=1).float().sum() / num_queries
        scores.append(recall.item())

    errors = []
    if paths is not None:
        rk = min(int(report_k), num_gallery)
        missed = ~hits[:, :rk].any(dim=1)  # (Q,) 在 top-rk 中未命中的 query
        for qi_local in missed.nonzero(as_tuple=True)[0].tolist():
            qi_global = query_idx[qi_local].item()
            best_wrong_local = sorted_gallery[qi_local, 0].item()
            gi_global = gallery_idx[best_wrong_local].item()
            errors.append({
                "type": "miss",
                "query": paths[qi_global],
                "query_pid": query_pids[qi_local].item(),
                "matched": paths[gi_global],
                "matched_pid": gallery_pids[best_wrong_local].item(),
                "similarity": sim_matrix[qi_local, best_wrong_local].item(),
            })
        errors.sort(key=lambda e: e["similarity"], reverse=True)

    return scores, errors


def _extract_single_embeddings(model, persons, img_size, device, batch_size=64,
                               local_criterion=None, local_weight=0.3):
    """用 SingleImageDataset 提取验证集每张图的 embedding、person_id 和路径。
    
    多卡环境下使用 DistributedSampler 分片，各卡只处理子集后 all_gather 合并。
    """
    ds = SingleImageDataset(persons, img_size=img_size)

    if dist.is_initialized():
        sampler = DistributedSampler(ds, shuffle=False)
    else:
        sampler = None

    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                    sampler=sampler,
                    num_workers=4, pin_memory=torch.cuda.is_available())
    local_module = None
    if local_criterion is not None:
        local_module = local_criterion.module if isinstance(local_criterion, DDP) else local_criterion

    all_embs, all_pids, local_indices = [], [], []
    for bx, bpid in ld:
        bx = bx.to(device, non_blocking=True, memory_format=INPUT_MEM_FMT)
        if local_module is None:
            emb = model(bx)
        else:
            global_emb, local_feat = model(bx, return_local=True)
            global_emb = F.normalize(global_emb, p=2, dim=1)

            local_embs = []
            for i, chunk in enumerate(torch.chunk(local_feat, local_module.splits, dim=2)):
                le = chunk.mean(dim=[2, 3])
                le = local_module.bns[i](le)
                le = F.normalize(le, p=2, dim=1)
                local_embs.append(le)
            local_emb_avg = torch.stack(local_embs, dim=0).mean(dim=0)
            emb = (1.0 - local_weight) * global_emb + local_weight * local_emb_avg

        all_embs.append(emb.detach().cpu())
        all_pids.append(bpid)

    all_embs = torch.cat(all_embs)
    all_pids = torch.cat(all_pids)

    if dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, {"emb": all_embs, "pid": all_pids})
        all_embs = torch.cat([g["emb"] for g in gathered])
        all_pids = torch.cat([g["pid"] for g in gathered])

    return all_embs, all_pids, ds.paths


@torch.no_grad()
def evaluate(model:nn.Module, val_ld, device, split="val", local_criterion=None,
             local_weight=0.3, calc_extreme_topk=False,
             extreme_topk=VAL_EXTREME_TOPK):
    model.eval()
    local_module = None
    local_training = False
    if local_criterion is not None:
        local_module = local_criterion.module if isinstance(local_criterion, DDP) else local_criterion
        local_training = local_criterion.training
        local_criterion.eval()
    margin_cal = Margin_cal()
    sims, labels = [], []
    acc = torch.zeros(1, device=device)
    margin = torch.zeros(1, device=device)
    extreme_topk_tensor = torch.zeros(len(extreme_topk), dtype=torch.float32, device=device)

    for bx1, bx2, by in val_ld:
        bx1 = bx1.to(device, non_blocking=True, memory_format=INPUT_MEM_FMT)
        bx2 = bx2.to(device, non_blocking=True, memory_format=INPUT_MEM_FMT)
        by = by.to(device, non_blocking=True)
        bx = torch.concat([bx1, bx2], dim=0)
        if local_module is None:
            emb = model(bx)
            emb1, emb2 = torch.chunk(emb, 2, dim=0)
            sim = F.cosine_similarity(emb1, emb2)
        else:
            global_emb, local_feat = model(bx, return_local=True)
            global_emb = F.normalize(global_emb, p=2, dim=1)
            emb1, emb2 = torch.chunk(global_emb, 2, dim=0)
            sim_global = F.cosine_similarity(emb1, emb2)

            local_sims = []
            for i, chunk in enumerate(torch.chunk(local_feat, local_module.splits, dim=2)):
                local_emb = chunk.mean(dim=[2, 3])
                local_emb = local_module.bns[i](local_emb)
                local_emb = F.normalize(local_emb, p=2, dim=1)
                local_emb1, local_emb2 = torch.chunk(local_emb, 2, dim=0)
                local_sims.append(F.cosine_similarity(local_emb1, local_emb2))
            sim_local = torch.stack(local_sims, dim=0).mean(dim=0)
            sim = (1.0 - local_weight) * sim_global + local_weight * sim_local

        sims.append(sim.detach().cpu())
        labels.append(by.detach().cpu())
    sims, labels = torch.cat(sims), torch.cat(labels)

    if dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, {"sim": sims, "label": labels})
        if is_main_process():
            sims = torch.cat([g["sim"] for g in gathered])
            labels = torch.cat([g["label"] for g in gathered])

    all_paths = None
    if calc_extreme_topk:
        persons = val_ld.dataset.persons
        img_size = val_ld.dataset.out_size
        all_embs, all_pids, all_paths = _extract_single_embeddings(
            model, persons, img_size, device,
            batch_size=val_ld.batch_size,
            local_criterion=local_criterion, local_weight=local_weight,
        )

    if is_main_process():
        margin_value = margin_cal.margin(sim=sims, by=labels)
        pred = sims >= margin_value
        label_mask = labels == 1
        acc = torch.tensor((pred == label_mask).float().mean().item(), device=device)
        margin = torch.tensor(margin_value, dtype=torch.float32, device=device)
        if calc_extreme_topk:
            extreme_scores, errors = retrieval_topk_accuracy(
                all_embs, all_pids,
                topk=extreme_topk,
                paths=all_paths,
            )
            extreme_topk_tensor = torch.tensor(extreme_scores, dtype=torch.float32, device=device)

        pos_sim = sims[label_mask]
        neg_sim = sims[~label_mask]
        print(f"[{split}] True:")
        if len(pos_sim) > 0:
            print(pos_sim.min().item(), pos_sim.max().item(), pos_sim.mean().item())
        else:
            print("no positive pairs")
        print(f"[{split}] False:")
        if len(neg_sim) > 0:
            print(neg_sim.min().item(), neg_sim.max().item(), neg_sim.mean().item())
        else:
            print("no negative pairs")
        print(f"[{split}] margin: {margin_value}")
        if calc_extreme_topk:
            topk_msg = ", ".join(
                f"top{k}: {score * 100:.2f}%"
                for k, score in zip(extreme_topk, extreme_topk_tensor.tolist())
            )
            print(f"[{split}] retrieval {topk_msg}")
            if errors:
                print(f"[{split}] retrieval misses: {len(errors)}")
                for e in errors[:10]:
                    print(f"  MISS sim={e['similarity']:.4f} | query(pid={e['query_pid']}): {e['query']}")
                    print(f"       top1(pid={e['matched_pid']}): {e['matched']}")
    if dist.is_initialized():
        dist.broadcast(acc, src=0)
        dist.broadcast(margin, src=0)
        dist.broadcast(extreme_topk_tensor, src=0)
    if local_criterion is not None and local_training:
        local_criterion.train()
    return acc.item(), margin.item(), tuple(extreme_topk_tensor.cpu().tolist())


class FaceSequential(nn.Sequential):
    def forward(self, input, return_local=False):
        for module in self:
            if isinstance(module, ViT):
                input = module(input, return_local=return_local)
            else:
                input = module(input)
        return input

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(model:nn.Module, train_ld, val_ld, device, criterion, local_criterion, supcon_criterion=None, supcon_weight=0.5, lr=2e-3, epochs=30, weight_decay=1e-4, resum=False):
    criterion_module = criterion.module if isinstance(criterion, DDP) else criterion
    local_criterion_module = local_criterion.module if isinstance(local_criterion, DDP) else local_criterion
    optim_modules = [model, criterion_module, local_criterion_module]
    if supcon_criterion is not None:
        sc_module = supcon_criterion.module if isinstance(supcon_criterion, DDP) else supcon_criterion
        optim_modules.append(sc_module)
    optimizer = build_optim(nn.ModuleList(optim_modules), lr, weight_decay)
    train_eval_ld = build_pair_eval_loader(train_ld, val_ld)
    
    warmup_epochs = 4
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3, total_iters=warmup_epochs)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=0.001,
        min_lr=5e-6,
    )
    best_acc = 0.0
    start_epoch = 1

    if resum and all_state.exists():
        ckpt = torch.load(all_state, map_location='cpu')
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler_state = ckpt.get("scheduler")
        if isinstance(scheduler_state, dict) and "warmup" in scheduler_state and "plateau" in scheduler_state:
            warmup_scheduler.load_state_dict(scheduler_state["warmup"])
            plateau_scheduler.load_state_dict(scheduler_state["plateau"])
        model.module.load_state_dict(ckpt["model"])
        if "criterion" in ckpt:
            criterion_module.load_state_dict(ckpt["criterion"])
        if "local_criterion" in ckpt:
            local_criterion_module.load_state_dict(ckpt["local_criterion"])
        best_acc = ckpt["best_acc"]
        start_epoch = ckpt["epoch"]

    for p in model.parameters():
        dist.broadcast(p.data, src=0) if dist.is_initialized() else None

    scaler = GradScaler("cuda", enabled=USE_GRAD_SCALER)
    # 收集 clip 用的参数列表只构建一次，避免每个 step 重建 list
    clip_params = list(model.parameters()) + list(criterion.parameters()) + list(local_criterion.parameters())
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if isinstance(train_ld.sampler, DistributedSampler):
            train_ld.sampler.set_epoch(epoch)
        if hasattr(train_ld.dataset, 'set_epoch'):
            train_ld.dataset.set_epoch(epoch)

        # device tensor 累加 [loss, top1, top10, top100, supcon]，避免 .item() 触发 H2D/D2H 同步
        metric_sum = torch.zeros(5, device=device)

        for bx, by in tqdm(train_ld, disable=not is_main_process()):
            bx = bx.to(device, non_blocking=True, memory_format=INPUT_MEM_FMT)
            by = by.to(device, non_blocking=True)
            with autocast("cuda", dtype=AMP_DTYPE):
                global_emb, local_feat = model(bx, return_local=True)
                loss_global, logits = criterion(global_emb, by)
                loss_local = local_criterion(local_feat, by)
                loss = loss_global + 0.6 * loss_local

                if supcon_criterion is not None:
                    loss_supcon = supcon_criterion(global_emb, by)
                    loss = loss + supcon_weight * loss_supcon
                    metric_sum[4] += loss_supcon.detach().float()

            with torch.no_grad():
                acc1, acc10, acc100 = accuracy(logits, by, topk=(1, 10, 100))
                metric_sum[0] += loss.detach().float()
                metric_sum[1] += acc1.squeeze().detach().float()
                metric_sum[2] += acc10.squeeze().detach().float()
                metric_sum[3] += acc100.squeeze().detach().float()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # 在进行梯度裁剪前必须先 unscale，否则裁剪的是放大后的梯度（bf16 时为 no-op）
            scaler.unscale_(optimizer)

            # 在训练前期极易发生梯度爆炸导致网络摆烂，加上梯度裁剪
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

        if dist.is_initialized():
            dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
            metric_sum /= world_size

        loss_avg, top1_avg, top10_avg, top100_avg, supcon_avg = (metric_sum / len(train_ld)).tolist()

        train_acc, _, _ = evaluate(model, train_eval_ld, device, split="train", local_criterion=local_criterion)
        acc, margin, val_topk = evaluate(
            model,
            val_ld,
            device,
            split="val",
            local_criterion=local_criterion,
            calc_extreme_topk=True,
        )
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(acc)

        if is_main_process() and acc > best_acc:
            best_acc = acc
            all_state.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"optimizer": optimizer.state_dict(), 
                    "scheduler": {
                        "warmup": warmup_scheduler.state_dict(),
                        "plateau": plateau_scheduler.state_dict(),
                    }, "model": model.module.state_dict(),
                "criterion": criterion_module.state_dict(),
                "local_criterion": local_criterion_module.state_dict(),
                "best_acc": best_acc, "epoch": epoch + 1, "margin": margin}, all_state)
        if is_main_process():
            writer.add_scalar("lr/epoch", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("loss/epoch", loss_avg, epoch)
            if supcon_criterion is not None:
                writer.add_scalar("supcon_loss/epoch", supcon_avg, epoch)
            writer.add_scalar("train_acc/epoch", train_acc, epoch)
            writer.add_scalar("train_top1/epoch", top1_avg, epoch)
            writer.add_scalar("train_top10/epoch", top10_avg, epoch)
            writer.add_scalar("train_top100/epoch", top100_avg, epoch)
            writer.add_scalar("acc/epoch", acc, epoch)
            for k, score in zip(VAL_EXTREME_TOPK, val_topk):
                writer.add_scalar(f"val_top{k}/epoch", score * 100.0, epoch)
        
        if is_main_process() and epoch % max(1, epochs // 200) == 0:
            val_topk_msg = ", ".join(
                f"val_top{k}: {score * 100:.2f}%"
                for k, score in zip(VAL_EXTREME_TOPK, val_topk)
            )
            supcon_msg = f", supcon: {supcon_avg:.4f}" if supcon_criterion is not None else ""
            print(f"  ========== epoch: {epoch} ==========\nloss: {loss_avg:.4f}{supcon_msg}, train_acc: {train_acc:.4f}, val_acc: {acc:.4f}\n"
                  f"train_top1: {top1_avg:.2f}%, train_top10: {top10_avg:.2f}%, train_top100: {top100_avg:.2f}%\n"
                  f"{val_topk_msg}\n")

        if is_main_process() and epoch % 5 == 0:
            try:
                print(f"[xgb] Training XGBoost at epoch {epoch}...")
                X_train_xgb, y_train_xgb = collect_features_and_labels(
                    model, local_criterion, train_eval_ld, device)
                X_val_xgb, y_val_xgb = collect_features_and_labels(
                    model, local_criterion, val_ld, device)
                bst = train_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
                print("[xgb] Val evaluation:")
                evaluate_xgboost(bst, X_val_xgb, y_val_xgb)
                print_feature_importance(bst)
                XGB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(XGB_STATE_PATH, "wb") as f:
                    pickle.dump(bst, f)
                if writer is not None:
                    import xgboost as xgb
                    dval = xgb.DMatrix(X_val_xgb, feature_names=FEATURE_NAMES[:X_val_xgb.shape[1]])
                    xgb_preds = (bst.predict(dval) >= 0.5).astype(int)
                    xgb_acc = (xgb_preds == y_val_xgb).mean()
                    writer.add_scalar("xgb_val_acc/epoch", xgb_acc, epoch)
            except Exception as e:
                print(f"[xgb] Skipped: {e}")

    return True
    

if __name__ == "__main__":
    local_rank = setup_ddp()
    if dist.is_initialized(): world_size = dist.get_world_size()
    torch.backends.cudnn.benchmark = True
    # 开启 TF32 让 Ampere+ 的 fp32 matmul/conv 走 TensorCore，单步更快、对精度几乎无影响
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if is_main_process(): writer = SummaryWriter(log_dir=log_path)

    data_process = Process(train_num=32000, val_num=1600, device=device)
    train_ld, val_ld = data_process.loader(
        world_size=world_size, rank=local_rank, batch_size=320,
        num_worker=16, prefetch_factor=2
    )

    model = FaceSequential(CowResNet(), ViT(count=112, emb_dim=512)).to(device)
    if torch.cuda.is_available():
        # CNN backbone 在 NHWC + AMP 下吞吐显著高于 NCHW
        model = model.to(memory_format=torch.channels_last)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_kwargs = dict(find_unused_parameters=False, gradient_as_bucket_view=True)
    if torch.cuda.is_available():
        ddp_kwargs["device_ids"] = [local_rank]
    model = DDP(model, **ddp_kwargs)

    criterion = ArcFaceLoss(512, data_process.num_train, label_smoothing=LABEL_SMOOTHING).to(device)
    criterion = DDP(criterion, **ddp_kwargs)

    local_criterion = LocalSplitArcFaceLoss(
        512, data_process.num_train, splits=2, label_smoothing=LABEL_SMOOTHING
    ).to(device)
    local_criterion = DDP(local_criterion, **ddp_kwargs)

    supcon_criterion = SupConLoss(temperature=0.07).to(device)

    train(model, train_ld, val_ld, device, criterion, local_criterion,
          supcon_criterion=supcon_criterion, supcon_weight=0.5, resum=False)

    if dist.is_initialized(): dist.destroy_process_group()




