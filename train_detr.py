import argparse
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rotated_box_to_xyxy(cx: float, cy: float, w: float, h: float, angle: float):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2.0
    dy = h / 2.0

    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy),
    ]

    points = []
    for x, y in corners:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        points.append((rx, ry))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    return x1, y1, x2, y2


class CowDetrDataset(Dataset):
    def __init__(self, image_paths, label_dir: Path, processor):
        self.image_paths = list(image_paths)
        self.label_dir = Path(label_dir)
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def _parse_xml(self, xml_path: Path, width: int, height: int):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = []

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            robndbox = obj.find("robndbox")

            if robndbox is not None:
                cx = float(robndbox.findtext("cx", default="0"))
                cy = float(robndbox.findtext("cy", default="0"))
                w = float(robndbox.findtext("w", default="0"))
                h = float(robndbox.findtext("h", default="0"))
                angle = float(robndbox.findtext("angle", default="0"))
                x1, y1, x2, y2 = rotated_box_to_xyxy(cx, cy, w, h, angle)
            elif bndbox is not None:
                x1 = float(bndbox.findtext("xmin", default="0"))
                y1 = float(bndbox.findtext("ymin", default="0"))
                x2 = float(bndbox.findtext("xmax", default="0"))
                y2 = float(bndbox.findtext("ymax", default="0"))
            else:
                continue

            x1, y1, x2, y2 = clip_xyxy(x1, y1, x2, y2, width, height)
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w < 1.0 or box_h < 1.0:
                continue

            annotations.append(
                {
                    "bbox": [x1, y1, box_w, box_h],
                    "category_id": 0,
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )

        return annotations

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        xml_path = self.label_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"未找到标注文件: {xml_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        annotations = self._parse_xml(xml_path, width, height)
        target = {"image_id": idx, "annotations": annotations}

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def build_datasets(image_dir: Path, label_dir: Path, processor, val_ratio: float, seed: int):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(sorted(image_dir.glob(pattern)))

    image_paths = [p for p in sorted(image_paths) if (label_dir / f"{p.stem}.xml").exists()]
    if not image_paths:
        raise RuntimeError("没有找到可用的图像/标注配对数据。")

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    val_size = max(1, int(len(image_paths) * val_ratio))
    if val_size >= len(image_paths):
        val_size = max(1, len(image_paths) - 1)

    val_paths = image_paths[:val_size]
    train_paths = image_paths[val_size:]
    if not train_paths:
        raise RuntimeError("训练集为空，请减少 val_ratio 或增加数据量。")

    train_dataset = CowDetrDataset(train_paths, label_dir, processor)
    val_dataset = CowDetrDataset(val_paths, label_dir, processor)
    return train_dataset, val_dataset


def make_collate_fn(processor):
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    return collate_fn


def move_labels_to_device(labels, device):
    moved = []
    for target in labels:
        moved.append({k: v.to(device) for k, v in target.items()})
    return moved


def train_one_epoch(model, dataloader, optimizer, device, scaler=None, amp_dtype=torch.float16, use_channels_last=False):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        if use_channels_last:
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        loss_val = loss.detach().item()
        total_loss += loss_val
        progress.set_postfix(loss=f"{loss_val:.4f}", step=f"{step}/{len(dataloader)}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="eval", leave=False)
    for step, batch in enumerate(progress, start=1):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = move_labels_to_device(batch["labels"], device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        batch_loss = outputs.loss.detach().item()
        total_loss += batch_loss
        progress.set_postfix(loss=f"{batch_loss:.4f}", step=f"{step}/{len(dataloader)}")

    return total_loss / len(dataloader)


def maybe_freeze_backbone(model):
    for name, param in model.named_parameters():
        if "backbone" in name or "model.backbone" in name:
            param.requires_grad = False


def parse_args():
    parser = argparse.ArgumentParser(description="微调 DETR")
    parser.add_argument("--model-path", type=str, default="./detr-resnet-50")
    parser.add_argument("--image-dir", type=str, default="./data/yolo/images")
    parser.add_argument("--label-dir", type=str, default="./data/yolo/labels")
    parser.add_argument("--output-dir", type=str, default="./detr-cow-finetuned")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float16

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id2label = {0: "cow"}
    label2id = {"cow": 0}

    # 当前 transformers 版本下，fast processor 的 pad 接口与旧版不同；
    # 这里固定使用 slow processor，避免 collate_fn 兼容问题。
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if args.freeze_backbone:
        maybe_freeze_backbone(model)

    model.to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    train_dataset, val_dataset = build_datasets(
        image_dir=Path(args.image_dir),
        label_dir=Path(args.label_dir),
        processor=processor,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    collate_fn = make_collate_fn(processor)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # bf16 不需要 GradScaler，但仍然走 autocast；用 enabled 控制 fp16 时才生效
    use_grad_scaler = use_cuda and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler) if use_cuda else None
    best_val_loss = float("inf")

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"output dir: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler,
                                     amp_dtype=amp_dtype, use_channels_last=use_cuda)
        val_loss = evaluate(model, val_loader, device)

        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"saved best model to {output_dir} | best_val_loss={best_val_loss:.4f}")

    print("training finished")


if __name__ == "__main__":
    main()
