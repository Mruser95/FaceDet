"""
离线人脸检测 & 裁切脚本。
使用 DETR 检测 color 图中的人脸，然后用相同的裁切框同时裁切 color 和 depth 图片。
结果保存到 Dataset/325_crop/，保持原始目录结构。
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


_project_root = Path(__file__).resolve().parents[1]
_default_src = Path(__file__).resolve().parent / "Dataset" / "325"
_default_dst = Path(__file__).resolve().parent / "Dataset" / "325_crop"


def build_detector(device):
    model_path = _project_root / "detr-cow-finetuned"
    if not model_path.exists():
        model_path = _project_root / "detr-resnet-50"
    print(f"Loading DETR from {model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(model_path).to(device).eval()
    return processor, model


def read_dataset(root):
    num_re = re.compile(r"(\d+)$")

    def extract_id(stem: str):
        m = num_re.search(stem)
        return int(m.group(1)) if m else None

    pairs = []
    persons = sorted(root.glob("*"))
    for sub in persons:
        colroot, deproot = sub / "color", sub / "depth"
        if not (colroot.exists() and deproot.exists()):
            continue
        color = {extract_id(p.stem): p for p in colroot.glob("*")}
        depth = {extract_id(p.stem): p for p in deproot.glob("*")}
        color.pop(None, None)
        depth.pop(None, None)
        for fid in sorted(set(color) & set(depth)):
            pairs.append((sub.name, color[fid], depth[fid]))
    return pairs


class FaceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person, color_path, depth_path = self.pairs[idx]
        color_img = Image.open(color_path).convert("RGB")
        target_size = [color_img.height, color_img.width]
        return color_img, target_size, color_path, depth_path


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        color_imgs = [item[0] for item in batch]
        target_sizes = [item[1] for item in batch]
        color_paths = [item[2] for item in batch]
        depth_paths = [item[3] for item in batch]
        
        # CPU 端并行进行 resize 和 normalize 等预处理
        inputs = self.processor(images=color_imgs, return_tensors="pt")
        target_sizes = torch.tensor(target_sizes)
        return inputs, target_sizes, color_imgs, color_paths, depth_paths


def pad_and_clip(box, img_w, img_h, pad_ratio):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(img_w, int(x2 + px)),
        min(img_h, int(y2 + py)),
    )


def scale_box(box, src_w, src_h, dst_w, dst_h):
    """将检测框从 color 坐标系映射到 depth 坐标系。"""
    x1, y1, x2, y2 = box
    sx, sy = dst_w / src_w, dst_h / src_h
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def crop_and_save(img: Image.Image, box, save_path: Path):
    cropped = img.crop(box)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(save_path)


def crop_pair_task(color_path, depth_path, cbox, src_size, out_color, out_depth):
    color_path = Path(color_path)
    depth_path = Path(depth_path)
    out_color = Path(out_color)
    out_depth = Path(out_depth)

    color_img = Image.open(color_path).convert("RGB")
    depth_img = Image.open(depth_path)
    cw, ch = src_size
    dw, dh = depth_img.size
    if (dw, dh) != (cw, ch):
        dbox = scale_box(cbox, cw, ch, dw, dh)
    else:
        dbox = cbox

    crop_and_save(color_img, cbox, out_color)
    crop_and_save(depth_img, dbox, out_depth)


def main():
    parser = argparse.ArgumentParser(description="离线人脸检测 & 裁切")
    parser.add_argument("--src", type=Path, default=_default_src, help="原始数据集目录")
    parser.add_argument("--dst", type=Path, default=_default_dst, help="裁切后输出目录")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载的进程数")
    parser.add_argument("--crop-workers", type=int, default=4,
                        help="裁切保存的进程数，设为 0 表示主进程串行执行")
    parser.add_argument("--pad-ratio", type=float, default=0.15,
                        help="检测框向外扩展的比例")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="DETR 检测阈值")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    processor, model = build_detector(args.device)
    pairs = read_dataset(args.src)
    print(f"共找到 {len(pairs)} 对 color/depth 图片")

    dataset = FaceDataset(pairs)
    collator = Collator(processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True if args.device != "cpu" else False
    )

    skipped = 0
    pending = set()
    max_pending = max(1, args.crop_workers) * max(1, args.batch_size)

    def flush_pending(force=False):
        nonlocal pending
        if not pending:
            return
        if force:
            done, pending = wait(pending)
        elif len(pending) >= max_pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
        else:
            return
        for future in done:
            future.result()

    executor_cm = ProcessPoolExecutor(max_workers=args.crop_workers) if args.crop_workers > 0 else nullcontext(None)
    with executor_cm as executor:
        for inputs, target_sizes, color_imgs, color_paths, depth_paths in tqdm(dataloader, desc="检测 & 裁切"):
            inputs = {k: v.to(args.device, non_blocking=True) for k, v in inputs.items()}
            target_sizes = target_sizes.to(args.device, non_blocking=True)

            with torch.no_grad():
                outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=args.threshold
                )

            boxes = []
            for dets in results:
                if len(dets["scores"]) > 0:
                    best = dets["scores"].argmax()
                    boxes.append(dets["boxes"][best].cpu().numpy())
                else:
                    boxes.append(None)

            for color_path, depth_path, color_img, box in zip(color_paths, depth_paths, color_imgs, boxes):
                if box is None:
                    skipped += 1
                    continue

                cw, ch = color_img.size
                cbox = pad_and_clip(box, cw, ch, args.pad_ratio)
                rel_color = Path(color_path).relative_to(args.src)
                rel_depth = Path(depth_path).relative_to(args.src)
                out_color = args.dst / rel_color
                out_depth = args.dst / rel_depth

                if executor is None:
                    crop_pair_task(color_path, depth_path, cbox, (cw, ch), out_color, out_depth)
                else:
                    pending.add(executor.submit(
                        crop_pair_task,
                        color_path,
                        depth_path,
                        cbox,
                        (cw, ch),
                        out_color,
                        out_depth,
                    ))
                    flush_pending()

        flush_pending(force=True)

    print(f"完成! 已保存到 {args.dst}")
    if skipped:
        print(f"  跳过 {skipped} 张未检测到目标的图片")


if __name__ == "__main__":
    main()
