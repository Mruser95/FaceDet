"""使用 SAM ViT-H 结合深度图自动生成 prompt，对数据集进行前景分割。

思路:
  - 俯视奶牛背部的 RGB 图片存在过曝，直接用 RGB 特征难以可靠区分前景
  - 深度图中奶牛背部和地面处于不同深度层，可通过 Otsu 二值化提取前景区域
  - 用前景区域的质心(正样本点)和边界框作为 SAM 的 prompt
  - SAM 在 RGB 图上生成精确的像素级 mask
  - 将 mask 同时应用到 RGB 和 depth，去除背景噪声

输出与原始数据集同结构 (person/color/, person/depth/)，
运行后 data_processor 会自动优先读取 325_sam 目录。

用法:
  python -m FaceNetPack.sam_processor
  python -m FaceNetPack.sam_processor --src Dataset/325 --dst Dataset/325_sam --device cuda
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

_script_dir = Path(__file__).resolve().parent
_default_src = _script_dir / "Dataset" / "325"
_default_dst = _script_dir / "Dataset" / "325_sam"


def build_sam(model_id: str, device: str):
    """从 HuggingFace Hub 加载 SAM ViT-H（首次运行会自动下载权重）。"""
    print(f"Loading SAM from {model_id} ...")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device)
    model.eval()
    print("SAM loaded.")
    return processor, model


# ---------------------------------------------------------------------------
# 深度图 → SAM prompt
# ---------------------------------------------------------------------------

def depth_to_prompt(depth_path, rgb_size=None):
    """从深度图提取前景区域的质心和边界框，用作 SAM 的正样本 prompt。

    策略：对非零深度做 Otsu 二值化，取最大连通域作为前景候选，
    优先选择居中且面积合理的区域。

    Args:
        depth_path: 深度图路径。
        rgb_size: RGB 图的 (W, H)，用于把深度坐标映射到 RGB 坐标系。

    Returns:
        {"point": (x, y), "box": [x1, y1, x2, y2]} 或 None。
    """
    depth_img = Image.open(depth_path)
    arr = np.array(depth_img, dtype=np.float32)
    d_h, d_w = arr.shape[:2]

    valid = arr > 0
    if valid.sum() < 100:
        return None

    valid_vals = arr[valid]
    d_min, d_max = float(valid_vals.min()), float(valid_vals.max())
    if d_max - d_min < 1:
        return None

    norm = np.zeros((d_h, d_w), dtype=np.uint8)
    norm[valid] = ((arr[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    for fg_val in (0, 255):
        fg = (binary == fg_val) & valid
        if fg.sum() < 100:
            continue

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fg.astype(np.uint8), connectivity=8,
        )
        if n_labels <= 1:
            continue

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_id = int(areas.argmax()) + 1
        area = int(areas[largest_id - 1])
        ratio = area / (d_h * d_w)
        if ratio < 0.03 or ratio > 0.85:
            continue

        cx, cy = centroids[largest_id]
        dist = ((cx - d_w / 2) ** 2 + (cy - d_h / 2) ** 2) ** 0.5
        max_dist = ((d_w / 2) ** 2 + (d_h / 2) ** 2) ** 0.5
        score = (1 - dist / max_dist) * 0.5 + min(ratio, 0.5)

        x1 = int(stats[largest_id, cv2.CC_STAT_LEFT])
        y1 = int(stats[largest_id, cv2.CC_STAT_TOP])
        bw = int(stats[largest_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[largest_id, cv2.CC_STAT_HEIGHT])
        x2, y2 = x1 + bw, y1 + bh

        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(d_w, x2 + pad), min(d_h, y2 + pad)

        cand = {"point": (float(cx), float(cy)), "box": [x1, y1, x2, y2], "score": score}
        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        best = {"point": (d_w / 2.0, d_h / 2.0), "box": None}

    if rgb_size is not None:
        rw, rh = rgb_size
        sx, sy = rw / d_w, rh / d_h
        px, py = best["point"]
        best["point"] = (px * sx, py * sy)
        if best["box"] is not None:
            bx1, by1, bx2, by2 = best["box"]
            best["box"] = [int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)]

    return best


# ---------------------------------------------------------------------------
# 数据集遍历
# ---------------------------------------------------------------------------

def read_dataset(root):
    """读取数据集，返回 (person_name, color_path, depth_path) 列表。"""
    num_re = re.compile(r"(\d+)$")

    def extract_id(stem):
        m = num_re.search(stem)
        return int(m.group(1)) if m else None

    pairs = []
    for sub in sorted(root.glob("*")):
        if not sub.is_dir():
            continue
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


# ---------------------------------------------------------------------------
# SAM 推理 + mask 应用
# ---------------------------------------------------------------------------

def segment_image(rgb_path, depth_path, sam_proc, sam_model, device):
    """对单张图做 SAM 前景分割，返回 bool mask (H, W) 或 None。"""
    rgb = Image.open(rgb_path).convert("RGB")
    rw, rh = rgb.size

    prompt = depth_to_prompt(depth_path, rgb_size=(rw, rh))
    if prompt is None:
        return None

    cx, cy = prompt["point"]
    kw = {
        "images": rgb,
        "input_points": [[[int(cx), int(cy)]]],
        "input_labels": [[1]],
        "return_tensors": "pt",
    }
    if prompt["box"] is not None:
        kw["input_boxes"] = [[prompt["box"]]]

    inputs = sam_proc(**kw).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    scores = outputs.iou_scores[0, 0]
    best_idx = scores.argmax().item()
    mask = masks[0][0, best_idx].numpy().astype(bool)
    return mask


def segment_batch(rgb_images, prompts, sam_proc, sam_model, device):
    """对一批图做 SAM 前景分割，返回 bool mask 列表。"""
    points = [[[int(p["point"][0]), int(p["point"][1])]] for p in prompts]
    labels = [[1] for _ in prompts]
    boxes = []
    for k, p in enumerate(prompts):
        if p["box"] is not None:
            boxes.append([p["box"]])
        else:
            w, h = rgb_images[k].size
            boxes.append([[0, 0, w, h]])

    inputs = sam_proc(
        images=rgb_images,
        input_points=points,
        input_labels=labels,
        input_boxes=boxes,
        return_tensors="pt",
    ).to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.no_grad(), torch.autocast(device_type, enabled=device_type == "cuda"):
        outputs = sam_model(**inputs)

    all_masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    results = []
    for k in range(len(rgb_images)):
        scores = outputs.iou_scores[k, 0]
        best_idx = scores.argmax().item()
        mask = all_masks[k][0, best_idx].numpy().astype(bool)
        results.append(mask)
    return results


def apply_and_save(rgb_path, depth_path, mask, out_rgb, out_depth, *, rgb_pil=None):
    """将 mask 应用到 RGB 和 depth 并保存（背景像素置零）。"""
    if rgb_pil is not None:
        rgb = np.array(rgb_pil if rgb_pil.mode == "RGB" else rgb_pil.convert("RGB"))
    else:
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img)

    if mask.shape[:2] != rgb.shape[:2]:
        mask_rgb = cv2.resize(
            mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_rgb = mask

    if mask.shape[:2] != depth.shape[:2]:
        mask_dep = cv2.resize(
            mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_dep = mask

    rgb[~mask_rgb] = 0
    depth[~mask_dep] = 0

    out_rgb.parent.mkdir(parents=True, exist_ok=True)
    out_depth.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(str(out_rgb))
    cv2.imwrite(str(out_depth), depth)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _load_one(item):
    """加载一张 RGB 并计算深度 prompt（供线程池调用）。"""
    _person, color_path, depth_path, out_color, out_depth = item
    try:
        rgb = Image.open(color_path).convert("RGB")
        prompt = depth_to_prompt(depth_path, rgb_size=rgb.size)
        if prompt is None:
            return None
        return rgb, prompt, (color_path, depth_path, out_color, out_depth)
    except Exception as e:
        print(f"\n读取失败 {color_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="SAM ViT-H + 深度图 prompt 前景分割",
    )
    parser.add_argument("--src", type=Path, default=_default_src,
                        help="原始数据集目录 (默认 Dataset/325)")
    parser.add_argument("--dst", type=Path, default=_default_dst,
                        help="输出目录 (默认 Dataset/325_sam)")
    parser.add_argument("--model-id", default="facebook/sam-vit-huge",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="重新处理已存在的输出文件")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="每批送入 SAM 的图片数 (默认 12，显存不足时调小)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="读图线程数 (默认 8)")
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing

    sam_proc, sam_model = build_sam(args.model_id, args.device)
    pairs = read_dataset(args.src)
    print(f"共找到 {len(pairs)} 对 color/depth 图片")

    todo = []
    skipped = 0
    for person, color_path, depth_path in pairs:
        rel_color = color_path.relative_to(args.src)
        rel_depth = depth_path.relative_to(args.src)
        out_color = args.dst / rel_color
        out_depth = args.dst / rel_depth
        if skip_existing and out_color.exists() and out_depth.exists():
            skipped += 1
            continue
        todo.append((person, color_path, depth_path, out_color, out_depth))

    print(f"待处理: {len(todo)}, 已跳过: {skipped}")

    done, failed = 0, 0
    pbar = tqdm(total=len(todo), desc="SAM 分割")
    bs = args.batch_size
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        # 预取第一个 batch 的读图任务
        next_futs = (
            [pool.submit(_load_one, item) for item in batches[0]]
            if batches else []
        )

        for b_idx in range(len(batches)):
            cur_futs = next_futs

            # GPU 跑当前 batch 之前，先提交下一个 batch 的读图，
            # 这样 GPU 推理期间 CPU 线程已经在并行读图了。
            if b_idx + 1 < len(batches):
                next_futs = [pool.submit(_load_one, item)
                             for item in batches[b_idx + 1]]
            else:
                next_futs = []

            # 收集当前 batch 的读图结果
            batch_rgb, batch_prompts, batch_meta = [], [], []
            for fut in cur_futs:
                result = fut.result()
                if result is None:
                    failed += 1
                    pbar.update(1)
                else:
                    rgb, prompt, meta = result
                    batch_rgb.append(rgb)
                    batch_prompts.append(prompt)
                    batch_meta.append(meta)

            if not batch_rgb:
                continue

            # GPU 推理（此时下一批图片正在后台线程加载）
            try:
                masks = segment_batch(
                    batch_rgb, batch_prompts, sam_proc, sam_model, args.device,
                )
            except Exception as e:
                print(f"\nSAM 推理失败: {e}")
                failed += len(batch_rgb)
                pbar.update(len(batch_rgb))
                continue

            # 保存结果
            for mask, rgb_img, (color_path, depth_path, out_color, out_depth) in zip(
                masks, batch_rgb, batch_meta,
            ):
                try:
                    apply_and_save(
                        color_path, depth_path, mask, out_color, out_depth,
                        rgb_pil=rgb_img,
                    )
                    done += 1
                except Exception as e:
                    print(f"\n保存失败 {color_path}: {e}")
                    failed += 1
                pbar.update(1)

    pbar.close()
    print(f"\n完成! 已保存到 {args.dst}")
    print(f"  成功: {done}, 跳过: {skipped}, 失败: {failed}")


if __name__ == "__main__":
    main()
