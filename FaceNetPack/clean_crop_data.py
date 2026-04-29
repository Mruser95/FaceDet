"""清洗 325_crop 数据集中的低质量裁切图片。

功能：
  1. 扫描 325_crop，找出尺寸过小或内容过少的图片
  2. 删除这些坏图（color + depth 成对删除）
  3. 输出报告

用法：
  python -m FaceNetPack.clean_crop_data                # 仅扫描，不删除
  python -m FaceNetPack.clean_crop_data --delete        # 扫描并删除坏图
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

_default_root = Path(__file__).resolve().parent / "Dataset" / "325_crop"

MIN_CROP_AREA = 6000
MIN_CROP_DIM = 50
MIN_CONTENT_RATIO = 0.15
MAX_DEPTH_MEAN = 2500
MIN_REL_AREA = 0.5
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 5.0
MIN_ASPECT_MEDIAN_FACTOR = 0.45
MAX_ASPECT_MEDIAN_FACTOR = 2.2


def _find_depth_path(color_path: Path) -> Path | None:
    """根据 color 路径推断对应的 depth 路径。"""
    stem = color_path.stem
    fid = stem.split("_")[-1]
    prefix = "_".join(stem.split("_")[:-2])
    depth_dir = color_path.parent.parent / "depth"
    for ext in [".png", ".jpg"]:
        dp = depth_dir / f"{prefix}_depth_{fid}{ext}"
        if dp.exists():
            return dp
    return None


def _collect_person_stats(root):
    stats = {}
    for person_dir in sorted(root.glob("*")):
        color_dir = person_dir / "color"
        if not color_dir.exists():
            continue
        areas, aspects = [], []
        for color_path in color_dir.glob("*"):
            if not color_path.is_file():
                continue
            try:
                with Image.open(color_path) as img:
                    w, h = img.size
            except Exception:
                continue
            if h <= 0:
                continue
            areas.append(w * h)
            aspects.append(w / h)
        if areas:
            stats[person_dir] = {
                "area_median": float(np.median(areas)),
                "aspect_median": float(np.median(aspects)),
            }
    return stats


def scan(root, min_area, min_dim, min_content, max_depth=MAX_DEPTH_MEAN,
         min_rel_area=MIN_REL_AREA,
         min_aspect=MIN_ASPECT_RATIO,
         max_aspect=MAX_ASPECT_RATIO,
         min_aspect_median_factor=MIN_ASPECT_MEDIAN_FACTOR,
         max_aspect_median_factor=MAX_ASPECT_MEDIAN_FACTOR):
    bad = []
    total = 0
    person_stats = _collect_person_stats(root)
    for color_path in sorted(root.rglob("color/*")):
        if not color_path.is_file():
            continue
        total += 1
        try:
            img = Image.open(color_path)
            w, h = img.size
        except Exception as e:
            bad.append((color_path, f"open failed: {e}"))
            continue

        reasons = []
        if w * h < min_area:
            reasons.append(f"area={w * h}<{min_area}")
        if min(w, h) < min_dim:
            reasons.append(f"min_dim={min(w, h)}<{min_dim}")

        aspect = w / h if h else 0.0
        if aspect < min_aspect or aspect > max_aspect:
            reasons.append(f"aspect={aspect:.2f} not in [{min_aspect}, {max_aspect}]")

        stats = person_stats.get(color_path.parent.parent)
        if stats is not None:
            area_median = stats["area_median"]
            aspect_median = stats["aspect_median"]
            rel_area = (w * h) / area_median if area_median > 0 else 1.0
            if rel_area < min_rel_area:
                reasons.append(f"rel_area={rel_area:.2f}<{min_rel_area}")
            if aspect_median > 0:
                rel_aspect = aspect / aspect_median
                if rel_aspect < min_aspect_median_factor or rel_aspect > max_aspect_median_factor:
                    reasons.append(
                        f"rel_aspect={rel_aspect:.2f} not in "
                        f"[{min_aspect_median_factor}, {max_aspect_median_factor}]"
                    )

        arr = np.array(img)
        ratio = (arr > 5).mean()
        if ratio < min_content:
            reasons.append(f"content={ratio:.3f}<{min_content}")

        if max_depth > 0:
            dp = _find_depth_path(color_path)
            if dp is not None:
                d_arr = np.array(Image.open(dp), dtype=np.float32)
                d_valid = d_arr[d_arr > 0]
                if d_valid.size > 0 and d_valid.mean() > max_depth:
                    reasons.append(f"depth={d_valid.mean():.0f}>{max_depth}")

        if reasons:
            bad.append((color_path, ", ".join(reasons)))

    return total, bad


def main():
    parser = argparse.ArgumentParser(description="清洗低质量裁切图片")
    parser.add_argument("--root", type=Path, default=_default_root)
    parser.add_argument("--min-area", type=int, default=MIN_CROP_AREA)
    parser.add_argument("--min-dim", type=int, default=MIN_CROP_DIM)
    parser.add_argument("--min-content", type=float, default=MIN_CONTENT_RATIO)
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_MEAN,
                        help="深度均值超过此阈值的图片视为远景/非目标 (默认 2500)")
    parser.add_argument("--min-rel-area", type=float, default=MIN_REL_AREA,
                        help="低于同身份中位面积该比例的裁切视为坏图 (默认 0.5)")
    parser.add_argument("--min-aspect", type=float, default=MIN_ASPECT_RATIO,
                        help="最小允许宽高比，低于此值多为竖条/半截图 (默认 0.45)")
    parser.add_argument("--max-aspect", type=float, default=MAX_ASPECT_RATIO,
                        help="最大允许宽高比，高于此值多为横条/半截图 (默认 5.0)")
    parser.add_argument("--min-aspect-median-factor", type=float, default=MIN_ASPECT_MEDIAN_FACTOR,
                        help="低于同身份中位宽高比该倍数视为突变 (默认 0.45)")
    parser.add_argument("--max-aspect-median-factor", type=float, default=MAX_ASPECT_MEDIAN_FACTOR,
                        help="高于同身份中位宽高比该倍数视为突变 (默认 2.2)")
    parser.add_argument("--delete", action="store_true", help="实际删除坏图")
    args = parser.parse_args()

    print(f"扫描目录: {args.root}")
    total, bad = scan(
        args.root,
        args.min_area,
        args.min_dim,
        args.min_content,
        args.max_depth,
        args.min_rel_area,
        args.min_aspect,
        args.max_aspect,
        args.min_aspect_median_factor,
        args.max_aspect_median_factor,
    )

    print(f"总计: {total} 张, 低质量: {len(bad)} 张 ({len(bad) / max(total, 1) * 100:.1f}%)")
    for path, reason in bad[:30]:
        print(f"  {path.relative_to(args.root)}  -- {reason}")
    if len(bad) > 30:
        print(f"  ... 还有 {len(bad) - 30} 张")

    if args.delete and bad:
        deleted = 0
        for color_path, _ in bad:
            depth_path = _find_depth_path(color_path)
            for p in [color_path, depth_path]:
                if p is None:
                    continue
                if p.exists():
                    p.unlink()
                    deleted += 1
        print(f"已删除 {deleted} 个文件")
    elif bad and not args.delete:
        print("\n提示: 添加 --delete 参数来实际删除这些文件")


if __name__ == "__main__":
    main()
