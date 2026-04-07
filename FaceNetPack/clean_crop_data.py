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


def scan(root, min_area, min_dim, min_content):
    bad = []
    total = 0
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

        arr = np.array(img)
        ratio = (arr > 5).mean()
        if ratio < min_content:
            reasons.append(f"content={ratio:.3f}<{min_content}")

        if reasons:
            bad.append((color_path, ", ".join(reasons)))

    return total, bad


def main():
    parser = argparse.ArgumentParser(description="清洗低质量裁切图片")
    parser.add_argument("--root", type=Path, default=_default_root)
    parser.add_argument("--min-area", type=int, default=MIN_CROP_AREA)
    parser.add_argument("--min-dim", type=int, default=MIN_CROP_DIM)
    parser.add_argument("--min-content", type=float, default=MIN_CONTENT_RATIO)
    parser.add_argument("--delete", action="store_true", help="实际删除坏图")
    args = parser.parse_args()

    print(f"扫描目录: {args.root}")
    total, bad = scan(args.root, args.min_area, args.min_dim, args.min_content)

    print(f"总计: {total} 张, 低质量: {len(bad)} 张 ({len(bad) / max(total, 1) * 100:.1f}%)")
    for path, reason in bad[:30]:
        print(f"  {path.relative_to(args.root)}  -- {reason}")
    if len(bad) > 30:
        print(f"  ... 还有 {len(bad) - 30} 张")

    if args.delete and bad:
        deleted = 0
        for color_path, _ in bad:
            depth_path = color_path.parent.parent / "depth" / color_path.name.replace("color", "depth")
            for p in [color_path, depth_path]:
                if p.exists():
                    p.unlink()
                    deleted += 1
        print(f"已删除 {deleted} 个文件")
    elif bad and not args.delete:
        print("\n提示: 添加 --delete 参数来实际删除这些文件")


if __name__ == "__main__":
    main()
