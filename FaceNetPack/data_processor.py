import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from pathlib import Path
from PIL import Image
import random
import re
import numpy as np

_sam_root = Path(__file__).resolve().parent / "Dataset" / "325_sam"
dataroot = Path(__file__).resolve().parent / "Dataset" / "325_crop"
_fallback_root = Path(__file__).resolve().parent / "Dataset" / "325"

# 复用同一组 transform 实例可省掉每张图片的对象构造开销
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE_4CH = transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)


def read_dataset(root=None):
    if root is None:
        for _candidate in [dataroot, _sam_root, _fallback_root]:
            if _candidate.exists():
                root = _candidate
                break
        else:
            root = _fallback_root

    num_re = re.compile(r"(\d+)$")

    def extract_id(stem: str):
        m = num_re.search(stem)
        return int(m.group(1)) if m else None

    persons = []
    for sub in sorted(root.glob('*')):
        colroot, deproot = sub / "color", sub / "depth"
        if not (colroot.exists() and deproot.exists()):
            continue
        color = {extract_id(p.stem): p for p in colroot.glob("*")}
        depth = {extract_id(p.stem): p for p in deproot.glob("*")}
        color.pop(None, None)
        depth.pop(None, None)
        imgs = sorted(set(color) & set(depth))
        person = []
        for p in imgs:
            person.append([color[p], depth[p]])
        if person:
            persons.append(person)

    return persons


def _jitter_rgb(img4: torch.Tensor, rng: random.Random):
    rgb = img4[:3]
    rgb = TF.adjust_brightness(rgb, 1.0 + rng.uniform(-0.15, 0.15))
    rgb = TF.adjust_contrast(rgb, 1.0 + rng.uniform(-0.15, 0.15))
    rgb = TF.adjust_saturation(rgb, 1.0 + rng.uniform(-0.1, 0.1))
    rgb = TF.adjust_hue(rgb, rng.uniform(-0.02, 0.02))
    return torch.cat([rgb.clamp(0.0, 1.0), img4[3:]], dim=0)


def _erase_patch(img4: torch.Tensor, rng: random.Random):
    _, h, w = img4.shape
    area = h * w
    for _ in range(10):
        target_area = area * rng.uniform(0.01, 0.08)
        aspect_ratio = rng.uniform(0.75, 1.33)
        erase_h = int(round((target_area * aspect_ratio) ** 0.5))
        erase_w = int(round((target_area / aspect_ratio) ** 0.5))
        if 0 < erase_h < h and 0 < erase_w < w:
            top = rng.randint(0, h - erase_h)
            left = rng.randint(0, w - erase_w)
            img4[:, top:top + erase_h, left:left + erase_w] = 0.0
            break
    return img4


def _apply_train_augment(img4: torch.Tensor, out_size, rng: random.Random):
    height, width = out_size
    max_dx = max(1, int(width * 0.02))
    max_dy = max(1, int(height * 0.02))
    img4 = TF.affine(
        img4,
        angle=rng.uniform(-2.0, 2.0),
        translate=(rng.randint(-max_dx, max_dx), rng.randint(-max_dy, max_dy)),
        scale=rng.uniform(0.98, 1.02),
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    if rng.random() < 0.5:
        img4 = _jitter_rgb(img4, rng)
    if rng.random() < 0.08:
        #img4 = _erase_patch(img4, rng)
        pass
    return img4.clamp(0.0, 1.0)


def _load_pair(img, out_size, train=False, rng=None):
    color_path, depth_path = img
    with Image.open(color_path) as color_img:
        color = _TO_TENSOR(color_img.convert("RGB"))
    with Image.open(depth_path) as depth_img:
        if depth_img.mode != "I;16":
            depth = _TO_TENSOR(depth_img)
        else:
            arr = np.array(depth_img, dtype=np.float32)
            arr = arr / 65535.0 if arr.max() > 1 else arr
            depth = torch.from_numpy(arr).unsqueeze(0)

    if rng is None:
        rng = random.Random()

    img4 = torch.cat([color, depth], dim=0)
    img4 = TF.resize(img4, out_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    if train:
        # 几何增强必须对 RGB 和 depth 同步，避免四通道错位。
        img4 = _apply_train_augment(img4, out_size, rng)
    return _NORMALIZE_4CH(img4)


class dataset(Dataset):
    def __init__(self, persons, train=True, train_num=1e5, img_size=(424, 240)):
        super().__init__()
        self.persons = persons
        self.train = train
        self.train_num = int(train_num)
        self.out_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.train_num

    def __getitem__(self, idx):
        rng = random.Random(idx + self.epoch * self.train_num)
        if not self.train:
            same = rng.choice([True, False])
            if same:
                group = rng.choice(self.persons)
                img1, img2 = rng.sample(group, 2) if len(group) > 1 else (group[0], group[0])
            else:
                g1, g2 = rng.sample(self.persons, 2)
                img1, img2 = rng.choice(g1), rng.choice(g2)
            img1 = _load_pair(img1, self.out_size, train=False, rng=rng)
            img2 = _load_pair(img2, self.out_size, train=False, rng=rng)
            return img1, img2, torch.tensor(same, dtype=torch.long)
        else:
            ids = rng.randint(0, len(self.persons) - 1)
            img = rng.choice(self.persons[ids])
            img = _load_pair(img, self.out_size, train=True, rng=rng)
            return img, torch.tensor(ids, dtype=torch.long)


class SingleImageDataset(Dataset):
    """展开 persons 为单张图片，返回 (image_tensor, person_id)。"""
    def __init__(self, persons, img_size=(424, 240)):
        super().__init__()
        self.out_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.samples = []
        self.paths = []
        for pid, group in enumerate(persons):
            for img in group:
                self.samples.append((img, pid))
                self.paths.append(str(img[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, pid = self.samples[idx]
        img = _load_pair(img_paths, self.out_size, train=False)
        return img, pid


class Process:
    def __init__(self, img_size=(424, 240), train_size=0.8, train_num=1e6, val_num=1e5, **_kw):
        self.img_size = img_size
        self.train_num = train_num
        self.val_num = val_num
        persons = read_dataset()
        random.Random(42).shuffle(persons)
        tn = int(len(persons) * train_size)
        self.train_ps, self.val_ps = persons[:tn], persons[tn:]
        self.num_train = tn

    def loader(self, world_size=1, rank=0, num_worker=8, prefetch_factor=2, batch_size=128):
        common = {
            "num_workers": num_worker,
            "prefetch_factor": prefetch_factor if num_worker > 0 else None,
            "persistent_workers": num_worker > 0,
            "pin_memory": torch.cuda.is_available()
        }
        common = {k: v for k, v in common.items() if v is not None}

        train_ds = dataset(self.train_ps, True, self.train_num, self.img_size)
        val_ds = dataset(self.val_ps, False, self.val_num, self.img_size)

        train_sp = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sp = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

        train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=train_sp, drop_last=True, **common)
        val_ld = DataLoader(val_ds, batch_size=batch_size, sampler=val_sp, drop_last=True, **common)
        return train_ld, val_ld
