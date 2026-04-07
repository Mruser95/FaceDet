# FaceDet

这是一个以 RGB-D 视觉识别为主线的实验项目。核心思路是把 RGB 图像和深度图对齐后拼成 4 通道输入，完成识别训练与检索评估；在这个主线之外，我也保留了两个配套方向：旋转框目标检测，以及基于 DETR 的通用检测模型验证。

## 项目亮点

- 基于 RGB + Depth 的 4 通道输入，而不是只做普通 RGB 图像识别。
- 自定义 `CowResNet` 提取多尺度特征，再接 `ViT` 做全局表征学习。
- 训练阶段组合了 `ArcFaceLoss`、局部特征分块损失和可选对比学习损失。
- 验证阶段除了二分类相似度判断，还补了基于余弦相似度的 top-k 检索评估。
- **完整的数据预处理流水线**：SAM 前景分割 → DETR 检测裁切 → 质量清洗，三步渐进式提升数据质量。
- 除主线识别外，还做了旋转框检测器和 DETR 微调与推理实验，方便横向比较方案。

## 我主要做了什么

### 1. 数据预处理流水线

原始 RGB-D 数据需要经过三步预处理才能用于训练。代码均在 `FaceNetPack/` 下。

```text
原始 325 ──► sam_processor ──► 325_sam ──► crop_offline ──► 325_crop ──► clean_crop_data ──► 训练就绪
            (SAM 前景分割)                (DETR 检测裁切)                 (质量清洗)
```

- **`sam_processor.py`** — 利用深度图的前景/背景差异（Otsu 二值化 + 连通域分析）自动生成 SAM 的点/框 prompt，让 SAM ViT-H 在 RGB 图上输出精确 mask，再同时应用到 color 和 depth。解决了俯视角 RGB 过曝导致前景难以直接分离的问题。
- **`crop_offline.py`** — 加载微调后的 DETR（优先 `detr-cow-finetuned`），批量检测 SAM 处理后的 color 图，用检测框同步裁切 color 和 depth，并在 SAM 输出质量差时自动回退到原始 `325` 数据。支持多进程并行写出。
- **`clean_crop_data.py`** — 扫描裁切结果，按面积、最短边、非零像素比例筛出低质量图片，支持 `--delete` 成对删除。

### 2. RGB-D 识别主线

主线代码在 `FaceNetPack/`。

- **`data_processor.py`** — 读取 color/depth 成对数据（优先级：`325_crop` > `325_sam` > `325`），内置质量过滤；增强时保证两路图像同步变换，避免通道错位；提供分类训练集、配对验证集和逐张 embedding 数据集三种 DataLoader。
- **`Model/Backbone.py`** — `CowResNet` 接收 4 通道输入，融合中层和高层特征，输出 512 维特征图。
- **`Model/VisionTransformer.py`** — 将卷积特征展平成 token，通过 Transformer 编码得到全局 embedding；支持同时返回局部特征图。
- **`Model/ArcFace.py`** — 包含 `ArcFaceLoss`、`LocalSplitArcFaceLoss`（沿 H 轴切分特征图做多块 ArcFace + BN）和 `SupConLoss`（监督对比学习损失），以及分组 weight decay 的 `AdamW` 构造。
- **`Model/MarginModel.py`** — 加载训练好的检查点，在验证集上扫阈值求最优 margin，并可绘制 ROC 曲线。
- **`cloud_train.py`** — 串起 DDP 训练、AMP 混合精度、梯度裁剪、warmup + plateau 调度、多种损失计算、top-k 检索评估和 TensorBoard 记录。

### 3. 旋转框检测实验

`ObjDetARec/` 是一套自定义旋转框检测实验代码。

- 支持从 XML 中解析 `robndbox` 标注。
- 检测头直接预测 `cx、cy、w、h、angle`。
- 推理阶段依赖 `mmcv.ops.nms_rotated` 做旋转框 NMS。

### 4. 通用模型验证

根目录下保留了几个快速实验脚本：

- **`train_detr.py`** — 基于 Hugging Face `transformers` 微调 DETR，支持 Pascal VOC 风格 XML 标注（`bndbox` / `robndbox`），含 backbone 冻结、自定义评估等。
- **`test_detr.py`** — 加载本地权重做单张图推理，NMS 后画框保存结果图。
- **`test_shape.py`** — 调试用，打印 `CowResNet` 各层输出形状。

## 目录结构

```text
FaceDet/
├── FaceNetPack/                  # RGB-D 识别主线 + 数据预处理
│   ├── sam_processor.py          #   SAM 前景分割
│   ├── crop_offline.py           #   DETR 检测裁切
│   ├── clean_crop_data.py        #   裁切质量清洗
│   ├── data_processor.py         #   训练数据加载
│   ├── cloud_train.py            #   分布式训练入口
│   ├── Model/
│   │   ├── Backbone.py           #     CowResNet (4ch → 512ch)
│   │   ├── VisionTransformer.py  #     ViT 全局编码
│   │   ├── ArcFace.py            #     损失函数集合
│   │   └── MarginModel.py        #     阈值分析 & ROC
│   ├── Dataset/                  #   数据 (gitignored)
│   │   ├── 325/                  #     原始 RGB-D
│   │   ├── 325_sam/              #     SAM 处理后
│   │   └── 325_crop/             #     裁切后
│   └── State/                    #   检查点 (cloud_model.pth)
├── ObjDetARec/                   # 旋转框检测实验 (gitignored)
├── detr-resnet-50/               # 本地 DETR 基础权重
├── detr-cow-finetuned/           # DETR 微调输出
├── train_detr.py
├── test_detr.py
├── test_shape.py
└── README.md
```

## 数据格式

RGB-D 识别分支默认读取下面这种结构：

```text
Dataset/
└── 325_crop/          ← 优先; 不存在则回退 325_sam → 325
    ├── person_a/
    │   ├── color/
    │   └── depth/
    └── person_b/
        ├── color/
        └── depth/
```

要求 `color/` 和 `depth/` 中的文件名末尾编号一致，代码会按编号自动配对。

## 快速开始

### 依赖安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision transformers pillow numpy tqdm matplotlib opencv-python tensorboard timm safetensors
```

如果要跑 `ObjDetARec/`，还需要安装与本机 CUDA / PyTorch 匹配的 `mmcv`。

如果是新克隆的仓库，记得执行：

```bash
git lfs pull
```

### 常用命令

#### 数据预处理

SAM 前景分割（`325` → `325_sam`）：

```bash
python -m FaceNetPack.sam_processor --src Dataset/325 --dst Dataset/325_sam --device cuda
```

DETR 检测裁切（`325_sam` → `325_crop`）：

```bash
python -m FaceNetPack.crop_offline --src Dataset/325_sam --dst Dataset/325_crop
```

清洗裁切结果（先扫描、确认后加 `--delete`）：

```bash
python -m FaceNetPack.clean_crop_data                # 仅扫描
python -m FaceNetPack.clean_crop_data --delete        # 删除低质量图
```

#### 训练与推理

训练 RGB-D 识别模型：

```bash
torchrun --nproc_per_node=1 FaceNetPack/cloud_train.py
```

阈值分析与 ROC 曲线：

```bash
python -m FaceNetPack.Model.MarginModel
```

微调 DETR：

```bash
python3 train_detr.py \
  --model-path ./detr-resnet-50 \
  --image-dir ./ObjDetARec/yolo/images \
  --label-dir ./ObjDetARec/yolo/labels \
  --output-dir ./detr-cow-finetuned
```

运行 DETR 推理：

```bash
python3 test_detr.py
```

训练旋转框检测器：

```bash
torchrun --nproc_per_node=1 ObjDetARec/main.py --epochs 10
```
