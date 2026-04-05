# FaceDet

这是一个以 RGB-D 视觉识别为主线的实验项目。核心思路是把 RGB 图像和深度图对齐后拼成 4 通道输入，完成识别训练与检索评估；在这个主线之外，我也保留了两个配套方向：旋转框目标检测，以及基于 DETR 的通用检测模型验证。

## 项目亮点

- 基于 RGB + Depth 的 4 通道输入，而不是只做普通 RGB 图像识别。
- 自定义 `CowResNet` 提取多尺度特征，再接 `ViT` 做全局表征学习。
- 训练阶段组合了 `ArcFaceLoss`、局部特征分块损失和可选对比学习损失。
- 验证阶段除了二分类相似度判断，还补了基于余弦相似度的 top-k 检索评估。
- 除主线识别外，还做了旋转框检测器和 DETR 微调与推理实验，方便横向比较方案。

## 我主要做了什么

### 1. RGB-D 识别主线

主线代码在 `FaceNetPack/`。

- `data_processor.py` 负责读取 `color/` 和 `depth/` 成对数据，并在增强时保证两路图像同步变换，避免通道错位。
- `Model/Backbone.py` 中的 `CowResNet` 接收 4 通道输入，融合中层和高层特征，输出 512 维特征图。
- `Model/VisionTransformer.py` 将卷积特征展平成 token，再通过 Transformer 编码得到全局 embedding。
- `cloud_train.py` 串起了 DDP 训练、损失计算、验证、top-k 检索评估和 TensorBoard 记录。

这一部分体现的重点是：多模态输入处理、识别模型设计，以及训练评估链路的完整性。

### 2. 旋转框检测实验

`ObjDetARec/` 是一套自定义旋转框检测实验代码。

- 支持从 XML 中解析 `robndbox` 标注。
- 检测头直接预测 `cx、cy、w、h、angle`。
- 推理阶段依赖 `mmcv.ops.nms_rotated` 做旋转框 NMS。

这一部分更多是在验证我对目标检测任务、旋转框表示和自定义损失设计的理解。

### 3. 通用模型验证

根目录下保留了几个快速实验脚本：

- `train_detr.py`：基于 Hugging Face `transformers` 微调 DETR。
- `test_detr.py`：加载本地权重做单张图推理并输出结果图。

这部分主要说明我不只停留在自定义模型，也会把通用预训练模型接进自己的数据流程里做验证。

## 目录结构

```text
FaceDet/
├── FaceNetPack/              # RGB-D 识别主线
├── ObjDetARec/               # 旋转框检测实验
├── detr-resnet-50/           # 本地 DETR 基础权重
├── detr-cow-finetuned/       # DETR 微调输出
├── train_detr.py
├── test_detr.py
└── README.md
```

## 数据格式

RGB-D 识别分支默认读取下面这种结构：

```text
Dataset/
└── 325_crop/
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

训练 RGB-D 识别模型：

```bash
torchrun --nproc_per_node=1 FaceNetPack/cloud_train.py
```

