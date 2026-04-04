# FaceDet

FaceDet 是一个以 RGB-D 人脸识别为主线的实验型视觉仓库，同时保留了两条配套实验分支：

- 基于 4 通道输入（RGB + depth）的人脸/身份识别训练流程
- 基于旋转框标注的自定义目标检测器 `ObjDetARec`
- 基于 Hugging Face `transformers` 的 DETR 微调与 SAM 分割验证脚本

这个仓库更接近“研究代码集合”而不是已经产品化的工程，因此 README 重点说明当前代码的真实结构、数据组织方式、运行入口和已知注意事项。

## 1. 项目概览

仓库里大致有三块内容：

1. `FaceNetPack/`
   - 负责 RGB-D 识别任务。
   - 输入是对齐的 color/depth 图像对，数据处理时会拼成 4 通道张量。
   - 主干模型是自定义 `CowResNet`，再接 `ViT` 输出全局 embedding。
   - 训练损失包含 `ArcFaceLoss`、`LocalSplitArcFaceLoss` 和可选 `SupConLoss`。

2. `ObjDetARec/`
   - 负责旋转框目标检测实验。
   - 读取 `yolo/images` 与 `yolo/proc_labels`，训练自定义检测头。
   - 依赖 `mmcv.ops.nms_rotated` 和 `box_iou_rotated`。

3. 根目录实验脚本
   - `train_detr.py`：使用本地 DETR 权重微调单类别检测模型（当前类别写死为 `cow`）。
   - `test_detr.py`：对 `cow.jpg` 做推理并保存 `result.jpg`。
   - `test_sam.py`：基于手工给定 bbox，用本地 SAM 权重做分割演示。
   - `test_shape.py`：检查 `CowResNet` 每一层的输出形状。

## 2. 仓库结构

```text
FaceDet/
├── FaceNetPack/
│   ├── Model/
│   │   ├── ArcFace.py
│   │   ├── Backbone.py
│   │   ├── MarginModel.py
│   │   └── VisionTransformer.py
│   ├── Dataset/
│   │   └── 325/              # 原始 RGB-D 数据（仓库内默认目录之一）
│   ├── cloud_train.py
│   ├── crop_offline.py
│   └── data_processor.py
├── ObjDetARec/
│   ├── Dataobj_proc/
│   │   ├── loader.py
│   │   └── parse.py
│   ├── Model/
│   │   ├── BasicBlock.py
│   │   ├── DetectModel.py
│   │   ├── build_loss.py
│   │   └── State/detect.pth
│   ├── Train/detect.py
│   ├── yolo/
│   │   ├── images/
│   │   ├── labels/
│   │   └── proc_labels/
│   ├── friend.py
│   └── main.py
├── detr-resnet-50/           # 本地 DETR 基础权重
├── detr-cow-finetuned/       # 本地微调 DETR 权重
├── sam/                      # 本地 SAM 权重
├── train_detr.py
├── test_detr.py
├── test_sam.py
├── test_shape.py
├── cow.jpg
└── result.jpg
```

## 3. 关键模块说明

### 3.1 `FaceNetPack`

#### `FaceNetPack/Model/Backbone.py`

- `CowResNet` 接收 4 通道输入。
- 网络由多层自定义残差块组成，融合中层和高层特征后输出 512 通道特征图。
- 按 `test_shape.py` 的输入设定（`1 x 4 x 424 x 240`），最终特征图空间尺寸会落到 `14 x 8`，因此后续 `ViT(count=112)` 中的 `112` 对应 `14 x 8 = 112` 个 token。

#### `FaceNetPack/Model/VisionTransformer.py`

- 将 `CowResNet` 输出的特征图展平为 token 序列。
- 增加 `cls token` 与位置编码后，使用多层 Transformer Encoder 提取全局表示。
- `return_local=True` 时还会返回局部特征图，供局部分块 ArcFace 损失使用。

#### `FaceNetPack/Model/ArcFace.py`

- `ArcFaceLoss`：全局 embedding 的分类损失。
- `LocalSplitArcFaceLoss`：把局部特征沿高度方向切分后分别计算 ArcFace。
- `SupConLoss`：监督对比损失，可与 ArcFace 联合训练。
- `build_optim`：按是否衰减参数拆分 `AdamW` 参数组。

#### `FaceNetPack/data_processor.py`

- `read_dataset()` 会扫描每个身份目录下的 `color/` 和 `depth/` 子目录。
- color/depth 配对规则基于文件名末尾数字，例如 `xxx_12.png` 会抽取尾部 `12` 作为匹配键。
- `_load_pair()` 会把 RGB 和 depth 拼成 4 通道张量，并进行同步 resize/增强/归一化。
- `dataset` 在训练阶段返回 `(image, person_id)`，在验证阶段返回 `(img1, img2, same_or_not)`。
- `Process.loader()` 会创建适配 DDP 的 `DistributedSampler`。

#### `FaceNetPack/crop_offline.py`

- 作用是离线检测 color 图，再把同一个框同步裁到 color/depth 两张图上。
- 支持检测框外扩、批量加载、多进程裁图保存。
- 当前默认优先加载 `detr-cow-finetuned/`，不存在时回退到 `detr-resnet-50/`。
- 如果你要拿它做人脸裁剪，需要把这里加载的检测权重替换成真正的“人脸检测器”，否则默认配置并不等价于 face detector。

#### `FaceNetPack/cloud_train.py`

- 训练主流程包含：
  - DDP 初始化
  - `CowResNet + ViT` 识别模型
  - 全局 ArcFace + 局部 ArcFace + 可选 SupCon 联合训练
  - 基于余弦相似度的验证、阈值搜索和检索式 top-k 评估
  - TensorBoard 记录
- 默认会把最优模型保存到 `FaceNetPack/State/cloud_model.pth`，日志写到 `tf-logs/cloud_model/`。

### 3.2 `ObjDetARec`

#### `ObjDetARec/Dataobj_proc/parse.py`

- 从 XML 中提取 `robndbox` 信息。
- 输出的每一行格式是：

```python
(1, cx, cy, w, h, angle)
```

- 解析结果会写入 `ObjDetARec/yolo/proc_labels/`。

#### `ObjDetARec/Dataobj_proc/loader.py`

- 读取 `yolo/images/` 与 `yolo/proc_labels/`。
- 图像会被 resize 到固定分辨率，并标准化到 `[-1, 1]` 附近。
- 单张图默认最多读取 3 个 anchor 对应的标注，多余标注会被截断，不足则补零。

#### `ObjDetARec/Model/DetectModel.py`

- `DetectNeck` 提取多尺度特征并做自顶向下融合。
- `DetectHead` 在多个尺度上预测 `objectness + cx + cy + w + h + angle`。
- `parse_anchor()` 会把网络输出解码为旋转框，并使用 `mmcv.ops.nms_rotated` 做 NMS。

#### `ObjDetARec/Model/build_loss.py`

- 基于 anchor 匹配构造监督信号。
- 损失由四部分组成：
  - objectness focal BCE
  - 坐标 `xy` 的 Huber loss
  - 尺寸 `wh` 的 Huber loss
  - 角度的三角形式回归损失

#### `ObjDetARec/main.py`

- 是自定义旋转框检测器的训练入口。
- 支持 DDP 和断点权重加载。
- 训练完成后会把最优结果写到 `ObjDetARec/Model/State/detect.pth`。

### 3.3 根目录脚本

#### `train_detr.py`

- 基于本地 `detr-resnet-50/` 权重微调单类别 DETR。
- 会读取 `ObjDetARec/yolo/images/` 与 XML 标注目录中的同名文件。
- 同时兼容普通 `bndbox` 和旋转框 `robndbox`。
- 若读到旋转框，会先转换成轴对齐 `xyxy` 再喂给 DETR。
- 当前类别映射固定为：

```python
id2label = {0: "cow"}
label2id = {"cow": 0}
```

#### `test_detr.py`

- 默认读取 `cow.jpg`。
- 如果存在 `detr-cow-finetuned/`，优先加载微调后的权重；否则回退到 `detr-resnet-50/`。
- 推理后会做一次 `torchvision.ops.nms`，并把检测框画到 `result.jpg`。

#### `test_sam.py`

- 默认加载 `sam/` 中的权重。
- 输入图像优先使用 `animal.jpg`，否则使用 `cow.jpg`。
- bbox 提示框是脚本里写死的：

```python
box = np.array([195.6, 268.9, 838.6, 512.5], dtype=np.float32)
```

- 适合做最小可运行演示，不是通用推理脚本。

## 4. 数据组织方式

### 4.1 RGB-D 识别数据

`FaceNetPack/data_processor.py` 期望的数据大致如下：

```text
Dataset/
└── 325_crop/
    ├── person_a/
    │   ├── color/
    │   │   ├── frame_1.png
    │   │   └── frame_2.png
    │   └── depth/
    │       ├── frame_1.png
    │       └── frame_2.png
    └── person_b/
        ├── color/
        └── depth/
```

如果没有 `Dataset/325_crop/`，代码会退回尝试读取 `Dataset/325/`。

配对逻辑要求：

- 每个身份目录下同时存在 `color/` 和 `depth/`
- color/depth 文件名末尾数字一致
- 同一身份下至少有 1 对图像，验证时最好多于 1 对

### 4.2 目标检测数据

`ObjDetARec/` 当前仓库里已经带有一份数据，目录形式如下：

```text
ObjDetARec/yolo/
├── images/       # 图像
├── labels/       # XML 标注
└── proc_labels/  # parse.py 处理后的文本标注
```

目前仓库内可以看到：

- `images/` 下约 1583 张图片
- `proc_labels/` 下约 1583 个解析后的标签文件

`labels/` 中是 XML 标注，`train_detr.py` 直接读取这里的 XML；`ObjDetARec/main.py` 则读取 `proc_labels/`。

## 5. 环境依赖

建议使用独立虚拟环境，并优先使用 `python3`。

### 5.1 基础依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install transformers pillow numpy tqdm matplotlib opencv-python tensorboard timm safetensors
```

### 5.2 仅自定义旋转框检测器额外需要

```bash
pip install mmcv
```

注意：

- `ObjDetARec` 依赖 `mmcv.ops.nms_rotated` 和 `box_iou_rotated`，因此 `mmcv` 必须与本机 `PyTorch/CUDA` 版本匹配。
- 仓库中的 `.bin` 和 `.safetensors` 由 Git LFS 管理；如果你是新克隆仓库，记得执行：

```bash
git lfs pull
```

## 6. 常见使用方式

### 6.1 解析 `ObjDetARec` 的旋转框 XML

```bash
python3 ObjDetARec/Dataobj_proc/parse.py
```

运行后会在 `ObjDetARec/yolo/proc_labels/` 生成文本标签。

### 6.2 微调 DETR

```bash
python3 train_detr.py \
  --model-path ./detr-resnet-50 \
  --image-dir ./ObjDetARec/yolo/images \
  --label-dir ./ObjDetARec/yolo/labels \
  --output-dir ./detr-cow-finetuned \
  --epochs 20 \
  --batch-size 4
```

可选参数：

- `--freeze-backbone`：冻结 backbone
- `--val-ratio`：验证集比例
- `--num-workers`：DataLoader worker 数

### 6.3 运行 DETR 推理示例

```bash
python3 test_detr.py
```

默认行为：

- 输入 `cow.jpg`
- 输出 `result.jpg`
- 终端打印类别、分数和框坐标

### 6.4 运行 SAM 推理示例

```bash
python3 test_sam.py
```

脚本会读取本地 SAM 权重，并对硬编码 bbox 执行一次分割演示。

### 6.5 离线裁切 RGB-D 数据

为了避免路径歧义，建议显式传参：

```bash
python3 FaceNetPack/crop_offline.py \
  --src ./FaceNetPack/Dataset/325 \
  --dst ./FaceNetPack/Dataset/325_crop \
  --batch-size 64 \
  --crop-workers 4 \
  --threshold 0.1
```

如果你要做人脸裁剪，请先把脚本里的检测权重换成真正的人脸检测模型。

### 6.6 训练自定义旋转框检测器

在 CUDA 环境下，这个脚本会主动初始化 DDP，因此建议通过 `torchrun` 启动：

```bash
torchrun --nproc_per_node=1 ObjDetARec/main.py --epochs 10
```

多卡时把 `1` 改成 GPU 数量即可。

### 6.7 训练 RGB-D 识别模型

理论入口是：

```bash
torchrun --nproc_per_node=1 FaceNetPack/cloud_train.py
```

但在当前仓库状态下，建议先阅读下面的“已知问题”一节并修正路径/导入后再运行。

## 7. 训练与输出文件

### 7.1 DETR

- 输入权重：`detr-resnet-50/`
- 微调输出：`detr-cow-finetuned/`
- 示例结果图：`result.jpg`

### 7.2 自定义旋转框检测器

- 默认权重文件：`ObjDetARec/Model/State/detect.pth`

### 7.3 RGB-D 识别

- 最优权重：`FaceNetPack/State/cloud_model.pth`
- TensorBoard 日志：`tf-logs/cloud_model/`

## 8. 已知问题与注意事项

下面这些点是阅读代码后比较值得优先注意的地方：

1. `FaceNetPack/cloud_train.py` 和 `FaceNetPack/Model/MarginModel.py`
   - 当前导入的是 `FaceNetPack.data_processor.img2img`
   - 但仓库里实际存在的是 `FaceNetPack/data_processor.py`
   - 也就是说，这两个脚本在当前目录结构下需要先改导入路径才能运行

2. `FaceNetPack/crop_offline.py` 与 `FaceNetPack/data_processor.py` 的默认数据根目录不一致
   - `crop_offline.py` 默认读写的是 `FaceNetPack/Dataset/...`
   - `data_processor.py` 默认读取的是仓库根目录下的 `Dataset/...`
   - 实际使用时最好统一目录，或者始终显式传入路径

3. `crop_offline.py` 的说明与默认权重并不完全一致
   - 脚本注释写的是“离线人脸检测与裁切”
   - 但默认加载的权重是 `detr-cow-finetuned/` 或 `detr-resnet-50/`
   - 如果目标是做人脸裁剪，需要替换成真正的 face detector

4. `ObjDetARec/main.py` 在检测到 CUDA 可用时会直接初始化 `torch.distributed`
   - 因此不要在有 CUDA 的环境里直接 `python3 ObjDetARec/main.py`
   - 更稳妥的方式是用 `torchrun`

5. `test_sam.py` 不是通用脚本
   - 输入框、图片路径都写死在脚本里
   - 适合验证权重加载与基本推理链路，不适合直接当成批量推理工具

## 9. 推荐阅读顺序

如果你是第一次接手这个仓库，推荐按下面顺序看代码：

1. 先看 `README.md`，了解全局结构
2. 看 `FaceNetPack/data_processor.py`，搞清楚 RGB-D 数据是怎样被组织和增强的
3. 看 `FaceNetPack/Model/Backbone.py` 和 `FaceNetPack/Model/VisionTransformer.py`，理解主模型结构
4. 看 `FaceNetPack/Model/ArcFace.py`，理解训练目标
5. 再看 `FaceNetPack/cloud_train.py`，串起整个识别训练流程
6. 若关注检测分支，再看 `ObjDetARec/` 与 `train_detr.py`

## 10. 一句话总结

这个仓库的核心价值不在于“已经封装好的完整产品”，而在于它把一套 RGB-D 识别实验、旋转框检测实验以及 DETR/SAM 验证脚本放在了同一个工作区里。只要先把数据路径、依赖环境和少量导入问题理顺，就能很快继续往下扩展和整理。
