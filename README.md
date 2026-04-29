# FaceDet

以 RGB-D 视觉识别为主线的实验项目：把 RGB 图像和深度图对齐拼成 4 通道输入，做识别训练与检索评估；同时保留了基于 DETR 的检测微调，用来给主线做数据裁切。

## 亮点

- **RGB + Depth 4 通道**输入，自定义 `CowResNet` 多尺度特征 + `ViT` 全局编码。
- 损失组合：`ArcFaceLoss` + `LocalSplitArcFaceLoss`（局部分块）+ 可选 `SupConLoss`。
- 验证：阈值扫描的二分类相似度 + 余弦相似度 top-k 检索 + ROC。
- **数据预处理流水线**：SAM 前景分割 → DETR 检测裁切 → 质量清洗。

## 目录结构

```text
FaceDet/
├── FaceNetPack/                  # RGB-D 识别主线 + 数据预处理
│   ├── sam_processor.py          #   SAM 前景分割
│   ├── crop_offline.py           #   DETR 检测裁切
│   ├── clean_crop_data.py        #   裁切质量清洗
│   ├── data_processor.py         #   训练数据加载
│   ├── cloud_train.py            #   分布式训练入口
│   ├── xgboost_verifier.py       #   embedding + XGBoost 验证器
│   ├── Model/
│   │   ├── Backbone.py           #     CowResNet (4ch → 512ch)
│   │   ├── VisionTransformer.py  #     ViT 全局编码
│   │   ├── ArcFace.py            #     损失函数集合
│   │   └── MarginModel.py        #     阈值分析 & ROC
│   └── Dataset/                  #   数据 (gitignored)
├── data/yolo/                    # DETR 微调用 images / labels (gitignored)
├── detr-resnet-50/               # DETR 基础权重
├── detr-cow-finetuned/           # DETR 微调输出
├── train_detr.py / test_detr.py
├── requirements.txt
└── README.md
```

数据约定：`Dataset/` 下按身份分文件夹，每个身份内有 `color/` 和 `depth/`，文件名末尾编号一致，代码自动配对。读取优先级：`325_crop` > `325_sam` > `325`。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

新克隆的仓库还需要：`git lfs pull`。

### 数据预处理

```bash
# SAM 前景分割
python -m FaceNetPack.sam_processor --src Dataset/325 --dst Dataset/325_sam --device cuda
# DETR 检测裁切
python -m FaceNetPack.crop_offline --src Dataset/325_sam --dst Dataset/325_crop
# 质量清洗（默认仅扫描，加 --delete 真删）
python -m FaceNetPack.clean_crop_data [--delete]
```

### 训练 / 评估

```bash
# 训练 RGB-D 识别模型
torchrun --nproc_per_node=1 FaceNetPack/cloud_train.py
# 阈值分析 & ROC
python -m FaceNetPack.Model.MarginModel
```

### DETR 微调与推理

```bash
python3 train_detr.py \
  --model-path ./detr-resnet-50 \
  --image-dir ./data/yolo/images \
  --label-dir ./data/yolo/labels \
  --output-dir ./detr-cow-finetuned

python3 test_detr.py
```
