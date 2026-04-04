from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import SamModel, SamProcessor


model_path = Path("./sam")
image_path = Path("animal.jpg") if Path("animal.jpg").exists() else Path("cow.jpg")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 1 加载 Hugging Face 版 SAM
processor = SamProcessor.from_pretrained(model_path, use_fast=False)
model = SamModel.from_pretrained(model_path).to(device)
model.eval()

# 2 读取图片
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"未找到图片: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3 你的 bbox
box = np.array([195.6, 268.9, 838.6, 512.5], dtype=np.float32)

# Hugging Face SAM 需要 batch 维和 box 维
inputs = processor(
    images=image,
    input_boxes=[[box.tolist()]],
    return_tensors="pt",
)
inputs = {
    k: (v.to(device=device, dtype=torch.float32) if torch.is_floating_point(v) else v.to(device))
    for k, v in inputs.items()
}

# 4 预测 mask
with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu(),
)
mask = masks[0][0][0].numpy()

# 5 显示 mask
plt.imshow(image)
plt.imshow(mask, alpha=0.5)
plt.axis("off")
plt.show()