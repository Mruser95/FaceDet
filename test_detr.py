from pathlib import Path

from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import nms

# 优先使用微调后的模型；如果不存在，再回退到原始预训练模型
finetuned_path = Path("./detr-cow-finetuned")
base_path = Path("./detr-resnet-50")
model_path = finetuned_path if finetuned_path.exists() else base_path
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
score_threshold = 0.1
nms_iou_threshold = 0.4

processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
model.eval()

image = Image.open("cow.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
    outputs = model(**inputs)

# 后处理
results = processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=score_threshold
)[0]

if len(results["scores"]) > 0:
    keep = nms(results["boxes"].cpu(), results["scores"].cpu(), nms_iou_threshold)
    results = {k: v[keep] for k, v in results.items()}

# 打印结果
print(f"loaded model from: {model_path}")
print(f"detections after nms: {len(results['scores'])}")
if len(results["scores"]) == 0:
    print("no detections")
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(
        model.config.id2label[label.item()],
        round(score.item(),3),
        [round(x,1) for x in box.tolist()]
    )


draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if model.config.id2label[label.item()] == "cow":
        x1,y1,x2,y2 = box.tolist()
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)

image.save("result.jpg")