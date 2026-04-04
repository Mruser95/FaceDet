import torch
from FaceNetPack.Model.Backbone import CowResNet

model = CowResNet()
x = torch.randn(1, 4, 424, 240)

# pre_layer
for layer in model.layer.pre_layer:
    x = layer(x)
print(f"After pre_layer: {x.shape}")

# layers
for idx, layer in enumerate(model.layer.layers):
    x = layer(x)
    print(f"After layer {idx}: {x.shape}")
