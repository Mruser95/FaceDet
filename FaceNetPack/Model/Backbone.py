import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_ch), 
            nn.BatchNorm2d(in_ch), 
            nn.GELU(), 
            nn.Dropout2d(0.05),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BasicBlock(nn.Module):
    expansion = 4
    def __init__(self, in_ch, min_ch, stride=1, downsample=None):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv3x3(in_ch, min_ch, stride=stride), 
            nn.BatchNorm2d(min_ch), 
            nn.GELU(), 
            Conv3x3(min_ch, min_ch, stride=1), 
            nn.BatchNorm2d(min_ch), 
            nn.GELU(), 
            nn.Conv2d(min_ch, min_ch * BasicBlock.expansion, 1), 
            nn.BatchNorm2d(min_ch * BasicBlock.expansion)
        ])
        self.downsample = downsample
    
    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        for layer in self.layers:
            x = layer(x)
        return F.gelu(x + identity)
    

class ResNet(nn.Module):
    def __init__(self, layer_list, ch_list, stride_list, in_channels=4):
        super().__init__()
        if len(ch_list) < 4:
            raise ValueError("ch_list must contain at least 4 stages for layer fusion.")
        self.in_ch = 32
        in_ch = self.in_ch
        mid = len(ch_list) // 2
        self.fusion_mid = mid
        self.fusion_last = len(ch_list) - 1
        fused_in_ch = (ch_list[mid] + ch_list[-1]) * BasicBlock.expansion
        
        self.pre_layer = nn.ModuleList([
            nn.Conv2d(in_channels, in_ch, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(in_ch), 
            nn.GELU(), 
            nn.MaxPool2d(3, stride=2, padding=1)
        ])
        
        self.layers = nn.ModuleList()
        for layer_n, min_ch, stride in zip(layer_list, ch_list, stride_list):
            self.layers.append(self._make_layer(min_ch, layer_n, stride))
        
        self.end_layer = nn.ModuleList([
            Conv3x3(fused_in_ch, 1024, stride=1), 
            nn.BatchNorm2d(1024), 
            nn.GELU(), 
            Conv3x3(1024, 512, stride=1), 
            nn.BatchNorm2d(512), 
            nn.GELU(), 
        ])

    def _make_layer(self, min_ch, blocks, stride):
        downsample = None
        out_ch = min_ch * BasicBlock.expansion
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride), 
                nn.BatchNorm2d(out_ch)
            )
        layers = nn.ModuleList()
        
        layers.append(BasicBlock(self.in_ch, min_ch, stride, downsample))
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, min_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        for layer in self.pre_layer:
            x = layer(x)
        fused_features = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == self.fusion_mid or idx == self.fusion_last:
                fused_features[idx] = x
        feat_mid = fused_features[self.fusion_mid]
        feat_last = fused_features[self.fusion_last]
        feat_last_up = F.interpolate(feat_last, size=feat_mid.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_mid, feat_last_up], dim=1)
        for layer in self.end_layer:
            x = layer(x)
        return x
    

class CowResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        layer_list =  [2, 2, 2, 2, 2, 2]
        ch_list =     [64, 128, 128, 256, 128, 128]
        stride_list = [2, 2, 1, 2, 1, 2]
        self.layer = ResNet(layer_list, ch_list, stride_list)

    def forward(self, x):
        return self.layer(x)

