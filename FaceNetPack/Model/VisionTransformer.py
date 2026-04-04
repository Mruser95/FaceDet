import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


class DropPathTransformerEncoderLayer(nn.Module):
    def __init__(self, base_layer: nn.TransformerEncoderLayer, drop_path_rate=0.0):
        super().__init__()
        self.base = base_layer
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        sa_out = self.base.self_attn(
            self.base.norm1(x), self.base.norm1(x), self.base.norm1(x),
            attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop_path(self.base.dropout1(sa_out))
        ff_out = self.base.linear2(self.base.dropout(self.base.activation(self.base.linear1(self.base.norm2(x)))))
        x = x + self.drop_path(self.base.dropout2(ff_out))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        count,
        emb_dim=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.05,
        drop_path_rate=0.0,
    ):
        super().__init__()
        if count <= 0:
            raise ValueError("count must be positive.")
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads.")

        self.num_patches = count
        self.emb_dim = emb_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(emb_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        layers = []
        for i in range(num_layers):
            base_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=int(emb_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            layers.append(DropPathTransformerEncoderLayer(base_layer, drop_path_rate=dpr[i]))
        self.transformer = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(emb_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_local: bool = False):
        if x.ndim != 4:
            raise ValueError("Input must have shape [B, C, H, W].")

        B, C, H, W = x.shape
        if C != self.emb_dim:
            raise ValueError(
                f"Input channel dim {C} must equal emb_dim {self.emb_dim}."
            )
        if H * W != self.num_patches:
            raise ValueError(
                f"Input token count {H * W} must equal count {self.num_patches}."
            )

        x = x.flatten(2).transpose(1, 2)
        x = self.input_norm(x)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        out = x
        for layer in self.transformer:
            out = layer(out)
        out = self.norm(out)
        
        global_feat = out[:, 0]
        
        if not return_local:
            return global_feat
            
        # 恢复局部tokens为原图(特征图)尺寸: [B, C, H, W]
        local_feat = out[:, 1:].transpose(1, 2).reshape(B, self.emb_dim, H, W)
        return global_feat, local_feat
