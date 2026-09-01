"""Inference-only RDAH-Net architecture, kept checkpoint-key compatible."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, value):
        return self.act(self.bn(self.conv(value)))


class BlockAttention(nn.Module):
    def __init__(self, dim, num_heads=4, block_size=8, mlp_dim=None, dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.mlp_dim = mlp_dim or dim * 2
        self.scale = self.head_dim**-0.5
        self.local_proj = ConvLayer(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.mlp_dim, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        batch, channels, height, width = value.shape
        if height % self.block_size or width % self.block_size:
            raise ValueError(f"Feature size {(height, width)} must be divisible by {self.block_size}")
        local = self.local_proj(value)
        blocks_h, blocks_w = height // self.block_size, width // self.block_size
        count = blocks_h * blocks_w
        blocked = value.reshape(
            batch, channels, blocks_h, self.block_size, blocks_w, self.block_size
        ).permute(0, 2, 4, 1, 3, 5)
        blocked = blocked.reshape(batch * count, channels, self.block_size, self.block_size)
        block_batch, block_channels, block_h, block_w = blocked.shape
        tokens = block_h * block_w
        qkv = self.qkv(blocked).reshape(block_batch, 3, block_channels, tokens).permute(1, 0, 2, 3)
        query, key, val = [
            item.reshape(block_batch, self.num_heads, self.head_dim, tokens) for item in qkv
        ]
        attention = torch.einsum("bhdn,bhdm->bhnm", query, key) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = torch.einsum("bhnm,bhdm->bhdn", attention, val)
        output = output.contiguous().reshape(block_batch, block_channels, block_h, block_w)
        output = self.proj_drop(self.proj(output))
        output = output.reshape(
            batch, blocks_h, blocks_w, channels, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)
        value = value + local + output
        return value + self.mlp(value)


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_heads=4, block_size=8, dropout=0.0):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        self.attention = BlockAttention(out_channels, num_heads, block_size, dropout=dropout)
        self.conv2 = ConvLayer(out_channels, out_channels, groups=out_channels)

    def forward(self, value):
        return self.conv2(self.attention(self.conv1(value)))


class MobileViT_S_Light(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.stem = ConvLayer(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            MobileViTBlock(32, 64, stride=2, num_heads=4, block_size=8),
            MobileViTBlock(64, 64, stride=1, num_heads=4, block_size=8),
        )
        self.stage2 = nn.Sequential(
            MobileViTBlock(64, 128, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(128, 128, stride=1, num_heads=8, block_size=8),
        )
        self.stage3 = nn.Sequential(
            MobileViTBlock(128, 256, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(256, 256, stride=1, num_heads=8, block_size=8),
        )
        self.proj1 = ConvLayer(64, 32, kernel_size=1, padding=0)
        self.proj2 = ConvLayer(128, 32, kernel_size=1, padding=0)
        self.proj3 = ConvLayer(256, 32, kernel_size=1, padding=0)

    def forward(self, value):
        value = self.stem(value)
        feat1 = self.stage1(value)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        return [self.proj1(feat1), self.proj2(feat2), self.proj3(feat3)]


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value):
        channel_attention = self.sigmoid(self.fc(self.avg_pool(value)) + self.fc(self.max_pool(value)))
        value = value * channel_attention
        average = torch.mean(value, dim=1, keepdim=True)
        maximum = torch.max(value, dim=1, keepdim=True).values
        return value * self.sigmoid(self.spatial(torch.cat((average, maximum), dim=1)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, H=64, W=64):
        super().__init__()
        self.d_model = d_model
        pos_x = torch.arange(W, dtype=torch.float32).repeat(H, 1)
        pos_y = torch.arange(H, dtype=torch.float32).repeat(W, 1).t()
        pos = torch.stack((pos_x, pos_y), dim=0)
        encoding = torch.zeros(1, d_model, H, W)
        divisor = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding[0, ::2] = torch.sin(pos[0:1] * divisor[None, :, None, None])
        encoding[0, 1::2] = torch.cos(pos[1:2] * divisor[None, :, None, None])
        self.register_buffer("pe", encoding)

    def forward(self, value):
        return value + self.pe[:, : value.size(1), : value.size(2), : value.size(3)]


class LightCrossAttention(nn.Module):
    def __init__(self, d_model=32, num_heads=4, block_size=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.proj_q = nn.Conv2d(d_model, d_model, 1)
        self.proj_k = nn.Conv2d(d_model, d_model, 1)
        self.proj_v = nn.Conv2d(d_model, d_model, 1)
        self.proj_out = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, q, k, v):
        batch, channels, height, width = q.shape
        if height % self.block_size or width % self.block_size:
            raise ValueError(f"Feature size {(height, width)} must be divisible by {self.block_size}")
        original = q
        blocks_h, blocks_w = height // self.block_size, width // self.block_size
        count = blocks_h * blocks_w

        def blockify(value):
            return value.reshape(
                batch, channels, blocks_h, self.block_size, blocks_w, self.block_size
            ).permute(0, 2, 4, 1, 3, 5).reshape(
                batch * count, channels, self.block_size, self.block_size
            )

        q, k, v = blockify(q), blockify(k), blockify(v)
        block_batch, block_channels, block_h, block_w = q.shape
        tokens = block_h * block_w
        q = self.proj_q(q).reshape(block_batch, self.num_heads, self.head_dim, tokens)
        k = self.proj_k(k).reshape(block_batch, self.num_heads, self.head_dim, tokens)
        v = self.proj_v(v).reshape(block_batch, self.num_heads, self.head_dim, tokens)
        attention = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attention = self.dropout(attention.softmax(dim=-1))
        output = torch.einsum("bhnm,bhdm->bhdn", attention, v)
        output = self.proj_out(output.contiguous().reshape(block_batch, block_channels, block_h, block_w))
        output = output.reshape(
            batch, blocks_h, blocks_w, channels, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)
        return self.norm(output + original)


class LightTransformerBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, hidden_dim=64, block_size=8, dropout=0.1):
        super().__init__()
        self.self_attn = LightCrossAttention(d_model, num_heads, block_size, dropout)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, d_model, 1),
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        value = self.self_attn(value, value, value)
        return value + self.dropout(self.ffn(self.norm(value)))


class HeightPredTransformer(nn.Module):
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth_encoder = MobileViT_S_Light(in_channels=1)
        self.img_encoder = MobileViT_S_Light(in_channels=3)
        self.cbam_blocks = nn.ModuleList([CBAM(d_model), CBAM(d_model), CBAM(d_model)])
        self.cross_attn_blocks = nn.ModuleList(
            [LightCrossAttention(d_model, num_heads, block_size=8) for _ in range(3)]
        )
        self.rev_cross_attn_blocks = nn.ModuleList(
            [LightCrossAttention(d_model, num_heads, block_size=8) for _ in range(3)]
        )
        self.pos_encoding = PositionalEncoding(d_model, H=64, W=64)
        self.global_transformer = LightTransformerBlock(d_model, num_heads, hidden_dim=64)
        self.skip_projs = nn.ModuleList(
            [nn.Conv2d(d_model, 16, 1), nn.Conv2d(d_model, 32, 1), nn.Conv2d(d_model, 64, 1)]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, depth, image):
        depth_features = self.depth_encoder(depth)
        image_features = self.img_encoder(image)
        depth_features = [block(value) for block, value in zip(self.cbam_blocks, depth_features)]
        image_features = [block(value) for block, value in zip(self.cbam_blocks, image_features)]
        fused = []
        for index in range(3):
            forward = self.cross_attn_blocks[index](
                depth_features[index], image_features[index], image_features[index]
            )
            reverse = self.rev_cross_attn_blocks[index](
                image_features[index], depth_features[index], depth_features[index]
            )
            fused.append((forward + reverse) / 2.0)
        value = self.global_transformer(self.pos_encoding(fused[2]))
        value = self.decoder[0:4](value)
        value = value + F.interpolate(
            self.skip_projs[2](fused[2]), size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder[4:8](value)
        value = value + F.interpolate(
            self.skip_projs[1](fused[1]), size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.decoder[8:12](value)
        value = value + F.interpolate(
            self.skip_projs[0](fused[0]), size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.decoder[12:](value)


def load_rdah_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model = HeightPredTransformer()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
