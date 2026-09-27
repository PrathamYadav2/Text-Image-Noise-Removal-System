"""
U-Net Architecture for Text Image Denoising
============================================
Encoder-Decoder with skip connections to preserve fine text details.
Input/Output: Grayscale images (1 channel), 256x256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsample: MaxPool2d → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample → concatenate skip → DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if sizes differ slightly
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for grayscale document denoising.

    Architecture:
        Input  (1, 256, 256)
        Enc1   (64,  256, 256)  ──────────────────────────────────────┐ skip1
        Enc2   (128, 128, 128)  ────────────────────────────────┐ skip2│
        Enc3   (256,  64,  64)  ──────────────────────────┐ skip3│     │
        Enc4   (512,  32,  32)  ──────────────────────┐ skip4│    │     │
        Bottleneck (1024, 16, 16)                     │     │    │     │
        Dec4   (512,  32,  32)  ◄──────────────── skip4     │    │     │
        Dec3   (256,  64,  64)  ◄─────────────────────── skip3   │     │
        Dec2   (128, 128, 128)  ◄──────────────────────────── skip2    │
        Dec1   (64,  256, 256)  ◄─────────────────────────────────skip1│
        Output (1,  256, 256)   sigmoid → [0, 1]
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)          # 256 → 256
        self.enc2 = Down(f, f * 2)                      # 256 → 128
        self.enc3 = Down(f * 2, f * 4)                  # 128 → 64
        self.enc4 = Down(f * 4, f * 8)                  # 64  → 32

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)           # 32  → 16

        # Decoder
        self.dec4 = Up(f * 16, f * 8)                   # 16  → 32
        self.dec3 = Up(f * 8,  f * 4)                   # 32  → 64
        self.dec2 = Up(f * 4,  f * 2)                   # 64  → 128
        self.dec1 = Up(f * 2,  f)                       # 128 → 256

        # Output head
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder path with skip connections
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return torch.sigmoid(self.out_conv(d1))


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LightUNet(UNet):
    """
    Lightweight U-Net for fast CPU training (~500K params vs 31M).
    Same architecture but base_features=16 instead of 64.
    Trains in minutes on CPU. Good for quick experiments.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_features=16)


if __name__ == "__main__":
    print("=== Full U-Net ===")
    model = UNet()
    x = torch.randn(1, 1, 128, 128)
    print(f"Parameters: {count_parameters(model):,}")

    print("\n=== Light U-Net (CPU-friendly) ===")
    light = LightUNet()
    out = light(x)
    print(f"Input : {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {count_parameters(light):,}")
