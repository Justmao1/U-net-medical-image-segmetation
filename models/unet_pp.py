import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """两次 [Conv3x3 -> BN -> ReLU]"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    """U-Net++ with dense nested skip connections, 输出 logits（不含 Sigmoid）

    完整嵌套结构（深度 L=5）：
        x1_1 -> x1_2 -> x1_3 -> x1_4 -> x1_5
        x2_1 -> x2_2 -> x2_3 -> x2_4
        x3_1 -> x3_2 -> x3_3
        x4_1 -> x4_2
        x5_1
    """

    def __init__(self, in_ch=1, out_ch=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        nb_filter = filters

        # Encoding path
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_ch, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # Nested nodes (dense skip connections)
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # Output convolutions
        self.final1 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoding
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Level 1 (direct skip)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], dim=1))

        # Level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], dim=1))

        # Level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], dim=1))

        # Level 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], dim=1))

        # Deep supervision: output from all levels
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]

        # Standard output (only the deepest path)
        out = self.final4(x0_4)
        return out
