import torch
import torch.nn as nn

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class U_Net_PP(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net_PP, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Encoding path
        self.Conv1_1 = conv_block_nested(in_ch, filters[0])
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2_1 = conv_block_nested(filters[0], filters[1])
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv3_1 = conv_block_nested(filters[1], filters[2])
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv4_1 = conv_block_nested(filters[2], filters[3])
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv5_1 = conv_block_nested(filters[3], filters[4])

        # Decoding path with dense skip connections
        self.Up5_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv4_2 = conv_block_nested(filters[4] + filters[3], filters[3])

        self.Up4_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv3_2 = conv_block_nested(filters[3] + filters[2], filters[2])

        self.Up3_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv2_2 = conv_block_nested(filters[2] + filters[1], filters[1])

        self.Up2_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv1_2 = conv_block_nested(filters[1] + filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.active = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        x1_1 = self.Conv1_1(x)
        x2_1 = self.Conv2_1(self.Maxpool1(x1_1))
        x3_1 = self.Conv3_1(self.Maxpool2(x2_1))
        x4_1 = self.Conv4_1(self.Maxpool3(x3_1))
        x5_1 = self.Conv5_1(self.Maxpool4(x4_1))

        # Decoding + Skip Connections
        x4_2 = self.Conv4_2(torch.cat([x4_1, self.Up5_1(x5_1)], dim=1))
        x3_2 = self.Conv3_2(torch.cat([x3_1, self.Up4_2(x4_2)], dim=1))
        x2_2 = self.Conv2_2(torch.cat([x2_1, self.Up3_2(x3_2)], dim=1))
        x1_2 = self.Conv1_2(torch.cat([x1_1, self.Up2_2(x2_2)], dim=1))

        output = self.final(x1_2)
        return self.active(output)
