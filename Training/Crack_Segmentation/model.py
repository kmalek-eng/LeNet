import torch
import torch.nn as nn
import torch.nn.functional as F


class UNET_MODEL(nn.Module):
    def __init__(self, args=None, channel_reduction=2):
        super().__init__()

        self.required_divisor = 16  # IMPORTANT

        c1 = int(32 / channel_reduction)
        c2 = int(64 / channel_reduction)
        c3 = int(128 / channel_reduction)
        c4 = int(256 / channel_reduction)
        c5 = int(512 / channel_reduction)
        c6 = int(1024 / channel_reduction)

        # -------- Encoder --------
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(c2)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c3, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(c3)
        self.conv4b = nn.Conv2d(c3, c3, 3, padding=1)
        self.bn4b = nn.BatchNorm2d(c3)

        self.conv5 = nn.Conv2d(c3, c4, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(c4)
        self.conv6 = nn.Conv2d(c4, c4, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(c4)
        self.conv6b = nn.Conv2d(c4, c4, 3, padding=1)
        self.bn6b = nn.BatchNorm2d(c4)

        self.conv7 = nn.Conv2d(c4, c5, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(c5)
        self.conv8 = nn.Conv2d(c5, c5, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(c5)
        self.conv8b = nn.Conv2d(c5, c5, 3, padding=1)
        self.bn8b = nn.BatchNorm2d(c5)

        # -------- Bottleneck --------
        self.conv9 = nn.Conv2d(c5, c5, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(c5)
        self.conv10 = nn.Conv2d(c5, c6, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(c6)
        self.conv11 = nn.Conv2d(c6, c6, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(c6)

        self.dropout = nn.Dropout2d(0.2)

        # -------- Decoder --------
        self.up1 = nn.ConvTranspose2d(c6, c5, 2, stride=2)
        self.conv12 = nn.Conv2d(c5 + c5, c5, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(c5)
        self.conv13 = nn.Conv2d(c5, c4, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(c4)

        self.up2 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.conv14 = nn.Conv2d(c4 + c4, c4, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(c4)
        self.conv15 = nn.Conv2d(c4, c3, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(c3)

        self.up3 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
        self.conv16 = nn.Conv2d(c3 + c3, c3, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(c3)
        self.conv17 = nn.Conv2d(c3, c2, 3, padding=1)
        self.bn17 = nn.BatchNorm2d(c2)

        self.up4 = nn.ConvTranspose2d(c2, c2, 2, stride=2)
        self.conv18 = nn.Conv2d(c2 + c2, c2, 3, padding=1)
        self.bn18 = nn.BatchNorm2d(c2)
        self.conv19 = nn.Conv2d(c2, c2, 3, padding=1)
        self.bn19 = nn.BatchNorm2d(c2)

        self.out = nn.Conv2d(c2, 1, 1)

    def forward(self, x):
        # -------- SAFE SIZE CHECK --------
        h, w = x.shape[2], x.shape[3]
        if h % self.required_divisor != 0 or w % self.required_divisor != 0:
            raise ValueError(
                f"Input size must be divisible by {self.required_divisor}. "
                f"Got ({h}, {w}). Example valid size: 448x448."
            )

        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn2b(self.conv2b(x1)))
        p1 = self.pool(x1)

        x2 = F.relu(self.bn3(self.conv3(p1)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2 = F.relu(self.bn4b(self.conv4b(x2)))
        p2 = self.pool(x2)

        x3 = F.relu(self.bn5(self.conv5(p2)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        x3 = F.relu(self.bn6b(self.conv6b(x3)))
        p3 = self.pool(x3)

        x4 = F.relu(self.bn7(self.conv7(p3)))
        x4 = F.relu(self.bn8(self.conv8(x4)))
        x4 = F.relu(self.bn8b(self.conv8b(x4)))
        p4 = self.pool(x4)

        x5 = F.relu(self.bn9(self.conv9(p4)))
        x5 = F.relu(self.bn10(self.conv10(x5)))
        x5 = F.relu(self.bn11(self.conv11(x5)))
        x5 = self.dropout(x5)

        d1 = self.up1(x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = F.relu(self.bn12(self.conv12(d1)))
        d1 = F.relu(self.bn13(self.conv13(d1)))

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = F.relu(self.bn14(self.conv14(d2)))
        d2 = F.relu(self.bn15(self.conv15(d2)))

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = F.relu(self.bn16(self.conv16(d3)))
        d3 = F.relu(self.bn17(self.conv17(d3)))

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = F.relu(self.bn18(self.conv18(d4)))
        d4 = F.relu(self.bn19(self.conv19(d4)))

        return self.out(d4)