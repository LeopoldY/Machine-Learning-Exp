import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch
import torch.nn as nn  

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential( # 3x128x128
            nn.Conv2d(3, 32, 3, 1, 1),  
            nn.ReLU(), 
            nn.MaxPool2d(2) 
        )  # 32x64x64
        self.conv2 = nn.Sequential( # 32x64x64
            nn.Conv2d(32, 64, 3, 1, 1),  
            nn.ReLU(),
            nn.MaxPool2d(2)  
        ) # 64x32x32
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),  
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  
        res = self.out(conv3_out)  
        res = F.interpolate(res, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = torch.sigmoid(res)
        return res

class USegNet(nn.Module):
    def __init__(self):
        super(USegNet, self).__init__()
        # 定义编码器网络的各个层
        self.encoder1 = nn.Sequential( # 3x128x128
            nn.Conv2d(3, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(2) 
        )  # 64x64x64
        self.encoder2 = nn.Sequential( # 64x64x64
            nn.Conv2d(64, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        ) # 128x32x32
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        ) # 256x16x16
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        ) # 512x8x8
        # 定义解码器网络的各个层
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2), # 256x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2), # 128x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2), # 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2), # 32x128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 定义最后一层卷积，输出每个像素的类别概率
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, x):
        # 获取编码器网络的各个层的输出
        encoder1_out = self.encoder1(x)
        encoder2_out = self.encoder2(encoder1_out)
        encoder3_out = self.encoder3(encoder2_out)  
        encoder4_out = self.encoder4(encoder3_out)  
        # 获取解码器网络的各个层的输出，并使用跳跃连接
        decoder4_out = self.decoder4(encoder4_out) + encoder3_out
        decoder3_out = self.decoder3(decoder4_out) + encoder2_out
        decoder2_out = self.decoder2(decoder3_out) + encoder1_out
        decoder1_out = self.decoder1(decoder2_out) 
        # 获取最终的输出
        res = self.out(decoder1_out)  
        res = torch.sigmoid(res)
        return res
