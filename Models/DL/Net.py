import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn  

class ConvolutionNet(nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
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