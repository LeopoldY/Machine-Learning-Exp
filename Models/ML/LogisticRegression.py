import torch
import torch.nn as nn
import torch.optim as optim

class logisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(logisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        return out
    
lr_loss_fn = nn.BCELoss() # 二分类交叉熵损失函数
