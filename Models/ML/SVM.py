import torch.nn as nn
import torch.nn.functional as F

import torch

class LinearSVM(nn.Module):
    '''线性SVM模型'''
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(3, 1) # 输出层
        self.centers = nn.Parameter(torch.randn(3)) # 中心

    def forward(self, x):
        x = x.float() # 将输入x转换为浮点数类型
        x = self.linear(x) # 输出层
        res = x * self.centers # 计算距离
        res = torch.sum(res, dim=1) # 求和
        res = torch.sigmoid(res) # Sigmoid函数
        return res

class SVM(nn.Module):
    '''支持向量机模型'''
    def __init__(self, input_size, output_size, num_centers, sigma):
        super(SVM, self).__init__()
        self.linear = nn.Linear(num_centers, output_size) # 输出层
        self.centers = nn.Parameter(torch.randn(num_centers, input_size)) # 中心
        self.sigma = sigma # RBF核函数的参数

    def forward(self, x):
        x = x.float() # 将输入x转换为浮点数类型
        dist = torch.cdist(x, self.centers, p=2) # 计算距离
        rbf = torch.exp(-dist.pow(2) / (2 * self.sigma ** 2)) # RBF核函数
        return nn.Sigmoid(self.linear(rbf)) # 输出层


svm_loss_fn = nn.HingeEmbeddingLoss() # SVM损失函数
