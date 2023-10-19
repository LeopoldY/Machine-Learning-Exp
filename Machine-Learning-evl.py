import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim


# 加载原图和掩码图
image_path = "./Resorce/images/img10.jpg"
mask_path = "./Resorce/maskPic/SegmentationClassPNG/img10.png"
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

# 将原图和掩码图转换为一维数组
X = image.reshape(-1, 3)
# 通过掩码图中的像素值判断该像素是否属于目标类别
y = np.array([[0 for i in range(mask.shape[1])] for j in range(mask.shape[0])])
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 128:
            y[i][j] = 1
        else:
            y[i][j] = 0
y = y.flatten()
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
lr = nn.Sequential(
    nn.Linear(X_train.shape[1], 1),
    nn.Sigmoid()
)
lr.load_state_dict(torch.load('lr.pth'))

# 在GPU上验证模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr.to(device)

# 可视化分割结果
redMask = np.array([[[0, 0, 255] for i in range(mask.shape[1])] for j in range(mask.shape[0])])
X_torch = torch.from_numpy(X).float().to(device)
mask_pred_lr = lr(X_torch).squeeze().cpu().detach().numpy().reshape(image.shape[0], image.shape[1])   # 将预测结果转换为二维数组

# 二值化处理
mask_pred_lr[np.where(mask_pred_lr >= 0.3)] = 1
mask_pred_lr[np.where(mask_pred_lr < 0.3)] = 0

mask_pred_lr = np.uint8(mask_pred_lr * 255) # 将预测结果转换为0-255的灰度图

cv2.imwrite("img10_pred.jpg", mask_pred_lr) # 将预测结果保存为图片

mask_pred_lr = cv2.cvtColor(mask_pred_lr, cv2.COLOR_GRAY2BGR) # 将灰度图转换为三通道图像

mask_pred_lr[np.where((mask_pred_lr == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # 将预测结果中的目标类别像素标记为红色

result_lr = cv2.addWeighted(image, 0.5, mask_pred_lr, 0.5, 0) # 将原图和预测结果融合

cv2.imshow("Logistic Regression", result_lr) # 显示融合后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()