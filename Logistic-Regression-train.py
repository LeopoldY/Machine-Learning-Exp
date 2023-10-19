import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Models.ML.LogisticRegression import logisticRegression
from Models.ML.LogisticRegression import lr_loss_fn

import torch
import torch.optim as optim
from tqdm import tqdm

# 加载原图和掩码图
image_path = "./Resorce/images/img10.jpg"
mask_path = "./Resorce/maskPic/SegmentationClassPNG/img10.png"
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

# 将原图和掩码图转换为一维数组
X = image.reshape(-1, 3)
# y = np.any(mask == [0, 0, 128], axis=-1).flatten() # 通过掩码图中的像素值判断该像素是否属于目标类别
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

# 将数据转换为PyTorch张量
X_train_torch = torch.from_numpy(X_train).float()
y_train_torch = torch.from_numpy(y_train).float()
X_test_torch = torch.from_numpy(X_test).float()

# 创建逻辑回归模型
lr = logisticRegression(3, 1)
lr_optimizer = optim.Adam(lr.parameters()) # Adam优化器

# 在GPU上训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr.to(device)

# 将数据移动到GPU上
X_train_torch = X_train_torch.to(device)
y_train_torch = y_train_torch.to(device)
X_test_torch = X_test_torch.to(device)

# 使用tqdm库创建进度条，并在每个迭代步骤中更新进度条
num_epochs = 2
batch_size = 32
n_batches = len(X_train) // batch_size
for epoch in range(num_epochs):
    lr.train() 
    train_loss = 0.0
    for i in tqdm(range(n_batches)):
        X_batch = X_train_torch[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train_torch[i * batch_size:(i + 1) * batch_size]
        lr_optimizer.zero_grad()
        y_pred = lr(X_batch).squeeze()
        loss = lr_loss_fn(y_pred, y_batch)
        loss.backward()
        lr_optimizer.step()
        train_loss += loss.item()
    train_loss /= n_batches
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))

torch.save(lr.state_dict(), './Results/Machine-Learning-rezults/lr.pth') # 保存模型参数

# 在GPU上测试模型
y_test_pred_lr = lr(X_test_torch).squeeze().cpu().detach().numpy()
y_test_pred_lr = (y_test_pred_lr > 0.5).astype(int)

print("Test set performance:")
print("Logistic Regression - Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}".format(
    accuracy_score(y_test, y_test_pred_lr), precision_score(y_test, y_test_pred_lr),
    recall_score(y_test, y_test_pred_lr), f1_score(y_test, y_test_pred_lr)))