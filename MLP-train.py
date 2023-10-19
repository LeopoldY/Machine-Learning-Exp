# 使用多层感知机训练
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.optim as optim

from Models.ML.MLP import MLP
from Models.ML.MLP import mlp_loss_fn

from tqdm import tqdm

# 加载原图和掩码图
image_path = "./Resorce/images/img10.jpg"
mask_path = "./Resorce/maskPic/SegmentationClassPNG/img10.png"
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

# 数据预处理

# 将原图和掩码图转换为一维数组
X = image.reshape(-1, 3)
# 通过掩码图中的像素值判断该像素是否属于目标类别
mask_red = np.all(mask == [0, 0, 128], axis=2)
y = mask_red.astype(int).flatten()
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train_torch = torch.from_numpy(X_train).float()
y_train_torch = torch.from_numpy(y_train).float()
X_test_torch = torch.from_numpy(X_test).float()

# 创建多层感知机模型
mlp = MLP(3, 1)
mlp_optimizer = optim.Adam(mlp.parameters()) # Adam优化器

# 在GPU上训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp.to(device)

# 将数据移动到GPU上
X_train_torch = X_train_torch.to(device)
y_train_torch = y_train_torch.to(device)
X_test_torch = X_test_torch.to(device)

# 使用tqdm库创建进度条，并在每个迭代步骤中更新进度条

num_epochs = 10
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    mlp.train() 
    train_loss = 0.0
    for i in tqdm(range(n_batches)):
        X_batch = X_train_torch[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train_torch[i * batch_size:(i + 1) * batch_size]
        mlp_optimizer.zero_grad()
        y_pred = mlp(X_batch).squeeze()
        loss = mlp_loss_fn(y_pred, y_batch)
        loss.backward()
        mlp_optimizer.step()
        train_loss += loss.item()
    print("Epoch: {} Train loss: {}".format(epoch + 1, train_loss / n_batches))

# 在测试集上评估模型
mlp.eval()
y_test_pred_mlp = mlp(X_test_torch).squeeze().cpu().detach().numpy()
y_test_pred_mlp = (y_test_pred_mlp >= 0.5).astype(int)

print("Test set performance:")
print("Multi-layer Perceptron - Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}".format(
    accuracy_score(y_test, y_test_pred_mlp), precision_score(y_test, y_test_pred_mlp),
    recall_score(y_test, y_test_pred_mlp), f1_score(y_test, y_test_pred_mlp)))