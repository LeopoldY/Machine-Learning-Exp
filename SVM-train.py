# 使用svm和sklearn库训练
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

from Models.ML.SVM import SVM
from Models.ML.SVM import LinearSVM
from Models.ML.SVM import svm_loss_fn

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

# 创建支持向量机模型
svm = LinearSVM()
svm_optimizer = torch.optim.Adam(svm.parameters()) # Adam优化器

# 在GPU上训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
svm.to(device)

# 将数据移动到GPU上
X_train_torch = torch.from_numpy(X_train).float().to(device)
y_train_torch = torch.from_numpy(y_train).float().to(device)
X_test_torch = torch.from_numpy(X_test).float().to(device)

# 训练模型
num_epochs = 1
batch_size = 128
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    svm.train() 
    train_loss = 0.0
    for i in tqdm(range(n_batches)):
        X_batch = X_train_torch[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train_torch[i * batch_size:(i + 1) * batch_size]
        svm_optimizer.zero_grad()
        y_pred = svm(X_batch).squeeze()
        loss = svm_loss_fn(y_pred, y_batch)
        loss.backward()
        svm_optimizer.step()
        train_loss += loss.item()
    train_loss /= n_batches
    print("epoch: {}, train loss: {:.6f}".format(epoch + 1, train_loss))



# 在测试集上评估模型
y_pred = svm(X_test_torch).squeeze().cpu().detach().numpy()
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision: %.2f" % precision_score(y_test, y_pred))
print("Recall: %.2f" % recall_score(y_test, y_pred))
print("F1: %.2f" % f1_score(y_test, y_pred))

# 保存模型
torch.save(svm.state_dict(), './Results/Machine-Learning-rezults/SVM.pth')