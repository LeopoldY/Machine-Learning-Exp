import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from Models.ML.SVM import LinearSVM

import torch

# 加载原图和掩码图
image_path = "./Resorce/images/img10.jpg"
mask_path = "./Resorce/maskPic/SegmentationClassPNG/img10.png"
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

# 将原图和掩码图转换为一维数组
X = image.reshape(-1, 3)
# 通过掩码图中的像素值判断该像素是否属于目标类别
mask_red = np.all(mask == [0, 0, 128], axis=2)
y = mask_red.astype(int).flatten()
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
svm = LinearSVM()

state_dict = torch.load('./Results/Machine-Learning-rezults/SVM.pth')
svm.load_state_dict(state_dict)

# 在GPU上验证模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
svm.to(device)

# 可视化分割结果
redMask = np.array([[[0, 0, 255] for i in range(mask.shape[1])] for j in range(mask.shape[0])])
X_torch = torch.from_numpy(X).float().to(device)
mask_pred_svm = svm(X_torch).squeeze().cpu().detach().numpy().reshape(image.shape[0], image.shape[1])   # 将预测结果转换为二维数组
# 二值化处理
mask_pred_svm[np.where(mask_pred_svm >= 0.5)] = 1
mask_pred_svm[np.where(mask_pred_svm < 0.5)] = 0

mask_pred_svm = np.uint8(mask_pred_svm * 255) # 将预测结果转换为0-255的灰度图

cv2.imwrite("./Results/Machine-Learning-rezults/SVM_img10_pred.jpg", mask_pred_svm) # 将预测结果保存为图片

mask_pred_svm = cv2.cvtColor(mask_pred_svm, cv2.COLOR_GRAY2BGR) # 将灰度图转换为三通道图像

mask_pred_svm[np.where((mask_pred_svm == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # 将预测结果中的目标类别像素标记为红色

result_SVM = cv2.addWeighted(image, 0.5, mask_pred_svm, 0.5, 0) # 将原图和预测结果融合

cv2.imshow("Multi-layer Perceptron", result_SVM) # 显示融合后的图像
cv2.imwrite("./Results/Machine-Learning-rezults/SVM_img10_result.jpg", result_SVM) # 将融合后的图像保存为图片
cv2.waitKey(0)
cv2.destroyAllWindows()