import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

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

# 随机选取20000条正例和反例数据进行训练
positive_index = np.random.choice(np.where(y == 1)[0], 20000)
negative_index = np.random.choice(np.where(y == 0)[0], 20000)
index = np.concatenate((positive_index, negative_index))
np.random.shuffle(index)
X = X[index]
y = y[index]

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import minmax_scale

X_train = minmax_scale(X_train)
X_test = minmax_scale(X_test)


# 正例数量
positive_num = sum(y_train == 1)
# 反例数量
negative_num = sum(y_train == 0)
print("正例数量：", positive_num)
print("反例数量：", negative_num)

# 训练模型
clf = SVC(verbose=True, probability=True)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估
print("准确率：", accuracy_score(y_test, y_pred))
print("精确率：", precision_score(y_test, y_pred))
print("召回率：", recall_score(y_test, y_pred))
print("F1值：", f1_score(y_test, y_pred))

# 保存模型
import pickle
with open("./output/ml/svm_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# 计算ROC曲线
y_score = clf.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print("AUC值：", roc_auc)

# 绘制ROC曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()
