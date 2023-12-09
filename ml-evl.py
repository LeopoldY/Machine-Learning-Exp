# 验证
import cv2
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale

clsfModel = 'mlp' # 'lr', 'svm', 'mlp
assert clsfModel in ['lr', 'svm', 'mlp'] #"clsfModel must be one of ['lr', 'svm', 'mlp']"

# 加载原图和掩码图
image_path = "PATH/TO/IMAGE"
image = cv2.imread(image_path)

# 将原图和掩码图转换为一维数组
X = image.reshape(-1, 3)

def predict(clf='svm', X=None):
    if clsfModel == 'lr':
        clf = LogisticRegression()
    elif clsfModel == 'svm':
        clf = SVC()
        X = minmax_scale(X)
    elif clsfModel == 'mlp': 
        clf = MLPClassifier()
    
    with open(f"./output/ml/{clsfModel}_model.pkl", "rb") as f:
        clf = pickle.load(f)

    y_pred = clf.predict(X)
    return y_pred

y_pred = predict(clsfModel, X)
# 可视化分割结果
mask = y_pred.reshape(image.shape[0], image.shape[1])   # 将预测结果转换为二维数组

# 二值化处理
mask[np.where(mask >= 0.5)] = 1
mask[np.where(mask < 0.5)] = 0

mask = np.uint8(mask * 255) # 将预测结果转换为0-255的灰度图

mask_pred = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 将灰度图转换为三通道图像

mask_pred[np.where((mask_pred == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # 将预测结果中的目标类别像素标记为红色
cv2.imwrite(f"./output/ml/{clsfModel}_result_mask.png", mask_pred) # 保存预测结果

result = cv2.addWeighted(image, 0.5, mask_pred, 0.5, 0) # 将原图和预测结果融合

cv2.imshow(f"result of {clsfModel}", result) # 显示融合后的图像
cv2.imwrite(f"./output/ml/{clsfModel}_result.jpg", result) # 保存融合后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()