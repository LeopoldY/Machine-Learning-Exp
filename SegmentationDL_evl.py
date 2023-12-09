import numpy as np
import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from Models.DL.Net import USegNet, SimpleCNN

use_cuda = True
model = SimpleCNN() #ConvolutionNet()
model.load_state_dict(torch.load('./output/dl/SCNN_params_final.pth'))
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

img = cv2.imread('./Resorce/pics/44.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0) 
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))

prediction = prediction.squeeze().cpu().data.numpy()
prediction[prediction>=0.5] = 255
prediction[prediction<0.5] = 0
prediction = prediction.astype(np.uint8)

mask_pred = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR) # 将灰度图转换为三通道图像

mask_pred[np.where((mask_pred == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # 将预测结果中的目标类别像素标记为红色
cv2.imwrite('output/dl/prediction-scnn.png', mask_pred)

result = cv2.addWeighted(img, 0.5, mask_pred, 0.5, 0) # 将原图和预测结果融合

cv2.imshow(f"result of SegNet", result) # 显示融合后的图像
cv2.imwrite('output/dl/result-scnn.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()