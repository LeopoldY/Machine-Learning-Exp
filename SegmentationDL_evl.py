import numpy as np
import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from Models.DL.Net import ConvolutionNet

use_cuda = False
model = ConvolutionNet()
model.load_state_dict(torch.load('output/params_10.pth'))
# model = torch.load('output/model.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

img = cv2.imread('./Resorce/pics/10.jpg')
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
cv2.imwrite('output/prediction.png', prediction)

cv2.imshow("image", prediction)
cv2.waitKey(0)