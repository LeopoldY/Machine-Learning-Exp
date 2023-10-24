import cv2
import numpy as np

from PIL import Image

# label = cv2.imread("./Resorce/TrainDataSet/Lable/img10_128_384.jpg")
lable = Image.open("./Resorce/TrainDataSet/Lable/img10_128_384.png")
target = np.array(lable).astype(np.int64)
target[target > 0] = 1
print(target.shape)