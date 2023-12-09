import math
import cv2 
import numpy as np

# 读取图片 
image = cv2.imread('mask_processed.png') 
 
# 定义黑色的范围 
lower_black = np.array([0, 0, 0], dtype=np.uint8) 
upper_black = np.array([30, 30, 30], dtype=np.uint8) 
 
# 创建掩码，将黑色部分标记为白色 
black_mask = cv2.inRange(image, lower_black, upper_black) 
 
# 将黑色部分替换为白色 
image[black_mask != 0] = [255, 255, 255]

# 剔除掩码图中红色色块过小的部分


# 保存处理后的图片 
cv2.imwrite('./output/output_image.jpg', image) 
 
src = image
# 得到原图像的长宽
srcWidth = src.shape[1]
srcHeight = src.shape[0]
dist_coefs = np.array([0, 0, 0, 0, 0], dtype = np.float64)
camera_matrix = np.array([[srcWidth * 0.5, 0, srcWidth / 2], [0, srcWidth * 0.5, srcHeight / 2], [0, 0, 1]],
                         dtype = np.float64)    # 相机内参矩阵


newWidth = 500 # 新图像宽
newHeight = 800 # 新图像高
# 新相机内参，自己设定
newCam = np.array([[newWidth * 0.15, 0, newWidth / 2], [0, newWidth * 0.15, newHeight / 2], [0, 0, 1]])
invNewCam = np.linalg.inv(newCam)  # 内参逆矩阵
map_x = np.zeros((newHeight, newWidth), dtype=np.float32)
map_y = np.zeros((newHeight, newWidth), dtype=np.float32) # 用于存储图像坐标的数组

for k in range(10, 90, 10):
    pitch = 50 * 3.14 / 180
    print("pitch = ", pitch)
    R = np.array([[1, 0, 0], [0, math.sin(pitch), math.cos(pitch)], [0, -math.cos(pitch), math.sin(pitch)]])
    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            ray = np.dot(invNewCam, np.array([j, i, 1]).T)  # 像素转换为入射光线
            rr = np.dot(R, ray)  # 乘以旋转矩阵
            # 光线投影到像素点
            point, _ = cv2.projectPoints(rr, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), camera_matrix,
                                        dist_coefs)
            map_x[i, j] = point[0][0][0]
            map_y[i, j] = point[0][0][1]
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)  # 它接受输入图像src以及之前计算的map_x和map_y，然后进行图像的重映射。
    cv2.imwrite("img_bev.jpg", dst)
    dst = cv2.resize(dst, (2352 // 2, 1728 // 2))
    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey()
