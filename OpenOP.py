# 导入必要的库
import numpy as np
import skimage.io
import skimage.morphology
import skimage.measure

# 读取掩码图
mask = skimage.io.imread('PATH/TO/MASK')

# 将掩码图转换为二值图，红色区域为1，黑色区域为0
mask = mask[:,:,0] > 0

# 使用面积开运算来剔除一些很小的红色区域
area_threshold = 0.005 # 面积阈值，表示像素数占总像素数的比例
connectivity = 8 # 连通性，表示相邻像素的连接方式
mask = skimage.morphology.area_opening(mask, area_threshold=area_threshold*mask.size, connectivity=connectivity) # 面积开运算

# 保存处理后的掩码图
skimage.io.imsave('mask_processed.png', mask.astype(np.uint8))

# 显示处理后的掩码图
skimage.io.imshow(mask)
skimage.io.show()

import cv2

# 将白色部分转换为红色
image = cv2.imread('mask_processed.png')
image[np.where((image == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # 将预测结果中的目标类别像素标记为红色
cv2.imwrite('mask_processed.png', image)