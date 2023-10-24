#将原图和掩码图分别分割成128*128的小图，保存在指定文件夹下
import os
import cv2


def split_image(imgPath, save_path):
    for root, dirs, files in os.walk(imgPath):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            height, width, _ = img.shape
            for i in range(0, height, 128):
                for j in range(0, width, 128):
                    if i + 128 > height or j + 128 > width:
                        continue
                    else:
                        crop_img = img[i:i + 128, j:j + 128]
                        cv2.imwrite(os.path.join(save_path, file.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png'),
                                    crop_img)
