#将原图和掩码图分别分割成128*128的小图，保存在指定文件夹下
import os
import cv2
import numpy as np


def split_image(imgPath, save_path, splitPixel, savePattern='.jpg'):
    for root, dirs, files in os.walk(imgPath):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            height, width, _ = img.shape
            for i in range(0, height, splitPixel):
                for j in range(0, width, splitPixel):
                    if i + splitPixel > height or j + splitPixel > width:
                        continue
                    crop_img = img[i:i + splitPixel, j:j + splitPixel]
                    cv2.imwrite(os.path.join(save_path, file.split('.')[0] + '_' + str(i) + '_' + str(j) + savePattern),
                                crop_img)

if __name__ == "__main__":
    splitPixel = 128
    ogPath = 'PATH/TO/ORIGINAL_IMAGE'
    OGsavePath = 'PATH/TO/SAVE_ORIGINAL_IMAGE'
    maskPath = 'PATH/TO/MASK_IMAGE'
    maskSavePath = 'PATH/TO/SAVE_MASK_IMAGE'
    # 清空OGsavePath和maskSavePath
    for root, dirs, files in os.walk(OGsavePath):
        for file in files:
            os.remove(os.path.join(root, file))

    for root, dirs, files in os.walk(maskSavePath):
        for file in files:
            os.remove(os.path.join(root, file))
    
    split_image(ogPath, OGsavePath, splitPixel, '.jpg')
    split_image(maskPath, maskSavePath, splitPixel, '.png')
    print('Done!')
    
    # 读取掩码图
    maskPath = maskSavePath
    maskList = os.listdir(maskPath)
    # 删除红色像素量少于10%的掩码图和对应的原图
    for mask in maskList:
        maskImg = cv2.imread(os.path.join(maskPath, mask))
        maskImg = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
        maskImg[maskImg > 0] = 1
        if np.sum(maskImg) / (maskImg.shape[0] * maskImg.shape[1]) < 0.05:
            os.remove(os.path.join(maskPath, mask))
            os.remove(os.path.join(OGsavePath, mask.strip('.png') + '.jpg'))
            print(f"remove {mask}")

    #将分割后的文件名写入txt文件
    os.remove('./Resorce/TrainDataSet/total.txt') if os.path.exists('./Resorce/TrainDataSet/total.txt') else None
    with open('./Resorce/TrainDataSet/total.txt', 'w') as f:
        for root, dirs, files in os.walk('./Resorce/TrainDataSet/Label'):
            for file in files:
                f.write(file.split('.')[0] + '\n')