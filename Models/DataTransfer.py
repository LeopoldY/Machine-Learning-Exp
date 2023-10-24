#将原图和掩码图分别分割成128*128的小图，保存在指定文件夹下
import os
import cv2


def split_image(imgPath, save_path, splitPixel):
    for root, dirs, files in os.walk(imgPath):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            height, width, _ = img.shape
            for i in range(0, height, splitPixel):
                for j in range(0, width, splitPixel):
                    if i + splitPixel > height or j + splitPixel > width:
                        continue
                    crop_img = img[i:i + splitPixel, j:j + splitPixel]
                    cv2.imwrite(os.path.join(save_path, file.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png'),
                                crop_img)

if __name__ == "__main__":
    # splitPixel = 256
    # ogPath = './Resorce/maskPic/SegmentationClassPNG'
    # savePath = './Resorce/TrainDataSet/Label'
    # split_image(ogPath, savePath, splitPixel)
    # print('Done!')

    #将分割后的文件名写入txt文件
    with open('./Resorce/TrainDataSet/total.txt', 'w') as f:
        for root, dirs, files in os.walk('./Resorce/TrainDataSet/Label'):
            for file in files:
                f.write(file.split('.')[0] + '\n')