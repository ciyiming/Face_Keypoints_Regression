import os
import cv2
import numpy as np


DATAPATH = '..\data'
FOLDER_LIST = ['I', 'II']
LABLE_FILE_NAME = ['label.txt', 'label.txt']
valset_ratio = 0.3


def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):   # usually ratio = 0.25
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
    return roi_x1, roi_y1, roi_x2, roi_y2, \
           roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


if __name__ == '__main__':
    train_set_file = open('train.txt', 'w')
    test_set_file = open('test.txt', 'w')
    for folder in range(len(FOLDER_LIST)):
        num = 1
        path = os.path.join(DATAPATH, FOLDER_LIST[folder])
        data_set_size = len(open(os.path.join(path, LABLE_FILE_NAME[folder])).readlines())
        labels = open(os.path.join(path, LABLE_FILE_NAME[folder]))
        for line in labels:
            label = line.split()
            file_name = label[0]
            file_path = os.path.join(path, file_name)
            x1, y1, x2, y2 = [float(i) for i in label[1:5]]
            key_points = np.array([float(i) for i in label[5:]]).reshape(-1, 2)
            try:
                img = cv2.imread(file_path)
            except Exception:
                print('Not a valid image.')
            else:
                h, w, _ = img.shape
                roi_x1, roi_y1, roi_x2, roi_y2, _, _ = expand_roi(x1, y1, x2, y2, w, h, 0.25)
                landmarks = key_points - np.array([roi_x1, roi_y1])

                #show image to check labels
                # img = img[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]
                # img = cv2.rectangle(img, (int(roi_x1), int(roi_y1)), (int(roi_x2), int(roi_y2)), (0, 255, 0), 1)
                # for point in landmarks:
                #     x, y = point
                #     cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 0)
                # cv2.imshow('img', img)
                # key = cv2.waitKey(1000)
                # if key == 27:
                #     exit(0)
                # cv2.destroyAllWindows()

                list = [str(i) for i in [roi_x1, roi_y1, roi_x2, roi_y2] + landmarks.reshape(1, -1).squeeze(0).tolist()]
                line = file_path + ' ' + ' '.join(list) + '\n'
                if num <= valset_ratio * data_set_size:
                    test_set_file.write(line)
                else:
                    train_set_file.write(line)
                num += 1
    train_set_file.close()
    test_set_file.close()
    print('Files train.txt and test.txt saved! ')
