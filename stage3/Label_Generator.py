import os
import cv2
import numpy as np
import pandas as pd


DATAPATH = '..\data'
FOLDER_LIST = ['I', 'II']
LABLE_FILE_NAME = ['label.txt', 'label.txt']
valset_ratio = 0.3
neg_example_ratio = 3.0
neg_iou_thres = 0.3
roi_expand_ratio = 0.25
min_pixel = 10  # minimum pixels of random crop
height_width = (0.5, 2)  # the range of height divided by width random crop

np.random.seed(0)


def expand_roi(titles, bboxes, key_points, img_width, img_height, ratio):
    result = pd.DataFrame(columns=titles)
    for idx, bbox in bboxes.iterrows():
        # expand bbox
        x1, y1, x2, y2 = bbox
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
        # expand landmarks
        landmarks = np.array(key_points.loc[idx, :]).reshape(-1, 2) - np.array([roi_x1, roi_y1])
        landmarks = landmarks.reshape(1, -1).squeeze(0).tolist()
        list = pd.DataFrame(pd.DataFrame([roi_x1, roi_y1, roi_x2, roi_y2, 1] + landmarks)).T
        list.columns = titles
        # add into result
        result = pd.concat([result, list], axis=0, sort=False)
    return result


def check_iou(rect1, rect2):
    # rect: 0-4: x1, y1, x2, y2
    left1, top1, right1, bottom1 = rect1
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2, top2, right2, bottom2 = rect2
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    box1_area = width1 * height1
    box2_area = width2 * height2
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou


def random_crop(height, width, min_pixel):
    x1 = np.random.random_integers(0, width - min_pixel)
    y1 = np.random.random_integers(0, height - min_pixel)
    x2 = np.random.random_integers(x1 + min_pixel, width)
    y2 = np.random.random_integers(y1 + min_pixel, height)
    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1
    return [x1, y1, x2, y2], crop_w, crop_h


if __name__ == '__main__':
    kps = ['kp%s%s' % (e, i) for i in range(1, 22) for e in ['x', 'y']]
    titles = ['x1', 'y1', 'x2', 'y2', 'class'] + kps
    all_labels = pd.DataFrame(columns=['file'] + titles)
    for folder in range(len(FOLDER_LIST)):
        path = os.path.join(DATAPATH, FOLDER_LIST[folder])
        labels = pd.read_table(os.path.join(path, LABLE_FILE_NAME[folder]), header=None, sep=' ')
        files = os.listdir(path)
        for file in files:
            if file[-4:] == '.jpg':
                try:
                    img = cv2.imread(os.path.join(path, file))
                except Exception:
                    print('Not a valid image.')
                else:
                    negs = []
                    h, w, _ = img.shape
                    try:
                        bboxes = labels.loc[labels[0] == file].iloc[:, 1:5]
                        key_points = labels.loc[labels[0] == file].iloc[:, 5:]
                    except Exception:
                        print('Not a valid label.')
                    else:
                        # get negative examples
                        while len(negs) < neg_example_ratio * len(bboxes):
                            rect, crop_w, crop_h = random_crop(h, w, min_pixel)
                            is_neg = False
                            if height_width[0] < crop_h / crop_w < height_width[1]:
                                is_neg = True
                                for idx, bbox in bboxes.iterrows():
                                    if check_iou(rect, bbox) > neg_iou_thres:
                                        is_neg = False
                                        break
                            if is_neg:
                                negs.append(rect)
                        negs = pd.DataFrame(negs, columns=['x1', 'y1', 'x2', 'y2'])
                        negs['class'] = [0] * len(negs)
                        negs.insert(0, 'file', os.path.join(path, file))
                        all_labels = pd.concat([all_labels, negs], axis=0, sort=False)
                        # expand roi
                        poss = expand_roi(titles, bboxes, key_points, w, h, roi_expand_ratio)
                        poss.insert(0, 'file', os.path.join(path, file))
                        all_labels = pd.concat([all_labels, poss], axis=0, ignore_index=True, sort=False)
                        # show image to check labels
                        # for idx, row in poss.iterrows():
                        #     roi_x1, roi_y1, roi_x2, roi_y2 = row[['x1', 'y1', 'x2', 'y2']]
                        #     img_t = img[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]
                        #     points = np.array(row[kps]).reshape(-1, 2)
                        #     for point in points:
                        #         x, y = point
                        #         cv2.circle(img_t, (int(x), int(y)), 1, (0, 0, 255), 0)
                        #     cv2.imshow('img', img_t)
                        #     key = cv2.waitKey(1000)
                        #     if key == 27:
                        #         exit(0)
                        #     cv2.destroyAllWindows()
    all_labels = all_labels.sample(frac=1.0)
    data_set_size = len(all_labels)
    print(data_set_size)
    val_set = all_labels[0:int(data_set_size * valset_ratio)]
    val_set.index = [i for i in range(len(val_set))]
    train_set = all_labels[int(data_set_size * valset_ratio):]
    train_set.index = [i for i in range(len(train_set))]
    val_set.to_csv('./test.csv', sep=',', header=True, index=False)
    train_set.to_csv('./train.csv', sep=',', header=True, index=False)
    print('Files train.csv and test.csv saved! ')
