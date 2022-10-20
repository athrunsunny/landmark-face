import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return torch.stack(imgs, 0), targets


def create_path(dir, name):
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        os.makedirs(path)


def create_train_datasets(root_path):
    """
    put label.txt to WIDER_train file
    WIDER_train
        ├── images
        │   ├── 0--Parade
        │   │   ├── 0.jpg
        │   │   └── 1.jpg ...
        │   └── 1--Handshaking
        │       ├── 0.jpg
        │       └── 1.jpg ...
        │
        └── label.txt
    Returns:
        None
    """
    if not os.path.isfile(os.path.join(root_path, 'label.txt')):
        print('Missing label.txt file.')
        exit(1)

    save_path = os.path.join(root_path, 'change')
    create_path(root_path, 'change')
    # create images and labels file
    image_path = os.path.join(save_path, 'images')
    label_path = os.path.join(save_path, 'labels')
    create_path(save_path, 'images')
    create_path(save_path, 'labels')

    data = WiderFaceDetection(os.path.join(root_path, 'label.txt'))

    for i in tqdm(range(len(data.imgs_path))):
        try:
            img = cv2.imread(data.imgs_path[i])
            shape = img.shape
            pic_name = data.imgs_path[i].replace("\\", "/").split("/")[-1].split(".")[0]
            save_img_path = f'{image_path}/{pic_name}.jpg'
            save_txt_path = f'{label_path}/{pic_name}.txt'
            with open(save_txt_path, "w") as f:
                height, width, _ = shape
                labels = data.words[i]

                if len(labels) == 0:
                    continue
                for idx, label in enumerate(labels):
                    annotation = np.zeros((1, 14))
                    # bbox
                    label[0] = max(0, label[0])
                    label[1] = max(0, label[1])
                    label[2] = min(width - 1, label[2])
                    label[3] = min(height - 1, label[3])
                    annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
                    annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
                    annotation[0, 2] = label[2] / width  # w
                    annotation[0, 3] = label[3] / height  # h
                    # if (label[2] -label[0]) < 8 or (label[3] - label[1]) < 8:
                    #    img[int(label[1]):int(label[3]), int(label[0]):int(label[2])] = 127
                    #    continue
                    # landmarks
                    annotation[0, 4] = label[4] / width  # l0_x
                    annotation[0, 5] = label[5] / height  # l0_y
                    annotation[0, 6] = label[7] / width  # l1_x
                    annotation[0, 7] = label[8] / height  # l1_y
                    annotation[0, 8] = label[10] / width  # l2_x
                    annotation[0, 9] = label[11] / height  # l2_y
                    annotation[0, 10] = label[13] / width  # l3_x
                    annotation[0, 11] = label[14] / height  # l3_y
                    annotation[0, 12] = label[16] / width  # l4_x
                    annotation[0, 13] = label[17] / height  # l4_y
                    str_label = "0 "
                    for i in range(len(annotation[0])):
                        str_label = str_label + " " + str(annotation[0][i])
                    str_label = str_label.replace('[', '').replace(']', '')
                    str_label = str_label.replace(',', '') + '\n'
                    f.write(str_label)
            cv2.imwrite(save_img_path, img)
        except:
            pass


if __name__ == '__main__':
    original_path = r'G:\deeplearning_dataset\widerface\WIDER_train\WIDER_train'
    create_train_datasets(original_path)