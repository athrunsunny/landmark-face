import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm


def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return x1, x2, y1, y2


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def wider2face(root, ignore_small=0):
    data = {}
    with open('{}/label.txt'.format(root), 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if '#' in line:
                path = '{}/images/{}'.format(root, line.split()[-1])
                img = cv2.imread(path)
                height, width, _ = img.shape
                data[path] = list()
            else:
                box = np.array(line.split()[0:4], dtype=np.float32)  # (x1,y1,w,h)
                if box[2] < ignore_small or box[3] < ignore_small:
                    continue
                box = convert((width, height), xywh2xxyy(box))
                label = '0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'.format(round(box[0], 4), round(box[1], 4),
                                                                             round(box[2], 4), round(box[3], 4))
                data[path].append(label)
    return data


def create_path(dir, name):
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        os.makedirs(path)


def create_val_datasets(root_path):
    """
    put label.txt to WIDER_val file
    WIDER_val
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

    datas = wider2face(root_path)
    for idx, data in enumerate(datas.keys()):

        pic_name = data.replace("\\", "/").split("/")[-1].split(".")[0]
        out_img = f'{image_path}/{pic_name}.jpg'
        out_txt = f'{label_path}/{pic_name}.txt'
        shutil.copyfile(data, out_img)
        labels = datas[data]
        f = open(out_txt, 'w')
        for label in labels:
            f.write(label + '\n')
        f.close()


if __name__ == '__main__':
    root_path = r'G:\deeplearning_dataset\widerface\WIDER_val\WIDER_val'
    create_val_datasets(root_path)
