# landmark-face

Directly replace files with the same name in yolov5 with files in the folder

# train

python landmark/train_landmark.py --batch 32 --data data/widerface.yaml --device 0 --name facenck --weights '' --epochs 300 --cfg models/landmark/yolov5n-lmk.yaml

# datasets

widerface datasets: http://shuoyang1213.me/WIDERFACE/

landmark gt:folder widerface_landmark_gt
