# landmark-face

Directly copy files (data,landmark,models,utils) to yolov5 folder

# train

python landmark/train.py --batch 32 --data data/widerface.yaml --device 0 --name facenck --weights '' --epochs 300 --cfg models/landmark/yolov5n-lmk.yaml

# export 

python landmark/export.py --weights yolov5n-lmk.pt --include torchscript onnx

# datasets

widerface datasets: http://shuoyang1213.me/WIDERFACE/

landmark gt:folder widerface_landmark_gt

# How to use widerface_landmark_gt

Put "train/label.txt" in the "WIDER_train" path
  
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
        
"../WIDER_train" path is written to "original_path"

python train2yolo.py    

