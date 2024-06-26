# YOLO-anime-hands
Example with YOLOv8x:
![yolo](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/c0c820d2-24d8-4d4a-b452-ad8e20811275)

A model that is trained on only gwerns data with seems to struggle with gloves, handshakes and more complex hands. I thus added some custom data. I also tried to train nano and medium sized yolo, but that resulted in models with severe accuracy problems.

There is also [adetailer](https://huggingface.co/Bingsu/adetailer) which has multiple models for this task, but these models usually have low conficence detections in drawn images which are sometimes below 50% and are prone to misdetection.

Training code reference:
```python
from ultralytics import YOLO

model = YOLO('model.pt')

# training will abort early due to early stopping
results = model.train(data='coco128.yaml', epochs=10000, imgsz=640, batch=20, amp=True)
```

Usage example:
```python
from PIL import Image
import cv2
from ultralytics import YOLO

model = YOLO('model.pt')

results = model('test.jpg') # conf=0.5)

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_output.jpg", img) 
```
To process a folder with images:
```python
import os
from PIL import Image
import cv2
from ultralytics import YOLO
from tqdm import tqdm

model = YOLO('model.pt')

input_folder = '/'
output_folder = '/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(input_folder):
    if file_name.endswith(('.jpg', '.png', '.webp')):
        image_path = os.path.join(input_folder, file_name)
        results = model(image_path)
        output_path = os.path.join(output_folder, file_name)

        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, img)
```

## Graphs

Training with a 4090 and Prodigy optimizer set to 1. Using [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) commit `db2af70d3910f168a62ecaae4d920e1440f08c7e` because newer versions seem to have converging problems and train much slower. May be due to unsuitable defaults.

YOLOv8x_*: [best](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_best.pt) [last](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_last.pt) [best onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_best_fp16_op18.onnx) [best dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_best_dynamic_op18.onnx) [last onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_last_fp16_op18.onnx.onnx) [last dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_last_dynamic_op18.onnx) [csv](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_results.csv)
- 992 epochs
- batch 30 (?)
- dataset:
    - [gwern](https://gwern.net/crop#hands-download) (5371 images)
 
![results](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/ed921cff-5f54-418f-b1f3-83f69e87981d)

YOLOv8x_*_finetuned: [best](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_best_finetuned.pt) [last](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_last_finetuned.pt) [best onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_best_finetuned_fp16_op18.onnx) [best dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_best_finetuned_dynamic_op18.onnx) [last onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_last_finetuned_fp16_op18.onnx.onnx) [last dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_last_finetuned_dynamic_op18.onnx) [csv](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv8x_results_finetuned.csv)
- used gwern trained YOLOv8x as pretrain
- 704 epochs
- batch 30 (?)
- dataset:
    - own custom data (924 images)

![results](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/e684f351-ded3-460d-93c1-7f89df38049c)

YOLOv9e_*: [best](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_best.pt) [last](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_last.pt) [best onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_best_fp16_op18.onnx) [best dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_best_dynamic_op18.onnx) [last onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_last_fp16_op18.onnx) [last dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_last_dynamic_op18.onnx) [csv](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_results.csv)
- 1181 epochs
- 53.533 hours
- batch 14
- dataset:
    - [gwern](https://gwern.net/crop#hands-download) (5371 images)

![results](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/aaae8564-0d55-49a7-8fbb-fffe5f491b54)

YOLOv9e_*_finetuned: [best](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_best_finetuned.pt) [last](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_last_finetuned.pt) [best onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_best_finetuned_fp16_op18.onnx) [best dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_best_finetuned_dynamic_op18.onnx) [last onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_last_finetuned_fp16_op18.onnx) [last dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_last_finetuned_dynamic_op18.onnx) [csv](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_results_finetuned.csv)
- used gwern trained YOLOv9e as pretrain
- 725 epochs
- 6.852 hours
- batch 14
- dataset:
    - own custom data (1069 images)

![results](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/6237c61a-9fb3-4b85-a23a-7d53dc3a38b3)

YOLOv9e_gwern+own_*: [best](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_gwern+own_best.pt) [last](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_gwern+own_last.pt) [best onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_gwern+own_best_fp16_op18.onnx) [best dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_gwern+own_best_dynamic_op18.onnx) [last onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_gwern+own_last_fp16_op18.onnx) [last dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_gwern+own_last_dynamic_op18.onnx) [csv](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_results_gwern+own.csv)
- 1177 epochs
- 56.442 hours
- batch 14
- dataset (6440 images):
    - [gwern](https://gwern.net/crop#hands-download) (5371 images)
    - own data (1069 images)

![results](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/28811ad0-3530-44ae-b0a0-a33d77b4a453)

YOLOv9e_all_*: [best](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_best.pt) [last](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_last.pt) [best onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_best_fp16_op18.onnx) [best dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_best_dynamic_op18.onnx) [last onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_last_fp16_op18.onnx) [last dynamic onnx](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/YOLOv9e_all_last_dynamic_op18.onnx) [csv](https://github.com/styler00dollar/YOLO-anime-hands/releases/download/models/results_all.csv)
- 1166 epochs 
- 172.749 hours (~7.2 days)
- batch 14
- dataset (17392 images):
    - [gwern](https://gwern.net/crop#hands-download) (5371 images)
    - own data (1069 images)
    - [1-yshhi/anhdet](https://universe.roboflow.com/1-yshhi/anhdet) (5705 images)
    - [catwithawand/hand-detection-fuao9](https://universe.roboflow.com/catwithawand/hand-detection-fuao9) (5247 images)

![results](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/dc83cb22-b43b-4c89-b1ab-1aed0bf39058)

Dataset graphs:

Gwerns dataset (5371 images):

- [gwern](https://gwern.net/crop#hands-download) (5371 images)


![labels](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/ebe840c4-698d-4675-85c2-bd05c6573792)

My custom dataset (1069 images):

- own data (1069 images)
  
![labels](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/fbf90d84-2cc3-4a6c-bdee-c20cbe4418f4)

Gwerns + own data (6440 images):

- [gwern](https://gwern.net/crop#hands-download) (5371 images)
- own data (1069 images)

![labels](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/d83d4342-31a0-41fe-bcb4-f2266affcdc4)

All combined (17392 images):

- [gwern](https://gwern.net/crop#hands-download) (5371 images)
- own data (1069 images)
- [1-yshhi/anhdet](https://universe.roboflow.com/1-yshhi/anhdet) (5705 images)
- [catwithawand/hand-detection-fuao9](https://universe.roboflow.com/catwithawand/hand-detection-fuao9) (5247 images)

![labels](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/89dcd317-23f9-4366-be56-73aba9ba3cf9)
