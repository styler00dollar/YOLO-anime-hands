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

YOLOv8x with gwerns dataset:
- 992 epochs
- batch 30 (?)
- dataset:
    - [gwern](https://gwern.net/crop#hands-download)
    - 5371 images
 
![results](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/ed921cff-5f54-418f-b1f3-83f69e87981d)

Finetuned YOLOv8x with own dataset:
- 704 epochs
- batch 30 (?)
- dataset:
    - own custom data
    - 924 images

![results](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/e684f351-ded3-460d-93c1-7f89df38049c)

YOLOv9e with gwerns dataset:
- 1181 epochs
- 53.533 hours
- batch 14
- dataset:
    - [gwern](https://gwern.net/crop#hands-download)
    - 5371 images

![results](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/aaae8564-0d55-49a7-8fbb-fffe5f491b54)

Finetuned YOLOv9e with own dataset:
- 725 epochs
- 6.852 hours
- batch 14
- dataset:
    - own custom data
    - 1069 images

![results](https://github.com/styler00dollar/YOLO-anime-hands/assets/51405565/6237c61a-9fb3-4b85-a23a-7d53dc3a38b3)

WIP:
- YOLOv9e with [gwern](https://gwern.net/crop#hands-download) + own data
- YOLOv9e with [gwern](https://gwern.net/crop#hands-download) + own data + [1-yshhi/anhdet](https://universe.roboflow.com/1-yshhi/anhdet) + [catwithawand/hand-detection-fuao9](https://universe.roboflow.com/catwithawand/hand-detection-fuao9)

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
