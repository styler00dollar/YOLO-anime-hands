# YOLOv8-anime-hands

![yolo](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/c0c820d2-24d8-4d4a-b452-ad8e20811275)

I used [gwerns hand dataset](https://gwern.net/crop) and trained YOLOv8 with [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). I modified the code to use Prodigy with 1 and applied batch 30. Afterwards I finetuned with a self-made custom dataset due to various reasons. The precision of annotations in gwerns dataset seems to be not that good and that is visible in the detection bounding box. A model that is trained with that data struggles with gloves, handshakes and more complex hands. Due to that, I created a dataset with 924 images that contains humanly drawn and diffusion images to make detections more accurate and more robust. I also tried to train nano and medium sized yolo, but that resulted in models with severe accuracy problems.

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

xl:
![results](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/ed921cff-5f54-418f-b1f3-83f69e87981d)

finetuned xl:
![results](https://github.com/styler00dollar/YOLOv8-anime-hands/assets/51405565/e684f351-ded3-460d-93c1-7f89df38049c)
