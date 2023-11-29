from ultralytics import YOLO
import io
from PIL import Image
import numpy as np

class YoloService():
    def __init__(self):
        self.model = YOLO("yolov8m-seg.pt")


    def people_count(self, image_file):
        # Read the image file into a stream
        img_stream = io.BytesIO(image_file.file.read())

        # Convert to a Pillow image
        img_obj = Image.open(img_stream)

        # Convert to a NumPy array
        img_array = np.array(img_obj)
        results = self.model(img_array)

        class_ids = []

        people_count = 0

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for class_id in boxes.cls:
                class_ids.append(result.names[int(class_id)])
                if result.names[int(class_id)] == 'person':
                    people_count += 1
                

        return {"person_count": people_count}
    
