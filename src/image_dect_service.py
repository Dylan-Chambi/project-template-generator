import mediapipe as mp
import io
import numpy as np
from PIL import Image
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse


class ImageDectService():
    def __init__(self):
        self.base_options = mp.tasks.BaseOptions(model_asset_path='efficientdet.tflite')
        self.object_detector_options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=self.base_options,
            max_results=5,
            running_mode=mp.tasks.vision.RunningMode.IMAGE)
        self.object_detector = mp.tasks.vision.ObjectDetector.create_from_options(self.object_detector_options)

    def people_count(self, image_file: UploadFile = File(...)):
        try:
            img_stream = io.BytesIO(image_file.file.read())
            
            image = Image.open(img_stream)

            numpy_image = np.array(image)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

            detection_result = self.object_detector.detect(mp_image)

            people_count = 0

            for detection in detection_result.detections:
                category = detection.categories[0]
                category_name = category.category_name
                if category_name == "person" and category.score > 0.3:
                    people_count += 1

            return JSONResponse(content={"person_count": people_count}, status_code=200)

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
