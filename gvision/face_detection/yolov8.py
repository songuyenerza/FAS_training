import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

class Face_Landmark:
    def __init__(self, model_dir: Path, device: int, image_size: int = 640, conf: float = 0.7, iou: float = 0.7) -> None:
        """
        instanciate the model.

        Parameters
        ----------
        model_dir : Path
            directory where to find the model weights.

        device : str
            the device name to run the model on.
        """
        self.model = YOLO(model_dir)
        self.image_size = image_size
        self.conf = conf
        self.iou = iou
        if device < 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda:{0}'.format(device)
        
        print('[YOLOv8-landmark] : DEVICE_ID = {0} - model path = {1}'.format(device, model_dir))

    def crop_image_with_padding(self, image, box, image_size, padding=0.2):
        xmin, ymin, xmax, ymax = box
        padding_percentage = padding

        # Calculate the width and height of the bounding box
        width = xmax - xmin
        height = ymax - ymin

        # Calculate the padding values
        padding_width = int(width * padding_percentage)
        padding_height = int(height * padding_percentage)

        # Add padding to the bounding box
        xmin -= padding_width
        ymin -= padding_height
        xmax += padding_width
        ymax += padding_height

        # Ensure padding is equal on all sides
        width_with_padding = xmax - xmin
        height_with_padding = ymax - ymin

        if width_with_padding > height_with_padding:
            diff = width_with_padding - height_with_padding
            ymin -= diff // 2
            ymax += diff // 2
        else:
            diff = height_with_padding - width_with_padding
            xmin -= diff // 2
            xmax += diff // 2

        # Ensure that the bounding box coordinates are within the image boundaries
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)

        # Recalculate width and height to ensure the box is still square within image bounds
        width = xmax - xmin
        height = ymax - ymin

        # If the adjusted box goes out of bounds, re-adjust to keep it square
        if width > height:
            ymax = ymin + width
            if ymax > image.shape[0]:
                ymax = image.shape[0]
                ymin = ymax - width
        else:
            xmax = xmin + height
            if xmax > image.shape[1]:
                xmax = image.shape[1]
                xmin = xmax - height

        # Ensure final box is within image boundaries
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)

        # Crop the image with square padding
        cropped_image = image[ymin:ymax, xmin:xmax].copy()

        cropped_image = cv2.resize(cropped_image, (image_size, image_size))

        return cropped_image

    def predict(self, input_image):# pragma: no cover
        """
        Get the predictions of a model on an input image.

        Args:
            model (YOLO): The trained YOLO model.
            input_image (Image): The image on which the model will make predictions.

        Returns:
            pd.DataFrame: A DataFrame containing the predictions.
        """
        try:
            # Make predictions
            predictions = self.model.predict(
                                imgsz=self.image_size,
                                source=input_image,
                                conf=self.conf,
                                iou=self.iou,
                                device=self.device,
                                verbose=False
                                )        
            keypoints = predictions[0].to("cpu").numpy().keypoints.xy
            list_boxes = []
            
            for i, box in enumerate(predictions[0].boxes):
                box_xyxy = [int(num) for num in box.xyxy[0]]
                width_box = box_xyxy[2] - box_xyxy[0]
                height_box = box_xyxy[3] - box_xyxy[1]
                box_xyxy = [box_xyxy[0], max(0, int(box_xyxy[1] - 0.15 * height_box )), box_xyxy[2], max(0, int(box_xyxy[3] - 0.02 * height_box))]

                list_boxes.append(box_xyxy)
            if len(list_boxes) > 0:
                return list_boxes[0]
            else:
                return None
        except:
            print(">>>>>> Predict Error .....")
            return None


# Face_detect = Face_Landmark(model_dir = "/data/datasets/thainq/sonnt373/dev/FAS/dev/Face-Anti-Spoofing/app/gvision/weights/yolov8n-face.pt",
#                         device = 0)

# image = cv2.imread("/data/datasets/thainq/sonnt373/dev/FAS/dev/Face-Anti-Spoofing/photo_2024-08-23_14-59-35.jpg")

# Face_detect.predict(image, padding)


# print("init ok")