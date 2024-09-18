import torch
import torch.nn.functional as F
import cv2
import numpy as np
from nets.utils import get_model, load_pretrain
import time
import os
import random
from tqdm import tqdm


from gvision.face_anti_spoof.inference import AntiSpoofClassifier

from gvision.face_anti_spoof.anti_spoof_predict import Detection, CropImage
from gvision.face_detection.yolov8 import Face_Landmark

    
AntiSpoofClassifier_v0 =  AntiSpoofClassifier()


Imagecroper = CropImage()
# Face detection
FaceDetect = Face_Landmark(model_dir = "./gvision/weights/yolov8n-face.pt",
                        device = 0)

def face_crop(image, image_size = 224, padding = 0):
    status, bboxes, kpss = FaceDetect(image)
    aligned_face= None
    if len(kpss) > 0:
        aligned_face = face_align.norm_crop(image, kpss[0], image_size=200, padding = padding)
    return aligned_face


class ImageClassifier:
    def __init__(self, model_path, arch='resnet50', num_classes=2, input_size=224, device_id=0):
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        self.input_size = input_size
        self.model = self._load_model(model_path, arch, num_classes)
        
    def _load_model(self, model_path, arch, num_classes):
        print(f"=> creating model '{arch}'")
        model = get_model(arch, num_classes, False)
        model.cuda() if self.device.startswith("cuda") else model.cpu()
        load_pretrain(model_path, model)
        model.eval()  # Set the model to evaluation mode
        return model

    def _preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (self.input_size, self.input_size))  # Resize the image

        # Normalize the image
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert the numpy array to a tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # Change from HWC to CHW format
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor.type(torch.float32).to(self.device)

    def predict(self, image, threshold = 0.5):
        image_tensor = self._preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            output = self.model(image_tensor)

        classification_output = output[1]

        # Apply softmax to get probabilities
        probabilities = F.softmax(classification_output, dim=1)

        # Get the predicted class and its probability score
        #   predicted_class:::  0: live || 1: spoof
        # _, predicted_class = torch.max(probabilities, 1)
        # predicted_score = probabilities[0, predicted_class.item()]

        live_prob = probabilities[0, 0].item()  # Probability for class 0 (live)
        spoof_prob = probabilities[0, 1].item()  # Probability for class 1 (spoof)

        # Determine label based on the threshold
        if live_prob > threshold:
            label = 0  # live
            score = live_prob
        else:
            label = 1  # spoof
            score = spoof_prob
        return probabilities[0].cpu().numpy()


# Example usage:
model_path = '/home/thainq97/dev/cvpr2024-face-anti-spoofing-challenge/outputs/0909_LIVE_WEIGHT_1.0_addblur_resnet_finetune/resnet50_epoch002_acc1_86.7705.pth'
root = "/home/thainq97/dev/DATASET/FAS_datatest_v0"
labels_path = os.path.join(root, "labels.txt")

classifier = ImageClassifier(model_path)

total_images = 0
correct_predictions = 0
false_accepts = 0
false_rejects = 0
live_count = 0
spoof_count = 0
have_face = 0
count = 0
with open(labels_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    for line in tqdm(lines):
        items = line.strip().split()
        path = items[0]
        ground_truth_label = int(items[1])

        if 'tas2' in path:

            count += 1
            img_path = os.path.join(root, path)
            image_org = cv2.imread(img_path)


            # model 1
            image_bbox = FaceDetect.predict(image_org)

            # image_crop2_7 = image_cropper.crop(image_org, image_bbox, 2.7, 224, 224, True)
            # image_crop4_0 = image_cropper.crop(image_org, image_bbox, 4.0, 224, 224, True)
            # image = image_cropper.crop(image_org, [image_bbox[0] , max(0, image_bbox[1] - int(image_bbox[3] * 0.07)), image_bbox[2], image_bbox[3] ], 1.5, 224, 224, True)

            # face align
            # image = face_crop(image_org, image_size = 224, padding = 0.05)

            # image_crop2_0 = face_crop(image_org, image_size = 224, padding = 0.2)
            # image_crop2_7 = face_crop(image_org, image_size = 224, padding = 0.25)

            # image_crop4_0 = face_crop(image_org, image_size = 224, padding = 0.4)
            
            if image_bbox is not None:
                
                image = FaceDetect.crop_image_with_padding(image_org, image_bbox, image_size = 224, padding = 0.02)


                image_crop2_0 = FaceDetect.crop_image_with_padding(image_org, image_bbox, image_size = 224, padding = 0.4)
                image_crop2_7 = FaceDetect.crop_image_with_padding(image_org, image_bbox, image_size = 224, padding = 0.2)
                image_crop4_0 = FaceDetect.crop_image_with_padding(image_org, image_bbox, image_size = 224, padding = 0.5)

                have_face += 1
                cv2.imwrite("facecrop.jpg", image)
                cv2.imwrite("image_crop2_0.jpg", image_crop2_0)

                cv2.imwrite("image_crop2_7.jpg", image_crop2_7)
                cv2.imwrite("image_crop4_0.jpg", image_crop4_0)

                # Count total images
                total_images += 1

                result_1 = AntiSpoofClassifier_v0.predict_croped(image_crop2_7, image_crop4_0)

                # try:
                result_2 = classifier.predict(image)
                result_2 += classifier.predict(image_crop2_0)

                result_2 = result_2 / 2

                # except:
                #     print("ERROR::: img_path :: ", img_path)
                #     result_2 = result_1
                # print("result_2:: ", result_2)
                # print("img_path :: ", img_path)

                # predict_final = np.array(result_2) * (5/10) + (5/10) * np.array(result_1)
                predict_final = 10/10 * np.array(result_2)
                # predict_final = 20/20 * np.array(result_1)


                live_score = predict_final[0]
                threshold_2 = 0.5
                threshold_1 = 0.4

                if result_2[0] > threshold_2 and result_1[0] > threshold_1:
                    predicted_class = 0
                else:
                    predicted_class = 1


                # Check if the prediction is correct
                if predicted_class == ground_truth_label:
                    correct_predictions += 1
                else:
                    if predicted_class == 0 and ground_truth_label == 1:  # Spoof incorrectly classified as live
                        print("result_1: ", result_1)
                        print("result_2: ", result_2)
                        print("img_path :: ", img_path)
                        false_accepts += 1
                    elif predicted_class == 1 and ground_truth_label == 0:  # Live incorrectly classified as spoof
                        false_rejects += 1
                        print("result_1: ", result_1)
                        print("result_2: ", result_2)
                        print("img_path :: ", img_path)

                    
                if ground_truth_label == 1:
                    spoof_count += 1
                else:
                    live_count += 1

            far = 0
            frr = 0
            if count % 20 == 0:
                print(f"===> Have {have_face} Croped Face in {count} || {have_face/count}")
                if spoof_count  > 0:
                    far = false_accepts / spoof_count  # FAR: Spoof classified as live
                    print(f"False Acceptance Rate (FAR): {100 * far:.4f} % || {false_accepts} || {spoof_count}")
                if live_count > 0:
                    frr = false_rejects / live_count  # FRR: Live classified as spoof
                    print(f"False Rejection Rate (FRR): {100 * frr:.4f} % || {false_rejects} || {live_count}")
                accuracy = correct_predictions / total_images
                print(f"Accuracy: {1 - far/2 -  frr/2} %")
                print("----------------------------------------")
if spoof_count  > 0:
    far = false_accepts / spoof_count  # FAR: Spoof classified as live
    print(f"False Acceptance Rate (FAR): {100 * far:.4f} %")
if live_count > 0:
    frr = false_rejects / live_count  # FRR: Live classified as spoof
    print(f"False Rejection Rate (FRR): {100 * frr:.4f} %")

accuracy = correct_predictions / total_images
print(f"Accuracy: {accuracy:.4f}")
