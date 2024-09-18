import numpy as np

from gvision.face_anti_spoof.anti_spoof_predict import AntiSpoofPredict
from gvision.face_detection.scrfd import FaceDetector
from gvision.face_detection import face_align

class AntiSpoofClassifier:
    def __init__(self, face_detection_model_path = "./gvision/weights/SCRFD_10G_KPS.onnx",
                face_detection_conf_threshold = 0.5,
                face_detection_index_gpu = 0,
                model_path_2_7 = "./gvision/weights/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
                model_path_4_0 = "./gvision/weights/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
                face_anti_spoof_index_gpu = 0,
                face_anti_spoof_conf_threshold = 0.7,
                ):
        
        if face_detection_index_gpu < 0:
            face_detection_use_gpu = False
        else:
            face_detection_use_gpu = True

        # Initialize model for face detection
        self.face_detector = FaceDetector(
            model_path=face_detection_model_path,
            threshold=face_detection_conf_threshold,
            useGpu=face_detection_use_gpu,
            index_gpu=face_detection_index_gpu
        )

        self.face_anti_spoof_conf_threshold = face_anti_spoof_conf_threshold

        # Initialize models for face anti-spoofing
        self.model_path_2_7 = model_path_2_7
        self.model_path_4_0 = model_path_4_0

        self.model_fas_2_7 = AntiSpoofPredict(device_id=face_anti_spoof_index_gpu)
        self.model_fas_2_7._load_model(self.model_path_2_7)

        self.model_fas_4_0 = AntiSpoofPredict(device_id=face_anti_spoof_index_gpu)
        self.model_fas_4_0._load_model(self.model_path_4_0)


    def predict(self, image):
        result = {
            "label": "",
            "score": 0
        }
        # 1: real ; 0: spoof
        prediction = np.zeros((1, 3))
        status, bboxes, kpss = self.face_detector(image)
        if len(bboxes) > 0:
            image_crop2_7 = face_align.norm_crop(image, kpss[0], image_size=80, padding=0.2)
            image_crop4_0 = face_align.norm_crop(image, kpss[0], image_size=80, padding=0.3)
            prediction += self.model_fas_2_7.inference(image_crop2_7)
            prediction += self.model_fas_4_0.inference(image_crop4_0)

            label = np.argmax(prediction)
            score = round(prediction[0][label] / 2, 2)

            if label == 1 and score > self.face_anti_spoof_conf_threshold:
                result["label"] = "live"
                result["score"] = score
            else:
                result["label"] = "spoofing"
                result["score"] = score
        else:
            print("Dont see face in image")

        return result

    def predict(self, image):
        result = {
            "label": "",
            "score": 0
        }

        prediction_output = np.zeros((1, 3))
        # 1: real ; 0: spoof
        prediction = np.zeros((1, 3))
        status, bboxes, kpss = self.face_detector(image)
        if len(bboxes) > 0:
            image_crop2_7 = face_align.norm_crop(image, kpss[0], image_size=80, padding=0.2)
            image_crop4_0 = face_align.norm_crop(image, kpss[0], image_size=80, padding=0.3)
            prediction += self.model_fas_2_7.inference(image_crop2_7)
            prediction += self.model_fas_4_0.inference(image_crop4_0)

            label = np.argmax(prediction)
            score = round(prediction[0][label] / 2, 2)

            prediction_output[0][0], prediction_output[0][1] = prediction[0][1], prediction[0][0] +  prediction[0][2]
            
            if label == 1 and score > self.face_anti_spoof_conf_threshold:
                result["label"] = "live"
                result["score"] = score
            else:
                result["label"] = "spoofing"
                result["score"] = score
        else:
            print("Dont see face in image")

        return prediction_output[0]

    
    def predict_croped(self, image_crop2_7, image_crop4_0):
        import cv2

        prediction_output = np.zeros((1, 2))
        # 1: real ; 0: spoof
        prediction = np.zeros((1, 3))

        image_crop2_7 = cv2.resize(image_crop2_7, (80, 80))
        image_crop4_0 = cv2.resize(image_crop4_0, (80, 80))



        prediction += self.model_fas_2_7.inference(image_crop2_7)
        # prediction += self.model_fas_2_7.inference(image_crop4_0)

        prediction += self.model_fas_4_0.inference(image_crop4_0)

        prediction_output[0][0], prediction_output[0][1] = prediction[0][1]/2, (prediction[0][0] +  prediction[0][2])/2
        
        return prediction_output[0]