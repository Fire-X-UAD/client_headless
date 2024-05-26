import sys
import numpy as np
import cv2
import time
import math
import threading
import socket
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QLabel, QPushButton,
    QVBoxLayout, QWidget, QMessageBox
)
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
import argparse
from cv2_enumerate_cameras import enumerate_cameras

# UDP Server setup
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_data(data: str):
    sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

class ObjectDetection:
    def __init__(self, capture_index, detect_mode, model_path, resolution):
        self.capture_index = capture_index
        self.detect_mode = detect_mode
        self.model_path = model_path
        self.resolution = resolution
        self.running = True

    def load_model(self, model_path):
        from ultralytics import YOLO
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        model = YOLO(model_path)  # load a pretrained YOLOv8n model
        model.fuse()
        return model
    
    def start(self):
        self.model = self.load_model(self.model_path)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=0.7)
        self.__call__()

    def predict(self, frame):
        results = self.model(frame, iou=0.2, conf=0.6)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)

        return frame

    def extract_data(self, results):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        return boxes, scores, class_ids

    def stop(self):
        self.running = False

    def __call__(self):

        def lerp(a: float, b: float, t: float) -> float:
            return (1 - t) * a + t * b

        # if windows use cv2.CAP_DSHOW else CAP_ANY 
        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        font = cv2.FONT_HERSHEY_SIMPLEX

        last_box = None
        buffered_frames = 0

        buffer_y1 = None
        buffer_x1 = None
        buffer_x2 = None
        buffer_y2 = None

        buffer_radius = 300
        manual_tracking_speed = 0.4

        posisi_bola = (None, None)        

        while self.running:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                print("Camera disconnected, waiting for reconnection...")
                cap.release()
                while not cap.isOpened():
                    cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                    time.sleep(1)
                print("Camera reconnected.")
                continue

            results = self.predict(frame)
            combined_img = self.plot_bboxes(results, frame)
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(combined_img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            boxes, scores, class_ids = self.extract_data(results)
            new_boxes = []
            new_scores = []
            new_classids = []

            for i, v in enumerate(class_ids):
                if v == 0:
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
                    new_classids.append(class_ids[i])

            boxes, scores, class_ids = new_boxes.copy(), new_scores.copy(), new_classids.copy()

            max_score = 0
            max_score_index = 0
            for i in range(len(scores)):
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_score_index = i

            if len(boxes) > 0:
                x1 = int(boxes[max_score_index][0])
                y1 = int(boxes[max_score_index][1])
                x2 = int(boxes[max_score_index][2])
                y2 = int(boxes[max_score_index][3])
                posisi_bola = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                last_box = boxes[max_score_index]
                buffered_frames = 0
                buffer_y1 = None
                buffer_x1 = None
                buffer_x2 = None
                buffer_y2 = None
            elif last_box is not None and buffered_frames < 10:
                x1 = int(last_box[0])
                y1 = int(last_box[1])
                x2 = int(last_box[2])
                y2 = int(last_box[3])

                if buffer_y1 == None:
                    buffer_y1 = int((y1 + y2) / 2) - int(buffer_radius / 2) if int((y1 + y2) / 2) - int(
                        buffer_radius / 2) > 0 else 0
                    buffer_x1 = int((x1 + x2) / 2) - int(buffer_radius / 2) if int((x1 + x2) / 2) - int(
                        buffer_radius / 2) > 0 else 0
                    buffer_x2 = int((x1 + x2) / 2) + int(buffer_radius / 2) if int((x1 + x2) / 2) + int(
                        buffer_radius / 2) < frame.shape[1] else frame.shape[1]
                    buffer_y2 = int((y1 + y2) / 2) + int(buffer_radius / 2) if int((y1 + y2) / 2) + int(
                        buffer_radius / 2) < frame.shape[0] else frame.shape[0]

                cropped_img = combined_img[buffer_y1:buffer_y2, buffer_x1:buffer_x2]

                cv2.rectangle(combined_img, (buffer_x1, buffer_y1), (buffer_x2, buffer_y2), (0, 255, 255), 2)

                cv2.putText(combined_img, "Deteksi warna manual...", (buffer_x1, buffer_y1 - 10), font, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)

                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                ORANGE_MIN = np.array([0, 92, 192], np.uint8)
                ORANGE_MAX = np.array([5, 255, 255], np.uint8)
                ORANGE_MIN2 = np.array([174, 92, 192], np.uint8)
                ORANGE_MAX2 = np.array([179, 255, 255], np.uint8)
                mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
                mask2 = cv2.inRange(hsv, ORANGE_MIN2, ORANGE_MAX2)
                mask = cv2.bitwise_or(mask, mask2)
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=3)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Find the biggest contour (the orange object)
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)

                    if w > 2 and h > 2:
                        cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(cropped_img, "Orange", (x, y), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

                        posisi_bola = (buffer_x1 + x + int(w / 2), buffer_y1 + y + int(h / 2))

                        buffer_x1_temp = int(
                            lerp(buffer_x1, buffer_x1 + x + int(w / 2) - int(buffer_radius / 2), manual_tracking_speed))
                        buffer_y1_temp = int(
                            lerp(buffer_y1, buffer_y1 + y + int(h / 2) - int(buffer_radius / 2), manual_tracking_speed))
                        buffer_x1 = buffer_x1_temp if buffer_x1_temp > 0 else 0
                        buffer_y1 = buffer_y1_temp if buffer_y1_temp > 0 else 0
                        buffer_x2 = buffer_x1 + buffer_radius if buffer_x1 + buffer_radius < frame.shape[1] else \
                        frame.shape[1]
                        buffer_y2 = buffer_y1 + buffer_radius if buffer_y1 + buffer_radius < frame.shape[0] else \
                        frame.shape[0]

                else: # Bola tidak terdeteksi
                    buffered_frames += 1
                    if buffered_frames > 5:
                        last_box = None
                        buffer_x1 = None
                        buffer_y1 = None
                        buffer_x2 = None
                        buffer_y2 = None
                        posisi_bola = (None, None)
            if posisi_bola[0] is not None and posisi_bola[1] is not None:
                if self.detect_mode == "Omnidirectional":
                    titik_tengah = (int(frame.shape[1] / 2), int(frame.shape[0] / 2)) # omni
                elif self.detect_mode == "Depan Normal":
                    titik_tengah = (int(frame.shape[1] / 2), int(frame.shape[0])) # depan
                    
                angle = -1
                if self.detect_mode != "360":
                    cv2.line(combined_img, titik_tengah, (posisi_bola[0], posisi_bola[1]), (0, 255, 0), 2, cv2.LINE_AA)
                    angle = float(math.atan2(titik_tengah[1] - posisi_bola[1], titik_tengah[0] - posisi_bola[0]) * 180 / math.pi)
                    angle = angle - 90 if angle > 90 else angle + 270
                else:
                    cv2.line(combined_img, (posisi_bola[0], 0), (posisi_bola[0], frame.shape[0]), (0, 255, 0), 2, cv2.LINE_AA)
                    angle = float((posisi_bola[0]/frame.shape[1])*360)
                    angle = angle - 180 if angle > 180 else angle + 180



                angle = round(angle, 2) 
                # show angle to screen+
                cv2.putText(combined_img, str(angle), (10, 300), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                angle = -1

            send_data(str(angle))

            cv2.imshow(f'YOLOv8 Detection kamera {self.capture_index}', combined_img)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()



class CameraSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Setting Vision')
        layout = QVBoxLayout()

        self.camera_label = QLabel('Pilih Kamera:')
        self.camera_combo = QComboBox()
        self.list_cameras()
        self.mode_label = QLabel('Pilih Mode Deteksi:')
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Depan Normal', 'Omnidirectional', '360'])
        self.model_label = QLabel('Pilih Model YOLO:')
        self.model_combo = QComboBox()
        self.list_models()
        self.resolution_label = QLabel('Pilih Resolusi:')
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(['1280x720', '640x480', '1920x906'])
        self.start_button = QPushButton('Mulai Deteksi')
        self.start_button.clicked.connect(self.start_detection)

        self.mode_combo.currentTextChanged.connect(self.update_mode)



        layout.addWidget(self.camera_label)
        layout.addWidget(self.camera_combo)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_combo)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_mode(self):
        if self.mode_combo.currentText() == '360':
            self.resolution_combo.setCurrentText('1920x906')
            self.model_combo.setCurrentText('best.pt')
            self.resolution_combo.setEnabled(False)
        elif self.mode_combo.currentText() == 'Omnidirectional':
            self.resolution_combo.setCurrentText('640x480')
            self.model_combo.setCurrentText('omni.pt')
            self.resolution_combo.setEnabled(True)
        else:
            self.resolution_combo.setCurrentText('1280x720')
            self.model_combo.setCurrentText('1_depan.pt')
            self.resolution_combo.setEnabled(True)
        
    def list_cameras(self):
        for index, name in enumerate(list_cameras_ext()):
            self.camera_combo.addItem(f'{name} (Kamera {index})')

    def list_models(self):
        models_folder = "models"
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        for file in os.listdir(models_folder):
            if file.endswith(".pt"):
                self.model_combo.addItem(file)

    def start_detection(self):
        camera_index = int(self.camera_combo.currentText().split(' ')[-1].strip('()'))
        detect_mode = self.mode_combo.currentText()
        model_path = os.path.join("models", self.model_combo.currentText())

        resolution = (int(self.resolution_combo.currentText().split('x')[0]), int(self.resolution_combo.currentText().split('x')[1]))

        self.object_detection = ObjectDetection(camera_index, detect_mode, model_path, resolution)
        self.detection_thread = threading.Thread(target=self.object_detection.start)
        self.detection_thread.start()
        self.close()

def list_cameras_ext():
    return list(enumerate_cameras(cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2))  

def headless():
    # ask for camera index
    print("Pilih Kamera: ")
    for index, name in enumerate(list_cameras_ext()):
        print(f"{index}: {name}")
    camera_index = int(input("Masukkan indeks kamera: "))
    print("Pilih Mode: ")
    print("1. Depan Normal")
    print("2. Omnidirectional")
    print("3. 360")
    mode = int(input("Masukkan pilihan mode: "))

    if mode == 1:
        detect_mode = "Depan Normal"
    elif mode == 2:
        detect_mode = "Omnidirectional"
    else:
        detect_mode = "360"

    print("Pilih Model: ")
    # list the models with number and just ask the user the number, starts from 1
    for index, file in enumerate(os.listdir("models")):
        if file.endswith(".pt"):
            print(f"{index+1}: {file}")
    model_index = int(input("Masukkan indeks model: ")) - 1
    model_path = os.path.join("models", os.listdir("models")[model_index])
    

    print("Pilih Resolusi: ")
    print("1. 1280x720")
    print("2. 640x480")
    print("3. 1920x906")
    resolution = int(input("Masukkan pilihan resolusi: "))

    if resolution == 1:
        resolution = (1280, 720)
    elif resolution == 2:
        resolution = (640, 480)
    else:
        resolution = (1920, 906)

    object_detection = ObjectDetection(camera_index, detect_mode, model_path, resolution)
    detection_thread = threading.Thread(target=object_detection.start)
    detection_thread.start()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Vision Client for FireX")
    parser.add_argument('--silent', '-q', action='store_true', help='run in headless mode')

    args = parser.parse_args()

    if args.silent:
        headless()
    else:
        app = QApplication(sys.argv)
        window = CameraSelectionWindow()
        window.show()
        sys.exit(app.exec())
