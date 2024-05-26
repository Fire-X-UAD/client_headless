import sys
import os
import numpy as np
import cv2
import math
import socket
import threading
from time import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QLabel, QPushButton,
    QVBoxLayout, QWidget, QMessageBox
)
from pygrabber.dshow_graph import FilterGraph
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

# UDP Server setup
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class ObjectDetection:
    def __init__(self, capture_index, camera_type, model_path):
        self.capture_index = capture_index
        self.camera_type = camera_type
        self.model_path = model_path
        self.running = True

    def load_model(self, model_path):
        from ultralytics import YOLO
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        model = YOLO(model_path)  # load the YOLO model
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

        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        font = cv2.FONT_HERSHEY_SIMPLEX
        last_box = None
        buffered_frames = 0

        buffer_y1 = None
        buffer_x1 = None
        buffer_x2 = None
        buffer_y2 = None

        buffer_radius = 100
        manual_tracking_speed = 0.2

        posisi_bola = (None, None)

        while self.running:
            start_time = time()
            ret, frame = cap.read()

            if not ret:
                print("Camera disconnected, waiting for reconnection...")
                cap.release()
                while not cap.isOpened():
                    cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW)
                    time.sleep(1)
                print("Camera reconnected.")
                continue

            results = self.predict(frame)
            combined_img = self.plot_bboxes(results, frame)
            end_time = time()
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

            max_score = 0
            max_score_index = 0
            for i in range(len(scores)):
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_score_index = i

            ball_detected = False

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
                ball_detected = True

            elif last_box is not None and buffered_frames < 20:
                x1 = int(last_box[0])
                y1 = int(last_box[1])
                x2 = int(last_box[2])
                y2 = int(last_box[3])

                if buffer_y1 is None:
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
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)

                    if w > 2 and h > 2:
                        cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(cropped_img, "Bola", (x, y - 10), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                        posisi_bola = (buffer_x1 + x + int(w / 2), buffer_y1 + y + int(h / 2))
                        last_box = [
                            lerp(x1, posisi_bola[0] - int(w / 2), manual_tracking_speed),
                            lerp(y1, posisi_bola[1] - int(h / 2), manual_tracking_speed),
                            lerp(x2, posisi_bola[0] + int(w / 2), manual_tracking_speed),
                            lerp(y2, posisi_bola[1] + int(h / 2), manual_tracking_speed)
                        ]
                        buffered_frames += 1
                        ball_detected = True
                    else:
                        buffered_frames = 20
                else:
                    buffered_frames = 20

            if not ball_detected:
                sock.sendto(str(-1).encode(), (UDP_IP, UDP_PORT))
            else:
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                angle = math.degrees(math.atan2(posisi_bola[0] - frame_center_x, frame_center_y - posisi_bola[1]))
                sock.sendto(str(angle).encode(), (UDP_IP, UDP_PORT))
                cv2.line(combined_img, (int(frame_center_x), int(frame_center_y)), (int(posisi_bola[0]), int(posisi_bola[1])), (0, 255, 0), 2)

            cv2.imshow("Object Detection", combined_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class CameraSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Camera Selection')
        layout = QVBoxLayout()

        self.camera_label = QLabel('Select Camera:')
        self.camera_combo = QComboBox()
        self.list_cameras()
        self.type_label = QLabel('Select Camera Type:')
        self.type_combo = QComboBox()
        self.type_combo.addItems(['USB Camera', 'Web Camera'])
        self.model_label = QLabel('Select YOLO Model:')
        self.model_combo = QComboBox()
        self.list_models()
        self.start_button = QPushButton('Start Detection')
        self.start_button.clicked.connect(self.start_detection)

        layout.addWidget(self.camera_label)
        layout.addWidget(self.camera_combo)
        layout.addWidget(self.type_label)
        layout.addWidget(self.type_combo)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def list_cameras(self):
        graph = FilterGraph()
        cameras = graph.get_input_devices()
        for index, name in enumerate(cameras):
            self.camera_combo.addItem(f'{name} (Camera {index})')

    def list_models(self):
        models_folder = "models"
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        for file in os.listdir(models_folder):
            if file.endswith(".pt"):
                self.model_combo.addItem(file)

    def start_detection(self):
        camera_index = int(self.camera_combo.currentText().split(' ')[-1].strip('()'))
        camera_type = self.type_combo.currentText()
        model_path = os.path.join("models", self.model_combo.currentText())

        self.object_detection = ObjectDetection(camera_index, camera_type, model_path)
        self.detection_thread = threading.Thread(target=self.object_detection.start)
        self.detection_thread.start()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraSelectionWindow()
    window.show()
    sys.exit(app.exec())
