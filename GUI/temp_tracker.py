import os
import sys
import random
import numpy as np
import cv2
import queue
import threading
import ultralytics
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoTrackerApp(QMainWindow):
    def __init__(self, model_path, frame_rate=30):
        super().__init__()
        self.model = ultralytics.YOLO(model_path)
        self.frame_queue = queue.Queue(maxsize=20)
        self.processed_frame_queue = queue.Queue(maxsize=10)
        self.crop_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        self.frame_rate = frame_rate
        self.cropped_images = []
        self.max_saved_images = 50
        self.initUI()
        self.setupThreads()

    def initUI(self):
        self.setWindowTitle('Video Tracker')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.select_button = QPushButton('Select Video', self)
        self.select_button.clicked.connect(self.select_video)
        self.layout.addWidget(self.select_button)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.layout.addWidget(self.stop_button)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.status_label = QLabel('Select a video to start tracking.', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.frame_rate))

    def setupThreads(self):
        self.capture_thread = None
        self.process_thread = None
        self.crop_thread = threading.Thread(target=self.process_crop_queue, daemon=True)
        self.crop_thread.start()

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.avi *.mp4)', options=options)
        if file_name:
            self.status_label.setText(f'Tracking video: {file_name}')
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self.capture_frames, args=(file_name,), daemon=True)
            self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.capture_thread.start()
            self.process_thread.start()

    def capture_frames(self, filename):
        video = cv2.VideoCapture(filename)
        while not self.stop_event.is_set():
            ret, frame = video.read()
            if not ret:
                self.frame_queue.put(None)
                break
            frame = self.preprocess_frame(frame)
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                continue
        video.release()

    def preprocess_frame(self, frame):
        height, width = frame.shape[:2]
        new_width = 800
        new_height = int(new_width * height / width)
        if new_height > 600:
            new_height = 600
            new_width = int(new_height * width / height)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame

    def process_frames(self):
        seen_ids = set()  # Initialize set to store seen IDs
        while not self.stop_event.is_set():
            frame = self.get_frame_from_queue()
            if frame is None:
                self.processed_frame_queue.put(None)
                break
            processed_frame, results = self.track_objects(frame)
            self.frame_queue.task_done()
            self.put_processed_frame_in_queue(processed_frame)
            self.process_results_and_update_ids(results, seen_ids, frame)

    def get_frame_from_queue(self):
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def put_processed_frame_in_queue(self, processed_frame):
        try:
            self.processed_frame_queue.put(processed_frame, timeout=1)
        except queue.Full:
            pass

    def process_results_and_update_ids(self, results, seen_ids, frame):
        for frame_results in results:
            current_boxes = frame_results.obb.xyxy.cpu().numpy()
            current_ids = frame_results.obb.id.cpu().numpy()  # Get the IDs of detected objects
            for box, id_ in zip(current_boxes, current_ids):
                if id_ not in seen_ids:  # Check if ID is not seen before
                    self.process_new_detection(box, id_, seen_ids, frame)

    def process_new_detection(self, box, id_, seen_ids, frame):
        # This is a new detection
        x1, y1, x2, y2 = box.astype(int)
        # Ensure the bounding box is within the frame dimensions
        height, width, _ = frame.shape
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return

        cropped_img = frame[y1:y2, x1:x2]  # Crop image
        if cropped_img.size == 0:
            return

        crop_filename = os.path.join('/home/swordlord/crimson_tech/waiwai_cake/data/crop_folder', f'crop_{id_}.jpg')
        cv2.imwrite(crop_filename, cropped_img)  # Save cropped image
        try:
            self.crop_queue.put((crop_filename, cropped_img, random.choice(["Good", "Bad"])), timeout=1)  # Put cropped image into queue with random tag
        except queue.Full:
            pass
        seen_ids.add(id_)
        self.cropped_images.append(crop_filename)

        # Process and manage cropped images
        self.process_and_manage_cropped_images()

    def process_and_manage_cropped_images(self):
        while len(self.cropped_images) > self.max_saved_images:
            oldest_file = self.cropped_images.pop(0)
            self.dummy_process_and_delete(oldest_file)

    def dummy_process_and_delete(self, file_path):
        # Dummy processing with 5 milliseconds delay
        time.sleep(0.25)
        # Delete the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

    def process_crop_queue(self):
        while not self.stop_event.is_set():
            try:
                crop_filename, cropped_img, tag = self.crop_queue.get(timeout=1)
                self.dummy_process_and_delete(crop_filename)
                self.crop_queue.task_done()
            except queue.Empty:
                continue

    def track_objects(self, frame):
        results = self.model.track(frame, persist=True, max_det = 15 )
        res_plotted = results[0].plot()
        return res_plotted, results

    def update_frame(self):
        if not self.processed_frame_queue.empty():
            frame = self.processed_frame_queue.get()
            if frame is None:
                self.status_label.setText('Tracking completed.')
                self.stop_tracking()
                return
            frame = self.postprocess_frame(frame)
            self.display_frame(frame)
            self.processed_frame_queue.task_done()

    def postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def display_frame(self, frame):
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def stop_tracking(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join()
        if self.process_thread is not None:
            self.process_thread.join()
        if self.crop_thread is not None:
            self.crop_thread.join()
        self.status_label.setText('Tracking stopped.')
        self.video_label.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = VideoTrackerApp('models/best.pt')
    main_window.show()
    sys.exit(app.exec_())
