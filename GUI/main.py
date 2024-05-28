from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from video_tracker import VideoTracker
import sys
import ultralytics
import numpy as np
import time
from all_masker import mask_image
import cv2

class VideoTrackerApp(QMainWindow):
    def __init__(self, model_path, frame_rate=30):
        super().__init__()
        self.model = ultralytics.YOLO(model_path)
        self.mask_image = mask_image('GUI/hsv_highlighter.pkl')
        self.tracker = VideoTracker(self.model, self.mask_image.highlighter, frame_rate=frame_rate)
        self.initUI()
        self.tracker_thread = QThread()
        self.tracker.moveToThread(self.tracker_thread)
        self.tracker_thread.start()

        self.tracker.frame_ready.connect(self.display_frame)
        self.tracker.status_update.connect(self.update_status)
        self.tracker.object_count_update.connect(self.update_object_count)

        self.prev_frame_time = None  # Initialize the previous frame time

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
        self.stop_button.clicked.connect(self.tracker.stop_tracking)
        self.layout.addWidget(self.stop_button)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        self.frame_label = QLabel(self)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.frame_label)

        self.status_label = QLabel('Select a video to start tracking.', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tracker.update_frame)
        self.timer.start(int(1000 / self.tracker.frame_rate))

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.avi *.mp4)', options=options)
        if file_name:
            self.status_label.setText(f'Tracking video: {file_name}')
            self.tracker.start_tracking(file_name)

    def display_frame(self, frame: np.ndarray):
        current_time = time.time()
        if self.prev_frame_time is not None:
            time_diff = current_time - self.prev_frame_time
            frame_rate = 1.0 / time_diff
            self.frame_label.setText(f'Frame Rate: {frame_rate:.2f} FPS')

        self.prev_frame_time = current_time

        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def update_status(self, status: str):
        self.status_label.setText(status)

    def update_object_count(self, count: int):
        self.status_label.setText(f'Total Objects Detected: {count}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = VideoTrackerApp('models/best.pt')
    main_window.show()
    sys.exit(app.exec_())
