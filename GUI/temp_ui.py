import sys
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import ultralytics
import cv2
import queue
from typing import Tuple
from scripts.ColorFrequencySingle import ImageColorComparator  # Assuming you have a comparator module

class VideoTrackerApp(QMainWindow):
    """
    Video tracker application that detects objects in a video, classifies them using a rule-based comparator,
    and displays the results in real-time.
    """

    def __init__(self, model_path: str, frame_rate: int = 15, reference_image_path: str = "reference_image.jpg"):
        """
        Initialize the VideoTrackerApp.

        Args:
            model_path (str): Path to the YOLO model.
            frame_rate (int, optional): Frame rate of the video. Defaults to 15.
            reference_image_path (str, optional): Path to the reference image for the comparator.
        """
        super().__init__()
        self.model = ultralytics.YOLO(model_path)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self.processed_frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self.stop_event: threading.Event = threading.Event()
        self.frame_rate: int = frame_rate
        self.comparator: ImageColorComparator = ImageColorComparator(reference_image_path)
        self.initUI()
        self.setupThreads()

    def initUI(self) -> None:
        """
        Initialize the user interface.
        """
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

    def setupThreads(self) -> None:
        """
        Set up capture and processing threads.
        """
        self.capture_thread: threading.Thread = None
        self.process_thread: threading.Thread = None

    def select_video(self) -> None:
        """
        Open a file dialog to select a video file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.avi *.mp4)',
                                                   options=options)
        if file_name:
            self.status_label.setText(f'Tracking video: {file_name}')
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self.capture_frames, args=(file_name,), daemon=True)
            self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.capture_thread.start()
            self.process_thread.start()

    def capture_frames(self, filename: str) -> None:
        """
        Capture frames from the video file.

        Args:
            filename (str): Path to the video file.
        """
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

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Preprocessed frame.
        """
        height, width = frame.shape[:2]
        new_width = 800
        new_height = int(new_width * height / width)
        if new_height > 600:
            new_height = 600
            new_width = int(new_height * width / height)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame

    def process_frames(self) -> None:
        """
        Process frames from the frame queue.
        """
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            if frame is None:
                self.processed_frame_queue.put(None)
                break
            processed_frame = self.track_objects(frame)
            self.frame_queue.task_done()
            try:
                self.processed_frame_queue.put(processed_frame, timeout=1)
            except queue.Full:
                continue
        self.processed_frame_queue.task_done()

    def track_objects(self, frame: np.ndarray) -> np.ndarray:
        """
        Track objects in the frame, classify them, and annotate the frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        results = self.model.predict(frame, save_crop = True)
        for result in results.xyxy:
            bbox = result[0:4]
            cropped_object = self.crop_object(frame, bbox)
            classification = self.comparator.classify(cropped_object)
            color = (0, 255, 0) if classification == 'good' else (0, 0, 255)
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        return frame

    def crop_object(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop the object from the frame using the bounding box coordinates.

        Args:
            frame (np.ndarray): Input frame.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (x_min, y_min, x_max, y_max).

        Returns:
            np.ndarray: Cropped object.
        """
        x_min, y_min, x_max, y_max = bbox
        return frame.crop((x_min, y_min, x_max, y_max))

    def update_frame(self) -> None:
        """
        Update the frame displayed in the QLabel.
        """
        if not self.processed_frame_queue.empty():
            frame = self.processed_frame_queue.get()
            if frame is None:
                self.status_label.setText('Tracking completed.')
               
                self.stop_tracking()
                return
            self.display_frame(frame)
            self.processed_frame_queue.task_done()

    def display_frame(self, frame: np.ndarray) -> None:
        """
        Display the frame in the QLabel.

        Args:
            frame (np.ndarray): Input frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def stop_tracking(self) -> None:
        """
        Stop the tracking process.
        """
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join()
        if self.process_thread is not None:
            self.process_thread.join()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = VideoTrackerApp('models/best.pt')
    main_window.show()
    sys.exit(app.exec_())
