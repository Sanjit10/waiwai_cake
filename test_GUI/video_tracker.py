import cv2
import queue
import threading
import ultralytics
from typing import Callable, Union
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
import os

class VideoTracker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    object_count_update = pyqtSignal(int)

    def __init__(self, model: ultralytics.YOLO, comparator: Callable[[np.ndarray], str], frame_rate: int = 30, scan_line: int = 100):
        super().__init__()
        self.model = model
        self.comparator = comparator
        self.frame_queue = queue.Queue(maxsize=10)
        self.processed_frame_queue = queue.Queue(maxsize=10)
        self.crop_queue = queue.Queue(maxsize=50)
        self.stop_event = threading.Event()
        self.frame_rate = frame_rate
        self.object_count = 0
        self.detection_tags = {}
        self.scan_line = scan_line  # Y-coordinate of the scan line
        self.setup_threads()

    def setup_threads(self):
        self.capture_thread = None
        self.process_thread = None
        self.postprocess_thread = threading.Thread(target=self.postprocess_crops, daemon=True)
        self.postprocess_thread.start()

    def start_tracking(self, source: Union[str, int]):
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self.capture_frames, args=(source,), daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

    def capture_frames(self, source: Union[str, int]):
        video = cv2.VideoCapture(source)
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
        desired_width, desired_height = 800, 600
        height, width = frame.shape[:2]

        # Calculate the scale factor
        scale = min(desired_width / width, desired_height / height)

        # Resize the frame with the scale factor
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a new frame with the desired dimensions and black padding
        letterbox_frame = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

        # Calculate top-left corner for the resized frame
        top = (desired_height - new_height) // 2
        left = (desired_width - new_width) // 2

        # Place the resized frame into the letterbox frame
        letterbox_frame[top:top + new_height, left:left + new_width] = resized_frame

        return letterbox_frame

    def process_frames(self):
        seen_ids = set()
        while not self.stop_event.is_set():
            frame = self.get_frame_from_queue()
            if frame is None:
                self.processed_frame_queue.put(None)
                break
            processed_frame, results = self.track_objects(frame)
            self.frame_queue.task_done()
            ## code needs to go in here
            
            self.put_processed_frame_in_queue(processed_frame)
            self.process_results_and_update_ids(results, seen_ids, frame)

    def get_frame_from_queue(self) -> Union[np.ndarray, None]:
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def put_processed_frame_in_queue(self, processed_frame: np.ndarray):
        try:
            self.processed_frame_queue.put(processed_frame, timeout=1)
        except queue.Full:
            pass

    def process_results_and_update_ids(self, results, seen_ids: set, frame: np.ndarray):
        for frame_results in results:
            current_boxes = frame_results.obb.xyxy.cpu().numpy()
            current_ids = frame_results.obb.id.cpu().numpy()
            for box, id_ in zip(current_boxes, current_ids):
                if id_ not in seen_ids:
                    self.process_new_detection(box, id_, seen_ids, frame)

    def process_new_detection(self, box: np.ndarray, id_: int, seen_ids: set, frame: np.ndarray):
        x1, y1, x2, y2 = box.astype(int)
        height, width, _ = frame.shape
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return

        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return

        crop_filename = os.path.join('crop_folder', f'crop_{id_}.jpg')
        cv2.imwrite(crop_filename, cropped_img)
        result = self.comparator(cropped_img)
        try:
            self.crop_queue.put((crop_filename, cropped_img, result), timeout=1)
            seen_ids.add(id_)
            self.detection_tags[id_] = result
            self.object_count += 1
            self.object_count_update.emit(self.object_count)
        except queue.Full:
            pass

    def track_objects(self, frame: np.ndarray) -> tuple[np.ndarray, any]:
        results = self.model.track(frame, persist=True)
        res_plotted = frame.copy()

        boxes = results[0].obb.xyxy.cpu().numpy().astype(int)
        ids = results[0].obb.id.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            id_ = ids[i]
            if box[1] > self.scan_line:  # Check if the object's top boundary crosses the scan line
                if id_ in self.detection_tags:
                    color = (0, 255, 0) if self.detection_tags[id_] == 'Good' else (0, 0, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(res_plotted, (x1, y1), (x2, y2), color, 2)

        return res_plotted, results

    def postprocess_crops(self):
        crop_files = []
        while not self.stop_event.is_set():
            try:
                crop_filename, cropped_img, result = self.crop_queue.get(timeout=1)
                crop_files.append(crop_filename)
                if len(crop_files) > 50:
                    oldest_file = crop_files.pop(0)
                    if os.path.exists(oldest_file):
                        os.remove(oldest_file)
            except queue.Empty:
                continue

    def stop_tracking(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join()
        if self.process_thread is not None:
            self.process_thread.join()
        self.status_update.emit('Tracking stopped.')

    @pyqtSlot()
    def update_frame(self):
        if not self.processed_frame_queue.empty():
            frame = self.processed_frame_queue.get()
            if frame is None:
                self.status_update.emit('Tracking completed.')
                self.stop_tracking()
                return
            frame = self.postprocess_frame(frame)
            self.frame_ready.emit(frame)
            self.processed_frame_queue.task_done()

    def postprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
