import cv2
import queue
import threading
import ultralytics
from typing import Callable, Union
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
import datetime

class VideoTracker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    object_count_update = pyqtSignal(int)

    def __init__(self, model: ultralytics.YOLO, comparator: Callable[[np.ndarray], tuple], frame_rate: int = 30, scan_line: int = 100):
        """
        Initializes the VideoTracker.

        :param model: An instance of a YOLO model for object detection.
        :param comparator: A callable that compares a cropped image and returns a mask and a result.
        :param frame_rate: Frame rate for video processing.
        :param scan_line: Y-coordinate of the scan line for object detection.
        """
        super().__init__()
        self.model = model
        self.comparator = comparator
        self.frame_queue = queue.Queue(maxsize=10)
        self.processed_frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_rate = frame_rate
        self.object_count = 0
        self.scan_line = scan_line
        self.scan_line_top = scan_line
        self.scan_line_bottom = scan_line + 300

        # Initialize variables for resizing
        self.resized_width = 800
        self.resized_height = 600
        
        # Dynamic naming for the output video file
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.output_filename = f'data/processed_video_{timestamp}.mp4'
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.out = cv2.VideoWriter(self.output_filename, self.fourcc, self.frame_rate, (800, 600))

        self.setup_threads()

    def setup_threads(self):
        
        self.capture_thread = None
        self.process_thread = None
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)

    def start_tracking(self, source: Union[str, int]):
        """
        Starts the video tracking process.

        :param source: The video source (file path or camera index).
        """
        self.stop_event.clear()
        
        self.capture_thread = threading.Thread(target=self.capture_frames,args=(source,), daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
    
    def initialize_resized_dimensions(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.resized_width / width, self.resized_height / height)
        self.resized_width = int(width * scale)
        self.resized_height = int(height * scale)

    def capture_frames(self, source: Union[str, int]):
        """
        Captures frames from the video source.

        :param source: The video source (file path or camera index).
        """
        video = cv2.VideoCapture(source)
        # Read the first frame to get dimensions for resizing
        ret, frame = video.read()
        if not ret:
            self.frame_queue.put(None)
            return
        self.initialize_resized_dimensions(frame)
        
        while not self.stop_event.is_set():
            ret, frame = video.read()
            if not ret:
                self.frame_queue.put(None)
                break
            frame = self.resize_frame(frame)
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                continue
        video.release()

    def process_frames(self):
        """
        Processes frames from the frame queue.
        """
        seen_ids = set()
        while not self.stop_event.is_set():
            frame = self.get_frame_from_queue()
            if frame is None:
                self.processed_frame_queue.put(None)
                break
            processed_frame, results = self.track_objects(frame)
            self.frame_queue.task_done()
            self.put_processed_frame_in_queue(processed_frame)
            self.process_results_and_update_ids(results, seen_ids, frame)

    def get_frame_from_queue(self) -> Union[np.ndarray, None]:
        """
        Retrieves a frame from the frame queue.

        :return: A frame or None if the queue is empty.
        """
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def put_processed_frame_in_queue(self, processed_frame: np.ndarray):
        """
        Puts the processed frame in the processed frame queue.

        :param processed_frame: The processed frame.
        """
        try:
            self.processed_frame_queue.put(processed_frame, timeout=1)
        except queue.Full:
            pass

    def process_results_and_update_ids(self, results, seen_ids: set, frame: np.ndarray):
        """
        Processes detection results and updates object IDs.

        :param results: Detection results.
        :param seen_ids: Set of seen object IDs.
        :param frame: The current frame.
        """
        try:
            for frame_results in results:
                current_boxes = frame_results.obb.xyxy.cpu().numpy()
                current_ids = frame_results.obb.id.cpu().numpy()
                for box, id_ in zip(current_boxes, current_ids):
                    if id_ not in seen_ids:
                        self.process_new_detection(box, id_, seen_ids, frame)
        except Exception as e:
            print('Skipping Frame')
        
    def process_new_detection(self, box: np.ndarray, id_: int, seen_ids: set, frame: np.ndarray):
        """
        Processes a new detection and updates the necessary data structures.

        :param box: Bounding box coordinates.
        :param id_: Object ID.
        :param seen_ids: Set of seen object IDs.
        :param frame: The current frame.
        """
        x1, y1, x2, y2 = box.astype(int)
        height, width, _ = frame.shape
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return

        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return

        try:
            seen_ids.add(id_)
            self.object_count += 1
            self.object_count_update.emit(self.object_count)
        except Exception as e:
            print(f"Error in process_new_detection: {e}")

    def track_objects(self, frame: np.ndarray) -> tuple[np.ndarray, any]:
        """
        Tracks objects in the given frame.

        :param frame: The input frame.
        :return: A tuple of the processed frame and detection results.
        """
        try:
            results = self.model.track(frame, persist=True, vid_stride= 10)
            res_plotted = frame.copy()

            boxes = results[0].obb.xyxy.cpu().numpy().astype(int)
            ids = results[0].obb.id.cpu().numpy()

            for id_, box in enumerate(boxes):
                if box[3] > self.scan_line_top and box[3] < self.scan_line_bottom:
                    
                    x1, y1, x2, y2 = box
                    cropped_img = frame[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        continue

                    mask, result = self.comparator(cropped_img)
                    color = (0, 255, 0) if result == 'Good' else (0, 0, 255)
                    cv2.rectangle(res_plotted, (x1, y1), (x2, y2), color, 2)
                    
                    #write id on top left corner of the bounding box
                    cv2.putText(res_plotted, str(ids[id_]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    

                    # Draw mask on the detected object
                    mask = cv2.resize(mask, (x2 - x1, y2 - y1))
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    if res_plotted[y1:y2, x1:x2].shape == mask.shape:
                        res_plotted[y1:y2, x1:x2] = cv2.addWeighted(res_plotted[y1:y2, x1:x2], 0.5, mask, 0.5, 0)
                    else:
                        print("Sizes of res_plotted and mask don't match.")
        
            return res_plotted, results
        except Exception as e:
            print(f"Error in track_objects: {e}")
            return frame, None

    def stop_tracking(self):
        """
        Stops the video tracking process.
        """
        self.stop_event.set()
        # self.out.release()
        print('Video released')
        if self.capture_thread is not None:
            self.capture_thread.join()
        if self.process_thread is not None:
            self.process_thread.join()
        self.status_update.emit('Tracking stopped.')
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        # Resize the frame with the calculated dimensions
        resized_frame = cv2.resize(frame, (self.resized_width, self.resized_height), interpolation=cv2.INTER_AREA)

        return resized_frame

    @pyqtSlot()
    def update_frame(self):
        """
        Updates the frame in the GUI.
        """
        if not self.processed_frame_queue.empty():
            frame = self.processed_frame_queue.get()
            if frame is None:
                self.status_update.emit('Tracking completed.')
                self.stop_tracking()
                return
            
            
            frame = self.resize_frame(frame) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.frame_ready.emit(frame)
            # print("frame written")
            # self.out.write(frame)
                       
            self.processed_frame_queue.task_done()