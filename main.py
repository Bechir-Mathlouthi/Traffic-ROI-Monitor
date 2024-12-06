import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import yt_dlp
from threading import Thread, Lock
from queue import Queue
import os
from datetime import datetime

class ROITracker:
    def __init__(self):
        self.cap = None
        self.roi = None
        self.drawing = False
        self.roi_points = []
        self.tracking = False
        self.model = None
        self.fps = 0
        self.frame_time = time.time()
        self.vehicle_count = 0
        self.tracked_objects = {}
        self.last_frame_time = None
        # Threading components
        self.frame_queue = Queue(maxsize=2)
        self.processed_frame_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.running = False
        self.frame_lock = Lock()
        # ROI Info
        self.roi_info = {
            'coordinates': None,
            'area': 0,
            'active': False
        }
        # Screenshot directory
        self.screenshot_dir = 'screenshots'
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
    def initialize_youtube_stream(self, url):
        """Initialize YouTube live stream"""
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info['url']
            
            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened():
                raise ValueError("Error: Could not open YouTube stream")
            
            # Set optimal buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Reset timing variables
            self.last_frame_time = time.time()
            
        except Exception as e:
            print(f"Error initializing YouTube stream: {str(e)}")
            raise
            
    def initialize_model(self, model_path='yolov8n.pt'):
        """Initialize YOLO model"""
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            print("Using GPU for inference")
            self.model.to('cuda')
            
    def process_frames_thread(self):
        """Background thread for frame processing"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    if frame is not None and self.tracking:
                        processed = self.process_frame(frame)
                        if len(self.processed_frame_queue.queue) < 2:
                            self.processed_frame_queue.put(processed)
                    else:
                        if len(self.processed_frame_queue.queue) < 2:
                            self.processed_frame_queue.put(frame)
            except Exception as e:
                print(f"Error in processing thread: {str(e)}")
                continue
    
    def calculate_fps(self):
        """Calculate FPS with error handling"""
        try:
            current_time = time.time()
            if self.last_frame_time is None:
                self.last_frame_time = current_time
                return
            
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                self.fps = 1.0 / time_diff
            else:
                self.fps = 0
                
            self.last_frame_time = current_time
        except Exception as e:
            print(f"Error calculating FPS: {str(e)}")
            self.fps = 0
            
    def normalize_roi(self, points):
        """Normalize ROI coordinates"""
        if len(points) != 2:
            return None
        x1, y1 = points[0]
        x2, y2 = points[1]
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        
    def update_roi_info(self):
        """Update ROI information"""
        if self.roi:
            x, y, w, h = self.roi
            self.roi_info = {
                'coordinates': f"X: {x}, Y: {y}",
                'dimensions': f"Width: {w}, Height: {h}",
                'area': f"Area: {w * h}px²",
                'active': True
            }
        
    def take_screenshot(self, frame):
        """Save a screenshot with timestamp"""
        try:
            if frame is not None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(self.screenshot_dir, f'traffic_screenshot_{timestamp}.jpg')
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {str(e)}")
    
    def draw_roi_info(self, frame):
        """Draw ROI information on frame"""
        if self.roi:
            x, y, w, h = self.roi
            # Draw ROI rectangle with thicker line
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            # Draw semi-transparent overlay for ROI info
            info_bg = frame.copy()
            overlay_h = 180  # Increased height for new controls
            cv2.rectangle(info_bg, (0, 0), (300, overlay_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(info_bg, 0.3, frame, 0.7, 0)
            
            # Draw ROI information
            info_text = [
                "ROI Information:",
                f"Position: {self.roi_info['coordinates']}",
                f"Size: {self.roi_info['dimensions']}",
                f"Area: {self.roi_info['area']}",
                f"Status: {'Active' if self.tracking else 'Paused'}",
                f"Vehicles Detected: {self.vehicle_count}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw control instructions
            controls = [
                "Controls:",
                "R - Reset ROI",
                "S - Take Screenshot",
                "Q - Quit"
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (frame.shape[1] - 200, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            frame = param.copy()
            cv2.rectangle(frame, self.roi_points[0], (x, y), (0, 255, 0), 2)
            self.draw_roi_info(frame)  # Draw ROI info while drawing
            cv2.imshow('Traffic Monitoring', frame)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi_points.append((x, y))
            self.roi = self.normalize_roi(self.roi_points)
            self.update_roi_info()  # Update ROI info after selection
            self.tracking = True
    
    def process_frame(self, frame):
        """Process frame with object detection within ROI"""
        try:
            if self.roi is None or frame is None:
                return frame
                
            display_frame = frame.copy()
            
            if self.tracking:
                x, y, w, h = self.roi
                height, width = frame.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                roi_frame = frame[y:y+h, x:x+w]
                if roi_frame.size == 0:
                    return frame
                
                # Perform detection on ROI
                results = self.model(roi_frame, verbose=False)[0]
                
                current_objects = set()
                
                for det in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = det
                    if conf > 0.5:
                        class_id = int(cls)
                        if class_id in [2, 5, 3, 7]:  # vehicles only
                            x1, y1, x2, y2 = map(int, [x1 + x, y1 + y, x2 + x, y2 + y])
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            object_id = f"{center[0]}_{center[1]}"
                            current_objects.add(object_id)
                            
                            if object_id not in self.tracked_objects:
                                self.tracked_objects[object_id] = True
                                self.vehicle_count += 1
                            
                            # Draw detection box and label
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'{results.names[class_id]} {conf:.2f}'
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.tracked_objects = {obj: val for obj, val in self.tracked_objects.items()
                                      if obj in current_objects}
            
            # Always draw ROI info
            display_frame = self.draw_roi_info(display_frame)
            
            return display_frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame
    
    def run(self, youtube_url):
        """Main loop for the application"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print("Initializing YouTube stream...")
                self.initialize_youtube_stream(youtube_url)
                print("Initializing YOLO model...")
                self.initialize_model()
                
                # Start processing thread
                self.running = True
                self.processing_thread = Thread(target=self.process_frames_thread)
                self.processing_thread.daemon = True
                self.processing_thread.start()
                
                cv2.namedWindow('Traffic Monitoring')
                print("Ready! Click and drag to select a Region of Interest (ROI)")
                print("Press 'S' to take a screenshot")
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        print("Error reading frame. Retrying stream connection...")
                        break
                    
                    self.calculate_fps()
                    
                    # Add frame to processing queue
                    if len(self.frame_queue.queue) < 2:
                        self.frame_queue.put(frame)
                    
                    # Get processed frame or original frame
                    display_frame = None
                    if not self.processed_frame_queue.empty():
                        display_frame = self.processed_frame_queue.get()
                    else:
                        display_frame = frame.copy()
                        display_frame = self.draw_roi_info(display_frame)
                    
                    if not self.tracking:
                        cv2.setMouseCallback('Traffic Monitoring', 
                                          self.mouse_callback, display_frame)
                    
                    cv2.imshow('Traffic Monitoring', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        return
                    elif key == ord('r'):
                        print("Resetting ROI...")
                        self.roi = None
                        self.tracking = False
                        self.roi_points = []
                        self.vehicle_count = 0
                        self.tracked_objects = {}
                        self.roi_info = {'coordinates': None, 'area': 0, 'active': False}
                    elif key == ord('s'):
                        self.take_screenshot(display_frame)
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                continue
            finally:
                self.running = False
                if self.processing_thread:
                    self.processing_thread.join(timeout=1)
                if self.cap:
                    self.cap.release()
                
        cv2.destroyAllWindows()
        print("Maximum retries reached. Please check your internet connection and try again.")

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
    tracker = ROITracker()
    tracker.run(youtube_url) 