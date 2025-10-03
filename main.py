import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import pandas as pd
from datetime import datetime
import os
import threading
import queue
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedFaceAttendance:
    def __init__(self):
        # Performance settings
        self.camera_width = 640
        self.camera_height = 480
        self.detection_size = (160, 160)  # Much smaller for speed
        self.max_faces_per_frame = 1  # Process only one face per frame
        self.processing_interval = 2  # Process every 2nd frame
        
        # Initialize model with optimizations
        logger.info("Loading optimized face recognition model...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']  # Force CPU since CUDA not available
        )
        
        # Use smaller model and disable unnecessary features
        self.app.prepare(
            ctx_id=-1,  # Force CPU
            det_size=self.detection_size,
            det_thresh=0.5  # Higher threshold for fewer detections
        )
        
        # Face database
        self.face_db = {}
        self.attendance_file = "attendance.csv"
        self.similarity_threshold = 0.6  # Lower threshold for better recognition
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # Attendance
        self.attendance_df = self.load_attendance()
        self.marked_today = set()
        self.load_marked_today()
        
        # Performance
        self.frame_count = 0
        self.last_processing_time = 0
        
        logger.info("Optimized system initialized!")

    def load_attendance(self):
        if os.path.exists(self.attendance_file):
            return pd.read_csv(self.attendance_file)
        return pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])

    def load_marked_today(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if len(self.attendance_df) > 0:
            today_records = self.attendance_df[self.attendance_df['Date'] == today]
            self.marked_today = set(today_records['Name'].values)

    def save_attendance(self):
        self.attendance_df.to_csv(self.attendance_file, index=False)

    def register_face(self, image_path, person_name):
        """Fast face registration with resizing"""
        if not os.path.exists(image_path):
            return False
        
        # Load and resize image for faster processing
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Resize large images
        if img.shape[1] > 800:
            scale = 800 / img.shape[1]
            new_width = 800
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        faces = self.app.get(img)
        if len(faces) == 0:
            logger.warning(f"No face detected in {image_path}")
            return False
        
        # Use the most prominent face (largest)
        largest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        self.face_db[person_name] = {
            'embedding': largest_face.embedding,
            'image_path': image_path
        }
        
        logger.info(f"Registered: {person_name}")
        return True

    def register_faces_from_folder(self, folder_path):
        supported_formats = ('.jpg', '.jpeg', '.png')
        registered_count = 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_formats):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(folder_path, filename)
                if self.register_face(image_path, name):
                    registered_count += 1
        
        logger.info(f"Successfully registered {registered_count} faces")

    def compare_faces_fast(self, embedding1, embedding2):
        """Optimized cosine similarity"""
        # Normalize once and reuse if needed
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        return np.dot(emb1_norm, emb2_norm)

    def recognize_face_fast(self, face_embedding):
        """Fast recognition with pre-computed normalized embeddings"""
        best_match = None
        highest_similarity = 0
        
        for name, data in self.face_db.items():
            # Pre-computed normalized embedding would be better here
            similarity = self.compare_faces_fast(face_embedding, data['embedding'])
            
            if similarity > highest_similarity and similarity > self.similarity_threshold:
                highest_similarity = similarity
                best_match = name
        
        return best_match, highest_similarity

    def mark_attendance(self, name):
        if name is None or name in self.marked_today:
            return False
        
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S")
        
        new_record = {
            'Name': name,
            'Date': date_str,
            'Time': time_str,
            'Status': 'Present'
        }
        
        self.attendance_df = pd.concat([self.attendance_df, pd.DataFrame([new_record])], ignore_index=True)
        self.marked_today.add(name)
        self.save_attendance()
        
        logger.info(f"Attendance marked for {name}")
        return True

    def camera_capture_thread(self):
        """Ultra-fast camera capture"""
        # Use MJPEG format for faster capture
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
        
        logger.info("Camera thread started")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed")
                break
            
            # Minimal processing - just flip
            frame = cv2.flip(frame, 1)
            
            # Always keep only the latest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            
            time.sleep(0.01)  # Small delay
        
        cap.release()
        logger.info("Camera thread stopped")

    def face_processing_thread(self):
        """Optimized face processing"""
        logger.info("Face processing thread started")
        
        last_recognition = None
        recognition_cooldown = 10  # seconds
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                current_time = time.time()
                
                # Skip processing if too frequent
                if current_time - self.last_processing_time < 0.1:  # 10 FPS max processing
                    continue
                
                self.last_processing_time = current_time
                
                # Process faces
                faces = self.app.get(frame)
                
                # Limit number of faces processed
                if len(faces) > self.max_faces_per_frame:
                    faces = faces[:self.max_faces_per_frame]
                
                results = {
                    'faces': [],
                    'frame': frame,
                    'timestamp': current_time
                }
                
                for face in faces:
                    bbox = face.bbox.astype(int)
                    
                    # Check cooldown
                    if last_recognition and (current_time - last_recognition < recognition_cooldown):
                        name = None
                        similarity = 0
                    else:
                        name, similarity = self.recognize_face_fast(face.embedding)
                        if name:
                            last_recognition = current_time
                            # Mark attendance in background
                            threading.Thread(
                                target=self.mark_attendance, 
                                args=(name,), 
                                daemon=True
                            ).start()
                    
                    results['faces'].append({
                        'bbox': bbox,
                        'name': name,
                        'similarity': similarity
                    })
                
                # Update results
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.result_queue.put_nowait(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
        
        logger.info("Face processing thread stopped")

    def display_thread(self):
        """Simple display thread with error handling"""
        logger.info("Display thread started")
        
        window_name = 'Face Attendance'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        last_frame = None
        
        while self.running:
            try:
                # Get latest results or use last frame
                try:
                    results = self.result_queue.get(timeout=0.5)
                    last_frame = results['frame'].copy()
                    
                    # Draw face boxes
                    for face_data in results['faces']:
                        bbox = face_data['bbox']
                        name = face_data['name']
                        
                        color = (0, 255, 0) if name else (0, 0, 255)
                        label = name if name else "Unknown"
                        
                        cv2.rectangle(last_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        
                        # Simple text
                        cv2.putText(last_frame, label, (bbox[0], bbox[1] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Add info
                    cv2.putText(last_frame, f"Registered: {len(self.face_db)}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(last_frame, f"Present Today: {len(self.marked_today)}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(last_frame, "Press 'q' to quit", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                except queue.Empty:
                    # No new frame, use last one
                    pass
                
                # Display frame if available
                if last_frame is not None:
                    cv2.imshow(window_name, last_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                    
                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
                    break
                    
            except Exception as e:
                logger.error(f"Display error: {e}")
                # Try to recreate window
                try:
                    cv2.destroyWindow(window_name)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                except:
                    pass
        
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
        
        logger.info("Display thread stopped")

    def run_system(self):
        """Run the optimized system"""
        if len(self.face_db) == 0:
            logger.error("No faces registered!")
            return
        
        logger.info("Starting optimized attendance system...")
        
        # Create threads
        threads = [
            threading.Thread(target=self.camera_capture_thread, daemon=True),
            threading.Thread(target=self.face_processing_thread, daemon=True),
            threading.Thread(target=self.display_thread, daemon=True)
        ]
        
        # Start threads
        for thread in threads:
            thread.start()
        
        # Main loop
        try:
            while self.running:
                time.sleep(0.1)
                
                # Check if all threads are alive
                if not all(thread.is_alive() for thread in threads):
                    logger.error("A thread died, shutting down...")
                    self.running = False
                    break
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.running = False
        finally:
            # Wait for threads
            logger.info("Shutting down...")
            for thread in threads:
                thread.join(timeout=2.0)
            
            self.view_attendance_summary()

    def view_attendance_summary(self):
        """Show final attendance"""
        if len(self.attendance_df) > 0:
            today = datetime.now().strftime("%Y-%m-%d")
            today_attendance = self.attendance_df[self.attendance_df['Date'] == today]
            
            print(f"\n📊 Today's Attendance ({today}):")
            print(today_attendance.to_string(index=False))
        else:
            print("No attendance records today.")

def main():
    print("🚀 Starting Optimized Face Recognition Attendance System")
    
    # Initialize system
    system = OptimizedFaceAttendance()
    
    # Register faces
    faces_folder = "registered_faces"
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)
        print(f"📁 Created '{faces_folder}' folder.")
        print("📸 Please add face images and restart.")
        return
    
    print("👤 Registering faces...")
    system.register_faces_from_folder(faces_folder)
    
    if len(system.face_db) == 0:
        print("❌ No faces found. Please add images to 'registered_faces' folder.")
        return
    
    print(f"✅ Registered {len(system.face_db)} people")
    
    # Run system
    system.run_system()

if __name__ == "__main__":
    main()
