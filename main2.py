import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import sqlite3
from datetime import datetime, date
import os
import threading
import queue
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedFaceAttendance:
    def __init__(self):
        # Performance settings for integrated graphics
        self.camera_width = 320  # Reduced resolution
        self.camera_height = 240
        self.detection_size = (128, 128)  # Smaller for speed
        self.max_faces_per_frame = 1
        self.processing_interval = 3  # Process every 3rd frame
        
        # Initialize model with optimizations
        logger.info("Loading ultra-light face recognition model...")
        
        # Try to use lighter model if available
        try:
            self.app = FaceAnalysis(
                name='buffalo_s',  # Try smaller model first
                providers=['CPUExecutionProvider']
            )
        except:
            # Fallback to buffalo_l if smaller not available
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
        
        # Optimized model preparation
        self.app.prepare(
            ctx_id=-1,
            det_size=self.detection_size,
            det_thresh=0.6,  # Higher threshold to reduce false detections
            rec_thresh=0.4   # Recognition threshold
        )
        
        # Face database with pre-computed normalized embeddings
        self.face_db = {}
        self.normalized_embeddings = {}
        
        # SQLite database
        self.db_path = "attendance.db"
        self.init_database()
        
        # Threading with smaller queues
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # Performance tracking
        self.frame_count = 0
        self.last_processing_time = 0
        self.processing_times = []
        
        logger.info("Optimized system initialized for integrated graphics!")

    def init_database(self):
        """Initialize SQLite database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date DATE NOT NULL,
                timestamp DATETIME NOT NULL,
                status TEXT DEFAULT 'Present',
                confidence REAL,
                UNIQUE(name, date)
            )
        ''')
        
        # Create face embeddings table for future use
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_date DATETIME NOT NULL
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_name_date ON attendance(name, date)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def save_attendance_to_db(self, name, confidence=None):
        """Save attendance record to database"""
        if name is None:
            return False
            
        current_time = datetime.now()
        today = current_time.date()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Use INSERT OR IGNORE to handle duplicates
            cursor.execute('''
                INSERT OR IGNORE INTO attendance (name, date, timestamp, status, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, today, current_time, 'Present', confidence))
            
            conn.commit()
            success = cursor.rowcount > 0
            
            if success:
                logger.info(f"✅ Attendance marked for {name} (Confidence: {confidence:.2f})")
            else:
                logger.debug(f"ℹ️ {name} already marked present today")
                
            return success
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def get_today_attendance(self):
        """Get today's attendance records"""
        today = date.today()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, timestamp, confidence 
            FROM attendance 
            WHERE date = ? 
            ORDER BY timestamp
        ''', (today,))
        
        records = cursor.fetchall()
        conn.close()
        return records

    def register_face(self, image_path, person_name):
        """Optimized face registration with memory management"""
        if not os.path.exists(image_path):
            return False
        
        try:
            # Load image with optimized settings
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return False
            
            # Resize large images to reduce processing time
            h, w = img.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Detect faces
            faces = self.app.get(img)
            if len(faces) == 0:
                logger.warning(f"No face detected in {image_path}")
                return False
            
            # Use the largest face
            largest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
            
            # Store normalized embedding for faster comparison
            embedding = largest_face.embedding
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            self.face_db[person_name] = {
                'embedding': embedding,
                'image_path': image_path
            }
            self.normalized_embeddings[person_name] = normalized_embedding
            
            # Save to database for persistence
            self.save_embedding_to_db(person_name, embedding)
            
            logger.info(f"✅ Registered: {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering face {person_name}: {e}")
            return False

    def save_embedding_to_db(self, name, embedding):
        """Save face embedding to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert numpy array to bytes
            embedding_blob = embedding.tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO face_embeddings (name, embedding, created_date)
                VALUES (?, ?, ?)
            ''', (name, embedding_blob, datetime.now()))
            
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving embedding to DB: {e}")
        finally:
            conn.close()

    def load_embeddings_from_db(self):
        """Load face embeddings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT name, embedding FROM face_embeddings')
            rows = cursor.fetchall()
            
            for name, embedding_blob in rows:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                self.face_db[name] = {'embedding': embedding}
                self.normalized_embeddings[name] = embedding / np.linalg.norm(embedding)
            
            logger.info(f"📥 Loaded {len(rows)} embeddings from database")
            
        except sqlite3.Error as e:
            logger.error(f"Error loading embeddings from DB: {e}")
        finally:
            conn.close()

    def register_faces_from_folder(self, folder_path):
        """Register faces from folder with progress tracking"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.webp')  # Added webp for smaller files
        registered_count = 0
        
        # Try to load existing embeddings first
        self.load_embeddings_from_db()
        
        for filename in os.listdir(folder_path):
            if not self.running:
                break
                
            if filename.lower().endswith(supported_formats):
                name = os.path.splitext(filename)[0]
                
                # Skip if already loaded from DB
                if name in self.face_db:
                    registered_count += 1
                    continue
                    
                image_path = os.path.join(folder_path, filename)
                if self.register_face(image_path, name):
                    registered_count += 1
                else:
                    logger.warning(f"Failed to register: {name}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
        
        logger.info(f"✅ Successfully registered {registered_count} faces")

    def compare_faces_optimized(self, embedding):
        """Ultra-optimized face comparison using pre-normalized embeddings"""
        if not self.normalized_embeddings:
            return None, 0
        
        # Normalize input embedding once
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        best_match = None
        highest_similarity = 0
        
        # Vectorized comparison for speed
        names = list(self.normalized_embeddings.keys())
        embeddings_list = list(self.normalized_embeddings.values())
        
        if embeddings_list:
            similarities = np.dot(embeddings_list, embedding_norm)
            best_idx = np.argmax(similarities)
            highest_similarity = similarities[best_idx]
            
            if highest_similarity > 0.5:  # Adjust threshold as needed
                best_match = names[best_idx]
        
        return best_match, highest_similarity

    def camera_capture_thread(self):
        """Optimized camera capture for integrated graphics"""
        cap = cv2.VideoCapture(0)
        
        # Optimized camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        
        # Try to set optimized codec
        for codec in ['MJPG', 'YUYV', '']:
            try:
                if codec:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
                break
            except:
                continue
        
        logger.info("📷 Camera thread started")
        
        last_frame_time = time.time()
        frame_interval = 1.0 / 8  # Target 8 FPS
        
        while self.running:
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
                
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Camera read failed")
                time.sleep(0.1)
                continue
            
            # Minimal processing
            frame = cv2.flip(frame, 1)
            
            # Always keep only the latest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put_nowait(frame)
                last_frame_time = current_time
            except queue.Full:
                pass
        
        cap.release()
        logger.info("📷 Camera thread stopped")

    def face_processing_thread(self):
        """Optimized face processing with memory management"""
        logger.info("🔍 Face processing thread started")
        
        last_recognition_time = 0
        recognition_cooldown = 5  # Reduced cooldown
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                current_time = time.time()
                
                # Skip processing if too frequent
                if current_time - self.last_processing_time < 0.2:  # 5 FPS max processing
                    continue
                
                self.last_processing_time = current_time
                self.frame_count += 1
                
                # Skip frames based on interval
                if self.frame_count % self.processing_interval != 0:
                    continue
                
                start_time = time.time()
                
                # Process faces
                faces = self.app.get(frame)
                
                results = {
                    'faces': [],
                    'frame': frame,
                    'processing_time': 0
                }
                
                for face in faces[:self.max_faces_per_frame]:  # Limit faces
                    bbox = face.bbox.astype(int)
                    
                    # Check cooldown
                    if current_time - last_recognition_time < recognition_cooldown:
                        name, similarity = None, 0
                    else:
                        name, similarity = self.compare_faces_optimized(face.embedding)
                        
                        if name and similarity > 0.6:  # Good match
                            last_recognition_time = current_time
                            # Mark attendance in background thread
                            threading.Thread(
                                target=self.save_attendance_to_db, 
                                args=(name, similarity), 
                                daemon=True
                            ).start()
                    
                    results['faces'].append({
                        'bbox': bbox,
                        'name': name,
                        'similarity': similarity
                    })
                
                results['processing_time'] = time.time() - start_time
                self.processing_times.append(results['processing_time'])
                
                # Keep only recent processing times
                if len(self.processing_times) > 10:
                    self.processing_times.pop(0)
                
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
                logger.error(f"❌ Processing error: {e}")
                time.sleep(0.1)
        
        logger.info("🔍 Face processing thread stopped")

    def display_thread(self):
        """Lightweight display thread"""
        logger.info("🖥️ Display thread started")
        
        window_name = 'Face Attendance - Press Q to quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        
        last_frame = None
        fps_counter = 0
        last_fps_time = time.time()
        
        while self.running:
            try:
                # Get latest results
                try:
                    results = self.result_queue.get(timeout=0.5)
                    frame = results['frame']
                    
                    # Draw face boxes and info
                    for face_data in results['faces']:
                        bbox = face_data['bbox']
                        name = face_data['name']
                        similarity = face_data['similarity']
                        
                        color = (0, 255, 0) if name else (0, 0, 255)
                        label = f"{name} ({similarity:.2f})" if name else "Unknown"
                        
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Performance info
                    avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = fps_counter / (current_time - last_fps_time)
                        fps_counter = 0
                        last_fps_time = current_time
                    
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Processing: {avg_processing_time*1000:.1f}ms", (10, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Registered: {len(self.face_db)}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    today_count = len(self.get_today_attendance())
                    cv2.putText(frame, f"Present Today: {today_count}", (10, 80), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    last_frame = frame
                
                except queue.Empty:
                    # Use last frame if no new data
                    frame = last_frame
                
                if frame is not None:
                    cv2.imshow(window_name, frame)
                
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
                logger.error(f"❌ Display error: {e}")
                time.sleep(0.1)
        
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
        
        logger.info("🖥️ Display thread stopped")

    def run_system(self):
        """Run the optimized system"""
        if len(self.face_db) == 0:
            logger.error("❌ No faces registered!")
            return
        
        logger.info("🚀 Starting optimized attendance system...")
        logger.info(f"📊 Registered faces: {list(self.face_db.keys())}")
        
        # Create and start threads
        threads = [
            threading.Thread(target=self.camera_capture_thread, daemon=True),
            threading.Thread(target=self.face_processing_thread, daemon=True),
            threading.Thread(target=self.display_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # Main monitoring loop
        try:
            while self.running:
                time.sleep(0.5)
                
                # Monitor thread health
                alive_threads = sum(thread.is_alive() for thread in threads)
                if alive_threads < len(threads):
                    logger.error("❌ Some threads died, shutting down...")
                    self.running = False
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
            self.running = False
        finally:
            # Clean shutdown
            logger.info("🛑 Shutting down...")
            self.running = False
            time.sleep(1)  # Give threads time to exit
            
            # Force thread termination if needed
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            self.view_attendance_summary()

    def view_attendance_summary(self):
        """Show attendance summary"""
        today_records = self.get_today_attendance()
        
        print(f"\n📊 Today's Attendance Summary ({date.today()}):")
        print("-" * 50)
        
        if today_records:
            for name, timestamp, confidence in today_records:
                time_str = timestamp.split(' ')[1] if ' ' in str(timestamp) else str(timestamp)
                print(f"👤 {name:20} | ⏰ {time_str:8} | ✅ {confidence:.2f}")
        else:
            print("No attendance records for today.")
        
        print(f"\nTotal present today: {len(today_records)}")

def main():
    print("🚀 Starting Ultra-Optimized Face Recognition Attendance System")
    print("💡 Designed for Integrated Graphics")
    
    # Initialize system
    system = OptimizedFaceAttendance()
    
    # Register faces
    faces_folder = "registered_faces"
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)
        print(f"📁 Created '{faces_folder}' folder.")
        print("📸 Please add face images (jpg, png, webp) and restart.")
        return
    
    print("👤 Registering faces...")
    system.register_faces_from_folder(faces_folder)
    
    if len(system.face_db) == 0:
        print("❌ No faces found. Please add images to 'registered_faces' folder.")
        return
    
    print(f"✅ Registered {len(system.face_db)} people: {list(system.face_db.keys())}")
    
    # Run system
    system.run_system()

if __name__ == "__main__":
    main()
