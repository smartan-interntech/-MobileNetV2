import cv2
import numpy as np
from ultralytics import YOLO
import datetime
from urllib.parse import quote
import threading
import queue
import time

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

def extract_rtsp_cap(show_cap=False):
    """Initialize RTSP camera connection"""
    username = quote("admin")
    password = quote("admin@123")
    ip = "192.168.0.113"
    stream = "stream1"  # try "stream2" if this fails
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/{stream}"
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS
    
    if not cap.isOpened():
        print("Failed to connect to the camera.")
        return None
    
    print("Camera is available!")
    return cap

class FrameReader:
    """Thread-safe frame reader for RTSP stream"""
    def __init__(self, cap):
        self.cap = cap
        self.q = queue.Queue(maxsize=2)
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Remove old frame
                except queue.Empty:
                    pass
                    
            self.q.put(frame)
            
    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None
            
    def stop(self):
        self.running = False
        self.thread.join()

# Initialize RTSP camera
cap = extract_rtsp_cap()
if cap is None:
    exit()

# Initialize frame reader for better real-time performance
frame_reader = FrameReader(cap)
frame_reader.start()

# Get frame properties (use defaults for RTSP)
frame_width = 1920  # Default, will be updated from first frame
frame_height = 1080  # Default, will be updated from first frame

# Optional: Save output video (disable for better real-time performance)
SAVE_VIDEO = False  # Set to True if you want to save output
out = None

if SAVE_VIDEO:
    out = cv2.VideoWriter(
        "output_bicep_curls_rtsp.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,  # Lower FPS for RTSP
        (frame_width, frame_height)
    )

# Logging setup
start_time = datetime.datetime.now()
log_filename = f"bicep_curl_log_rtsp_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, "w")
log_file.write(f"RTSP Session started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write("Set | Rep | Side | Duration (s) | Timestamp | Method\n")
log_file.write("-----------------------------------------------------------\n")

# Exercise tracking variables
rep_count = 0
rep_start_time = None
direction = None
rep_durations = []

# Chest line method variables
left_wrist_above_line = False
right_wrist_above_line = False
prev_left_above = False
prev_right_above = False
chest_line_y = None

# Set and break tracking variables
set_count = 0
current_set_reps = 0
last_rep_time = None
break_threshold = 7  # seconds of inactivity to consider a break
in_break = False
break_start_time = None
total_reps = 0

# Performance tracking
frame_count = 0
fps_start_time = time.time()
current_fps = 0

def calculate_angle(a, b, c):
    a = np.array(a.cpu())
    b = np.array(b.cpu())
    c = np.array(c.cpu())
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def get_view_and_visible_arm(kp, confidence_threshold=0.3):
    """Simplified and more robust view detection"""
    left_shoulder = kp[5]
    right_shoulder = kp[6]
    left_elbow = kp[7]
    right_elbow = kp[8]
    left_wrist = kp[9]
    right_wrist = kp[10]
   
    shoulder_distance = abs(float(left_shoulder[0]) - float(right_shoulder[0]))
   
    left_arm_visible = (float(left_shoulder[0]) > 10 and float(left_shoulder[1]) > 10 and
                       float(left_elbow[0]) > 10 and float(left_elbow[1]) > 10 and
                       float(left_wrist[0]) > 10 and float(left_wrist[1]) > 10)
    right_arm_visible = (float(right_shoulder[0]) > 10 and float(right_shoulder[1]) > 10 and
                        float(right_elbow[0]) > 10 and float(right_elbow[1]) > 10 and
                        float(right_wrist[0]) > 10 and float(right_wrist[1]) > 10)
   
    if shoulder_distance < 80:
        if left_arm_visible:
            return "left_side", "left"
        elif right_arm_visible:
            return "right_side", "right"
        else:
            return "side_unclear", "none"
    else:
        if left_arm_visible and right_arm_visible:
            return "front", "both"
        elif left_arm_visible:
            return "front", "left"
        elif right_arm_visible:
            return "front", "right"
        else:
            return "front", "none"

def is_valid_keypoint(kp, indices):
    """Check if keypoints at given indices are valid (not zero)"""
    for idx in indices:
        if kp[idx][0] <= 0 or kp[idx][1] <= 0:
            return False
    return True

def calculate_chest_line(left_shoulder, right_shoulder):
    """Calculate chest line Y coordinate based on shoulders"""
    left_y = float(left_shoulder[1])
    right_y = float(right_shoulder[1])
    chest_y = (left_y + right_y) / 2 + 30
    return int(chest_y)

def check_wrist_above_chest_line(wrist, chest_line_y):
    """Check if wrist is above the chest line"""
    if chest_line_y is None:
        return False
    wrist_y = float(wrist[1])
    return wrist_y < chest_line_y

def start_new_set_if_needed():
    """Start a new set if we're not currently in one"""
    global set_count, current_set_reps, in_break
   
    if current_set_reps == 0:
        set_count += 1
        in_break = False
        log_file.write(f"--- SET {set_count} STARTED ---\n")
        log_file.flush()

def check_for_break_and_new_set(current_time):
    """Check if enough time has passed to consider it a break and start a new set"""
    global set_count, current_set_reps, last_rep_time, in_break, break_start_time, total_reps
   
    if last_rep_time is None:
        return False
   
    time_since_last_rep = (current_time - last_rep_time).total_seconds()
   
    if time_since_last_rep >= break_threshold:
        if not in_break and current_set_reps > 0:
            in_break = True
            break_start_time = current_time
           
            log_file.write(f"--- SET {set_count} COMPLETED: {current_set_reps} reps ---\n")
            log_file.write(f"BREAK | --- | --- | {time_since_last_rep:.1f} | {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Break\n")
            log_file.flush()
           
            current_set_reps = 0
            return True
   
    return False

session_start = datetime.datetime.now()
print("Starting RTSP bicep curl tracking... Press 'q' to quit")

try:
    while True:
        # Read frame from RTSP stream
        frame = frame_reader.read()
        if frame is None:
            print("No frame received, retrying...")
            time.sleep(0.1)
            continue

        # Update frame dimensions from first frame
        if frame_count == 0:
            frame_height, frame_width = frame.shape[:2]
            if SAVE_VIDEO and out is not None:
                out.release()
                out = cv2.VideoWriter(
                    "output_bicep_curls_rtsp.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    15,
                    (frame_width, frame_height)
                )

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            current_fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()

        # YOLO inference
        results = model.predict(frame, conf=0.5, verbose=False)  # Disable verbose for cleaner output
        annotated_frame = results[0].plot()
        keypoints = results[0].keypoints

        feedback = "No person detected"
        color = (0, 0, 255)
        detection_method = "None"

        if keypoints and keypoints.xy is not None and len(keypoints.xy) > 0:
            kp = keypoints.xy[0]
            view, visible_arm = get_view_and_visible_arm(kp)

            current_time = datetime.datetime.now()
            break_detected = check_for_break_and_new_set(current_time)

            try:
                left_shoulder = kp[5]
                left_elbow = kp[7]
                left_wrist = kp[9]
                right_shoulder = kp[6]
                right_elbow = kp[8]
                right_wrist = kp[10]

                # Calculate and draw chest line for RIGHT side view only
                if is_valid_keypoint(kp, [5, 6]) and view == "right_side":
                    chest_line_y = calculate_chest_line(left_shoulder, right_shoulder)
                    cv2.line(annotated_frame, (0, chest_line_y), (frame_width, chest_line_y), (255, 0, 255), 2)
                    cv2.putText(annotated_frame, "Chest Line", (10, chest_line_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                else:
                    chest_line_y = None

                active_angle = None
                side = None
               
                shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
                left_valid = is_valid_keypoint(kp, [5, 7, 9])
                right_valid = is_valid_keypoint(kp, [6, 8, 10])

                left_above_chest = check_wrist_above_chest_line(left_wrist, chest_line_y) if left_valid else False
                right_above_chest = check_wrist_above_chest_line(right_wrist, chest_line_y) if right_valid else False

                angle_based_detection = False
                chest_line_detection = False
               
                # ARM SELECTION AND ANGLE CALCULATION
                if view == "front":
                    if visible_arm == "both":
                        if left_valid:
                            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        else:
                            left_angle = 180
                       
                        if right_valid:
                            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        else:
                            right_angle = 180
                       
                        if left_angle < right_angle:
                            active_angle = left_angle
                            side = "Left"
                        else:
                            active_angle = right_angle
                            side = "Right"
                    elif visible_arm == "left" and left_valid:
                        active_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        side = "Left"
                    elif visible_arm == "right" and right_valid:
                        active_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        side = "Right"
                       
                elif "left" in view or view == "side_unclear":
                    if left_valid:
                        active_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        side = "Left"
                    elif right_valid:
                        active_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        side = "Right (Fallback)"
                       
                elif "right" in view:
                    if right_valid:
                        active_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                        side = "Right"
                    elif left_valid:
                        active_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        side = "Left (Fallback)"

                # ANGLE-BASED DETECTION
                if active_angle is not None and (view == "front" or view == "left_side" or view == "side_unclear"):
                    now = datetime.datetime.now()

                    if view == "front":
                        curl_threshold = 50
                        extension_threshold = 140
                    else:
                        curl_threshold = 70
                        extension_threshold = 110

                    if active_angle < curl_threshold:
                        if direction != "up":
                            direction = "up"
                            rep_start_time = now
                            feedback = "Good curl!"
                            color = (0, 255, 0)
                            detection_method = "Angle"
                            angle_based_detection = True
                        else:
                            feedback = "Hold the curl!"
                            color = (0, 255, 0)
                           
                    elif active_angle > extension_threshold:
                        if direction == "up" and rep_start_time is not None:
                            direction = "down"
                           
                            start_new_set_if_needed()
                           
                            current_set_reps += 1
                            total_reps += 1
                            rep_count = current_set_reps
                           
                            rep_duration = (now - rep_start_time).total_seconds()
                            rep_durations.append(rep_duration)
                            last_rep_time = now

                            log_file.write(f"{set_count} | {current_set_reps} | {side} | {rep_duration:.2f} | {now.strftime('%Y-%m-%d %H:%M:%S')} | Angle\n")
                            log_file.flush()
                           
                            feedback = f"Rep {current_set_reps} completed!"
                            color = (0, 255, 0)
                            detection_method = "Angle"
                            angle_based_detection = True
                           
                            rep_start_time = None
                            direction = None
                        else:
                            feedback = "Good extension!"
                            color = (0, 255, 0)
                           
                    elif curl_threshold <= active_angle <= 90:
                        feedback = "Go deeper!"
                        color = (0, 165, 255)
                    elif 90 < active_angle <= extension_threshold:
                        feedback = "Extend more!"
                        color = (0, 255, 255)

                # CHEST LINE-BASED DETECTION
                if chest_line_y is not None and view == "right_side" and not angle_based_detection:
                    now = datetime.datetime.now()
                   
                    left_crossed_up = left_above_chest and not prev_left_above and left_valid
                    right_crossed_up = right_above_chest and not prev_right_above and right_valid
                    left_crossed_down = not left_above_chest and prev_left_above and left_valid
                    right_crossed_down = not right_above_chest and prev_right_above and right_valid
                   
                    if (left_crossed_up or right_crossed_up):
                        if direction != "up":
                            direction = "up"
                            rep_start_time = now
                            feedback = "Curl detected (chest line)!"
                            color = (0, 255, 0)
                            detection_method = "Chest Line"
                            chest_line_detection = True
                            if left_crossed_up and right_crossed_up:
                                side = "Both"
                            elif left_crossed_up:
                                side = "Left"
                            else:
                                side = "Right"
                        else:
                            feedback = "Keep curling!"
                            color = (0, 255, 0)
                   
                    elif (left_crossed_down or right_crossed_down) and direction == "up" and rep_start_time is not None:
                        direction = "down"
                       
                        start_new_set_if_needed()
                       
                        current_set_reps += 1
                        total_reps += 1
                        rep_count = current_set_reps
                       
                        rep_duration = (now - rep_start_time).total_seconds()
                        rep_durations.append(rep_duration)
                        last_rep_time = now
                       
                        if left_crossed_down and right_crossed_down:
                            side = "Both"
                        elif left_crossed_down:
                            side = "Left"
                        else:
                            side = "Right"

                        log_file.write(f"{set_count} | {current_set_reps} | {side} | {rep_duration:.2f} | {now.strftime('%Y-%m-%d %H:%M:%S')} | Chest Line\n")
                        log_file.flush()
                       
                        feedback = f"Rep {current_set_reps} completed (chest line)!"
                        color = (0, 255, 0)
                        detection_method = "Chest Line"
                        chest_line_detection = True
                       
                        rep_start_time = None
                        direction = None
                   
                    elif direction == "up" and (left_above_chest or right_above_chest):
                        feedback = "Hold the curl!"
                        color = (0, 255, 0)
                    elif direction is None or direction == "down":
                        feedback = "Ready for next curl"
                        color = (255, 255, 0)
                   
                    prev_left_above = left_above_chest
                    prev_right_above = right_above_chest

                # Default feedback
                if not angle_based_detection and not chest_line_detection:
                    if (view == "front" or view == "left_side" or view == "side_unclear") and active_angle is not None:
                        feedback = f"Tracking: {int(active_angle)}°"
                        color = (255, 255, 0)
                    elif view == "right_side" and chest_line_y is not None:
                        feedback = "Using chest line detection"
                        color = (255, 255, 0)
                    else:
                        feedback = "Arm not clearly visible"
                        color = (0, 165, 255)

                # Debug overlay
                cv2.putText(annotated_frame, f"RTSP FPS: {current_fps:.1f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"View: {view}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Visible: {visible_arm}", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               
                if (view == "front" or view == "left_side" or view == "side_unclear") and active_angle is not None:
                    cv2.putText(annotated_frame, f"{side} Arm: {int(active_angle)}°", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
               
                if view == "right_side" and chest_line_y is not None:
                    cv2.putText(annotated_frame, f"L_Above: {left_above_chest} | R_Above: {right_above_chest}", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               
                # Exercise tracking display
                cv2.putText(annotated_frame, f"Set: {set_count} | Reps: {current_set_reps}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Total Reps: {total_reps}", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
               
                if in_break:
                    cv2.putText(annotated_frame, "BREAK TIME", (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                elif last_rep_time:
                    time_since_last = (datetime.datetime.now() - last_rep_time).total_seconds()
                    cv2.putText(annotated_frame, f"Time since last rep: {time_since_last:.1f}s", (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               
                cv2.putText(annotated_frame, feedback, (30, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            except (IndexError, Exception) as e:
                feedback = "[WARNING] Error processing keypoints"
                color = (0, 0, 255)
                cv2.putText(annotated_frame, feedback, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # Add FPS display even when no person detected
            cv2.putText(annotated_frame, f"RTSP FPS: {current_fps:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, feedback, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save frame if enabled
        if SAVE_VIDEO and out is not None:
            out.write(annotated_frame)

        # Display frame
        cv2.imshow("RTSP Bicep Curl Tracker", annotated_frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Cleanup
    frame_reader.stop()
    
    # Session summary
    session_end = datetime.datetime.now()
    total_time = (session_end - session_start).total_seconds()
    avg_rep_time = np.mean(rep_durations) if rep_durations else 0

    if current_set_reps > 0:
        log_file.write(f"--- SET {set_count} COMPLETED: {current_set_reps} reps ---\n")

    log_file.write("\nRTSP Session Summary\n")
    log_file.write("-----------------\n")
    log_file.write(f"Start Time: {session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"End Time: {session_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total Duration: {total_time:.2f} seconds\n")
    log_file.write(f"Total Sets: {set_count}\n")
    log_file.write(f"Total Reps: {total_reps}\n")
    log_file.write(f"Average Rep Time: {avg_rep_time:.2f} seconds\n")
    log_file.write(f"Average FPS: {frame_count/total_time:.1f}\n")

    if set_count > 0:
        log_file.write(f"Average Reps per Set: {total_reps/set_count:.1f}\n")

    cap.release()
    if out is not None:
        out.release()
    log_file.close()
    cv2.destroyAllWindows()
    
    print(f"[INFO] RTSP session complete. Log saved to {log_filename}")
    print(f"[INFO] Total reps: {total_reps}, Average FPS: {frame_count/total_time:.1f}")