import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load model ===
model = load_model("frontraise_model.h5")

# === Preprocessing function ===
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Match training input size
    img = img / 255.0
    return img

# === Input video path ===
video_path = "/home/intern-tech/Downloads/WhatsApp Video 2025-07-30 at 12.35.42 PM.mp4"  # Change this to your actual video file
cap = cv2.VideoCapture(video_path)

# === Output video setup ===
output_path = "output_predicted_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# === Class labels (update with your actual class names) ===
class_names = ["Correct", "Incorrect"]  # or however many you have

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    input_img = preprocess_frame(frame)
    input_img = np.expand_dims(input_img, axis=0)

    # Predict
    prediction = model.predict(input_img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    # Annotate
    label = f"{class_names[class_id]} ({confidence:.2f})"
    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    out.write(frame)
    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import os
# from datetime import datetime
# from tensorflow.keras.models import load_model

# # === User Inputs ===
# exercise_name = input("Enter the exercise name (e.g., bicep_curl, front_raise): ").strip()
# use_webcam = input("Use webcam? (y/n): ").strip().lower() == 'y'

# # === Load Model ===
# model_path = f"{exercise_name}_model.h5"  # Example: front_raise_model.h5
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found: {model_path}")
# model = load_model(model_path)

# # === Class Labels ===
# class_names = ["Correct", "Incorrect"]  # Update based on your model

# # === Video Capture ===
# if use_webcam:
#     cap = cv2.VideoCapture(0)
# else:
#     video_path = input("Enter video file path: ").strip()
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video file not found: {video_path}")
#     cap = cv2.VideoCapture(video_path)

# # === Output video setup ===
# fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
# output_path = f"output_{exercise_name}.mp4"
# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# # === Log file ===
# log_file = open(f"{exercise_name}_prediction_log.txt", "w")
# log_file.write("Timestamp, Prediction, Confidence\n")

# # === Preprocessing Function ===
# def preprocess_frame(frame):
#     img = cv2.resize(frame, (224, 224))
#     img = img / 255.0
#     return img

# # === Prediction Loop ===
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     input_img = preprocess_frame(frame)
#     input_img = np.expand_dims(input_img, axis=0)

#     # Predict
#     prediction = model.predict(input_img, verbose=0)
#     class_id = np.argmax(prediction)
#     confidence = np.max(prediction)
#     label = f"{class_names[class_id]} ({confidence:.2f})"

#     # Log
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_file.write(f"{timestamp}, {class_names[class_id]}, {confidence:.2f}\n")

#     # Annotate & Display
#     cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#     out.write(frame)
#     cv2.imshow("Posture Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # === Cleanup ===
# cap.release()
# out.release()
# cv2.destroyAllWindows()
# log_file.close()
# print(f"Video saved to {output_path}")
# print(f"Log saved to {exercise_name}_prediction_log.txt")
