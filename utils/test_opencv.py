import cv2
import os

# Specific file found in your debug screenshot
video_path = os.path.join("data", "Animal", "MVI_2999.MOV")

print(f"Testing Video: {video_path}")

if not os.path.exists(video_path):
    print("❌ Error: File does not exist at that path.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: OpenCV cannot open the file. (Codec/Format issue)")
else:
    print("✅ Success: OpenCV opened the file.")
    ret, frame = cap.read()
    if ret:
        print("✅ Success: Read the first frame!")
        print(f"   Frame Shape: {frame.shape}")
    else:
        print("❌ Error: Opened the file, but could not read any frames.")

cap.release()