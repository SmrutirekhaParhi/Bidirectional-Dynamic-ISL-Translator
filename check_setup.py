import tensorflow as tf
import mediapipe as mp
import cv2

print(f"✅ TensorFlow Version: {tf.__version__}")
print(f"✅ MediaPipe Version: {mp.__version__}")
print(f"✅ OpenCV Version: {cv2.__version__}")

# Quick test of the "Brain"
print("Testing the AI Brain...")
try:
    tensor = tf.constant([[1, 2], [3, 4]])
    print("TensorFlow is responding!")
except:
    print("❌ TensorFlow failed.")

print("\n🎉 SYSTEM READY! You can start coding.")