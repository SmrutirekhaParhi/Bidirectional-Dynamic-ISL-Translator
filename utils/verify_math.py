import os
import numpy as np
from sklearn.model_selection import train_test_split

KEYPOINTS_PATH = "keypoints"

print("========================================")
print("📊 PAPER DATASET VERIFICATION SCRIPT 📊")
print("========================================")

# 1. Count exactly what is saved on the hard drive
total_saved_files = 0
if os.path.exists(KEYPOINTS_PATH):
    for action in os.listdir(KEYPOINTS_PATH):
        action_path = os.path.join(KEYPOINTS_PATH, action)
        if os.path.isdir(action_path):
            files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
            total_saved_files += len(files)
            
original_collected = int(total_saved_files / 2)

print(f"🎥 Original Videos Collected: {original_collected}")
print(f"📂 Files saved on disk (Original + Mirrored): {total_saved_files}")

# 2. Simulate the Noise Augmentation that happens inside train_model.py
# Your code takes all files and duplicates them with Gaussian noise
total_after_noise = total_saved_files * 2
print(f"🧬 Total sequences after Noise Injection in memory: {total_after_noise}")
print(f"   -> Augmentation Factor: {total_after_noise / original_collected}x")

# 3. Simulate your exact Train/Test split
# We create dummy arrays of the same size just to let scikit-learn do the math
dummy_X = np.zeros(total_after_noise)
dummy_y = np.zeros(total_after_noise)

X_train, X_test, y_train, y_test = train_test_split(dummy_X, dummy_y, test_size=0.15, random_state=0)

print("\n--- TABLE II EXACT VALUES ---")
print(f"Training Sequences (85%): {len(X_train)}")
print(f"Validation / Test Sequences (15%): {len(X_test)}")
print("========================================")