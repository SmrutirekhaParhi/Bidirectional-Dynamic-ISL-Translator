import cv2
import numpy as np
import os
import mediapipe as mp

# --- CONFIGURATION ---
DATA_PATH = "data"          
EXPORT_PATH = "keypoints"   
# ---------------------

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    # 1. Isolate the Upper Body (Landmarks 0 to 12) + Hands
    if results.pose_landmarks:
        # Get the Nose coordinates to act as our mathematical "Center" (0,0)
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        
        pose = []
        for i in range(13): # Only grab Face, Shoulders, and Arms
            res = results.pose_landmarks.landmark[i]
            pose.append([res.x - nose_x, res.y - nose_y, res.z, res.visibility])
        pose = np.array(pose).flatten()
    else:
        pose = np.zeros(13 * 4) # 52 values

    # 2. Extract and Normalize Hands (Relative to the Nose)
    if results.left_hand_landmarks and results.pose_landmarks:
        lh = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3) # 63 values

    if results.right_hand_landmarks and results.pose_landmarks:
        rh = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3) # 63 values

    return np.concatenate([pose, lh, rh])

def process_videos():
    print(f"🚀 STARTING SMART PROCESSING from '{DATA_PATH}'...")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if not os.path.exists(DATA_PATH):
            print(f"❌ Error: Folder '{DATA_PATH}' not found.")
            return

        actions = os.listdir(DATA_PATH)
        
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.isdir(action_path): continue

            # DEBUG: Print what we are checking
            all_files = os.listdir(action_path)
            videos = [f for f in all_files if f.lower().endswith(('.mp4', '.mov', '.avi'))]
            
            print(f"\n📂 Action: '{action}' | Found {len(videos)} videos")

            if len(videos) == 0:
                print(f"   ⚠️ SKIPPING: No videos found inside '{action}'")
                continue

            for video_name in videos:
                video_path = os.path.join(action_path, video_name)
                save_dir = os.path.join(EXPORT_PATH, action)
                os.makedirs(save_dir, exist_ok=True)
                
                file_stem = os.path.splitext(video_name)[0]
                npy_path = os.path.join(save_dir, file_stem + ".npy")
                
                # --- SMART RESUME LOGIC ---
                # Check if file exists AND is valid (has frames)
                if os.path.exists(npy_path):
                    try:
                        existing_data = np.load(npy_path)
                        if existing_data.shape[0] > 0:
                            # Only print every 5th skip to reduce clutter
                            # print(f"   ⏩ Skipped (Already Valid): {video_name}") 
                            continue 
                        else:
                            print(f"   ♻️ Re-doing (File empty): {video_name}")
                    except:
                        print(f"   ♻️ Re-doing (File corrupt): {video_name}")

                # --- IF WE GET HERE, WE NEED TO PROCESS ---
                cap = cv2.VideoCapture(video_path)
                frames_data = []

                if not cap.isOpened():
                    print(f"   ❌ ERROR: Could not open {video_name}")
                    continue

                # Process Frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    keypoints = extract_keypoints(results)
                    frames_data.append(keypoints)
                cap.release()
                # Save Data
                if len(frames_data) > 0:
                    data_array = np.array(frames_data)
                    np.save(os.path.join(save_dir, file_stem), data_array)
                    
                    # Mirror Magic
                    # Mirror Magic (Updated for 178 Relative Coordinates)
                    mirrored_data = data_array.copy()
                    for i in range(len(mirrored_data)):
                        frame = mirrored_data[i]
                        # Pose X (13 landmarks * 4 values = 52)
                        for j in range(0, 52, 4): 
                            frame[j] = -frame[j] 
                        # Hands X (42 landmarks * 3 values = 126, starting at index 52)
                        for j in range(52, 178, 3): 
                            frame[j] = -frame[j]
                    
                    np.save(os.path.join(save_dir, file_stem + "_mirrored"), mirrored_data)
                    print(f"   ✅ Processed: {video_name}")
                else:
                    print(f"   ⚠️ WARNING: 0 frames read from {video_name}")

    print("\n DONE! All videos processed.")

if __name__ == "__main__":
    process_videos()