import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import speech_recognition as sr  # NEW: For Speech-to-Text

# --- CONFIGURATION ---
model = load_model('action.h5')
actions = np.load('actions.npy') 
print(f"✅ Model Loaded. Classes: {actions}")

# 🎨 Colors for the probability bars (Blue-Green-Red style)
colors = [(245,117,16)] * len(actions) 

# ==========================================
# 🎤 NEW: SPEECH-TO-SIGN CONFIGURATION
# ==========================================
# Map your 100+ words to the video files in your 'videos' folder

video_database = {
    "baby": "videos/baby.mp4",
    "bad": "videos/bad.mp4",
    "big": "videos/big.mp4",
    "black": "videos/black.MOV",
    "blue": "videos/blue.MOV",
    "brother": "videos/brother.mp4",
    "brown": "videos/brown.MOV",
    "city": "videos/city.mp4",
    "cold": "videos/cold.mp4",
    "colour": "videos/colour.MOV",
    "color": "videos/colour.MOV",  # Synonym for US spelling
    "cool": "videos/cool.MOV",
    "court": "videos/court.mp4",
    "daughter": "videos/daughter.mp4",
    "dog": "videos/dog.mp4",
    "dry": "videos/dry.MOV",
    "fast": "videos/fast.mp4",
    "father": "videos/father.mp4",
    "fish": "videos/fish.mp4",
    "friday": "videos/friday.MOV",
    "good": "videos/good.mp4",
    "good afternoon": "videos/goodafternoon.MOV",
    "good evening": "videos/goodevening.MOV",
    "good morning": "videos/goodmorning.mp4",
    "good night": "videos/goodnight.mp4",
    "green": "videos/green.MOV",
    "grey": "videos/grey.MOV",
    "gray": "videos/grey.MOV",  # Synonym for US spelling
    "happy": "videos/happy.mp4",
    "healthy": "videos/healthy.mp4",
    "hello": "videos/hello.mp4",
    "horse": "videos/horse.mp4",
    "hot": "videos/hot.mp4",
    "house": "videos/house.mp4",
    "loud": "videos/loud.mp4",
    "man": "videos/man.mp4",
    "mother": "videos/mother.mp4",
    "mouse": "videos/mouse.MOV",
    "narrow": "videos/narrow.MOV",
    "new": "videos/new.mp4",
    "office": "videos/office.mp4",
    "old": "videos/old.mp4",
    "orange": "videos/orange.MOV",
    "parent": "videos/parent.MOV",
    "park": "videos/park.mp4",
    "pink": "videos/pink.MOV",
    "pleased": "videos/pleased.mp4",
    "quiet": "videos/quiet.mp4",
    "red": "videos/red.MOV",
    "restaurant": "videos/restaurant.mp4",
    "school": "videos/school.mp4",
    "sick": "videos/sick.mp4",
    "sister": "videos/sister.mp4",
    "slow": "videos/slow.mp4",
    "small": "videos/small.mp4",
    "son": "videos/son.mp4",
    "street": "videos/street.mp4",
    "thank you": "videos/thankyou.mp4",
    "they": "videos/they.mp4",
    "train station": "videos/trainstation.mp4",
    "university": "videos/university.mp4",
    "warm": "videos/warm.MOV",
    "we": "videos/we.mp4",
    "wet": "videos/wet.MOV",
    "white": "videos/white.MOV",
    "woman": "videos/woman.mp4",
    "women": "videos/woman.mp4",
    "yellow": "videos/yellow.MOV",
    "you": "videos/you.MOV",
    "young": "videos/young.mp4"
}

def play_video(video_path):
    """Plays the ISL video in the EXACT SAME window as the webcam."""
    cap_video = cv2.VideoCapture(video_path)
    
    # The exact name of your main OpenCV window
    window_name = 'ISL Translator | Press A for Audio | Press Q to Quit'
    
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret:
            break
            
        # 1. Resize the video frame to match your webcam dimensions (640x480)
        frame = cv2.resize(frame, (640, 480))
        
        # 2. Add a neat UI banner so they know it's the video playing
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, 'PLAYING TRANSLATION...', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 3. Show it in the SAME window
        cv2.imshow(window_name, frame)
        
        # Wait 30ms (approx 30fps). Press 'q' to skip video early.
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap_video.release()
    
def listen_and_play():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n" + "="*40)
        print("🎤 AUDIO MODE ACTIVATED: Speak now...")
        print("="*40)
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3) 
        except sr.WaitTimeoutError:
            print("No speech detected. Returning to camera...")
            return
        
    try:
        spoken_text = recognizer.recognize_google(audio).lower()
        print(f"🗣️ You said: '{spoken_text}'")
        
        word_found = False
        for word in video_database.keys():
            if word in spoken_text:
                print(f"✅ Match found! Playing: {word}.mp4")
                play_video(video_database[word])
                word_found = True
                break 
                
        if not word_found:
            print("❌ Word not found in dictionary.")

    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError:
        print("❌ Could not request results from the speech service.")
# ==========================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

# 📊 Visualization Function (Draws the bar chart)
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    
    # Get top 5 predictions
    top_indices = res.argsort()[-5:][::-1]
    
    for num, idx in enumerate(top_indices):
        prob = res[idx]
        
        # 1. SETUP: Define location
        x_start = 0
        y_start = 60 + num * 40
        y_end = 90 + num * 40
        
        # 2. BACKGROUND: Draw a Gray Box
        cv2.rectangle(output_frame, (x_start, y_start), (300, y_end), (50, 50, 50), -1)
        
        # 3. BAR: Draw the Prediction Bar (Orange)
        bar_width = int(prob * 300) 
        cv2.rectangle(output_frame, (x_start, y_start), (bar_width, y_end), (245, 117, 16), -1)
        
        # 4. TEXT: Write the name and percentage
        txt = f"{actions[idx]}: {int(prob*100)}%"
        cv2.putText(output_frame, txt, (5, y_end - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
    return output_frame

# --- VARIABLES ---
sequence = []
sentence = []
res_history = []  # 🧠 NEW: Array to hold the probability history for smoothing
threshold = 0.85  # Slightly stricter to prevent garbage text

cap = cv2.VideoCapture(0)
# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- SYSTEM READY ---")
print("Press 'A' to switch to Audio Mode (Speech-to-Sign)")
print("Press 'Q' to Quit the Application\n")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 2. Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # 3. Prediction Logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep only last 30 frames

        if len(sequence) == 30:
            # FAST INFERENCE: Replaced .predict() to kill the lag
            input_data = np.expand_dims(np.array(sequence), axis=0)
            res = model(input_data, training=False)[0].numpy()
            
            # SMOOTHING LOGIC: Average the last 5 probability arrays
            res_history.append(res)
            if len(res_history) > 5:
                res_history = res_history[-5:]
            
            smoothed_res = np.mean(res_history, axis=0)

            # VIZ: Draw bars using the SMOOTHED results so they stop flickering
            image = prob_viz(smoothed_res, actions, image, colors)

            best_class_index = np.argmax(smoothed_res)
            confidence = smoothed_res[best_class_index]
            predicted_action = actions[best_class_index]
            
            if confidence > threshold:
                # Optional: Ignore the idle class if you have one
                if predicted_action != "Nothing": 
                    if len(sentence) > 0:
                        if predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                    else:
                        sentence.append(predicted_action)

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        # 4. Display UI
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Update window title to show instructions
        cv2.imshow('ISL Translator | Press A for Audio | Press Q to Quit', image)

        # --------------------------------------------------
        # ⌨️ KEYBOARD TOGGLE LOGIC
        # --------------------------------------------------
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('a'):
            cv2.rectangle(image, (0, 200), (640, 280), (50, 50, 50), -1)
            cv2.putText(image, "LISTENING... Speak now!", (120, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('ISL Translator | Press A for Audio | Press Q to Quit', image)
            
            cv2.waitKey(100) 

            listen_and_play()
            
            sequence = []
            res_history = [] # Clear the smoothing buffer too
            print("\n📸 CAMERA MODE REACTIVATED.")

    cap.release()
    cv2.destroyAllWindows()