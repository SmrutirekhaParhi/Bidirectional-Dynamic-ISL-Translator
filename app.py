from flask import Flask, render_template, Response, jsonify, send_from_directory
import threading
import cv2
import numpy as np
import os
import time  
import mediapipe as mp
from tensorflow.keras.models import load_model
import speech_recognition as sr

app = Flask(__name__)

# --- CONFIGURATION ---
model = load_model('action.h5')
actions = np.load('actions.npy')
print(f"✅ Model Loaded. Classes: {len(actions)}")

colors = [(0, 140, 255)] * len(actions)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

video_database = {
    "baby": "baby.mp4", "bad": "bad.mp4", "big": "big.mp4", "brother": "brother.mp4",
    "cold": "cold.mp4", "court": "court.mp4", "daughter": "daughter.mp4", "fast": "fast.mp4",
    "father": "father.mp4", "fish": "fish.mp4", "good": "good.mp4", "good morning": "goodmorning.mp4",
    "good night": "goodnight.mp4", "happy": "happy.mp4", "healthy": "healthy.mp4", "hello": "hello.mp4",
    "horse": "horse.mp4", "hot": "hot.mp4", "house": "house.mp4", "loud": "loud.mp4",
    "man": "man.mp4", "new": "new.mp4", "office": "office.mp4", "old": "old.mp4",
    "park": "park.mp4", "pleased": "pleased.mp4", "quiet": "quiet.mp4", "restaurant": "restaurant.mp4",
    "school": "school.mp4", "sick": "sick.mp4", "sister": "sister.mp4", "slow": "slow.mp4",
    "small": "small.mp4", "son": "son.mp4", "street": "street.mp4", "thank you": "thankyou.mp4",
    "they": "they.mp4", "train station": "trainstation.mp4", "university": "university.mp4",
    "we": "we.mp4", "woman": "woman.mp4", "young": "young.mp4"
}

# --- OPTIMIZED THREADED CAMERA CLASS ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                # 🚨 BULLETPROOF FIX: Force exactly 640x480 in software so the bars ALWAYS fit perfectly
                frame = cv2.resize(frame, (640, 480))
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01) 

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
        return self.grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        
def extract_keypoints(results):
    if results.pose_landmarks:
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        pose = []
        for i in range(13): 
            res = results.pose_landmarks.landmark[i]
            pose.append([res.x - nose_x, res.y - nose_y, res.z, res.visibility])
        pose = np.array(pose).flatten()
    else:
        pose = np.zeros(13 * 4) 

    if results.left_hand_landmarks and results.pose_landmarks:
        lh = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3) 

    if results.right_hand_landmarks and results.pose_landmarks:
        rh = np.array([[res.x - nose_x, res.y - nose_y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3) 

    return np.concatenate([pose, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    top_indices = res.argsort()[-5:][::-1]
    for num, idx in enumerate(top_indices):
        prob = res[idx]
        x_start, y_start = 0, 60 + num * 40 # Restored size
        y_end = 90 + num * 40
        cv2.rectangle(output_frame, (x_start, y_start), (300, y_end), (50, 50, 50), -1)
        bar_width = int(prob * 300) 
        cv2.rectangle(output_frame, (x_start, y_start), (bar_width, y_end), colors[idx], -1)
        txt = f"{actions[idx]}: {int(prob*100)}%"
        cv2.putText(output_frame, txt, (5, y_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame

def generate_frames():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.85  
    
    camera = ThreadedCamera(0).start()
    frame_counter = 0  
    last_res = np.zeros(len(actions)) 
    best_class_index = 0

    red_points = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    green_lines = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as holistic:
        while True:
            success, frame = camera.read()
            if not success or frame is None: 
                time.sleep(0.01)
                continue
            
            frame_counter += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, red_points, green_lines)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, red_points, green_lines)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                if frame_counter % 3 == 0:
                    input_data = np.expand_dims(np.array(sequence), axis=0)
                    last_res = model(input_data, training=False)[0].numpy()
                    best_class_index = np.argmax(last_res)
                    
                    predictions.append(best_class_index)
                    if len(predictions) > 5: predictions = predictions[-5:]

                    if np.unique(predictions[-5:])[0] == best_class_index and last_res[best_class_index] > threshold:
                        if len(sentence) > 0:
                            if actions[best_class_index] != sentence[-1]:
                                sentence.append(actions[best_class_index])
                        else:
                            sentence.append(actions[best_class_index])

            if len(sentence) > 5: sentence = sentence[-5:]

            if np.sum(last_res) > 0: 
                image = prob_viz(last_res, actions, image, colors)
                
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (640, 50), (40, 40, 40), -1) # Restored size
            image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
            display_text = ' '.join(sentence)
            cv2.putText(image, display_text, (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            ret, buffer = cv2.imencode('.jpg', image, encode_param)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.01)
                   
    camera.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/listen', methods=['POST'])
def listen():
    print("▶️ /listen endpoint was hit by the frontend!") # Debug 1
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("🎤 Microphone opened. Adjusting for noise...") # Debug 2
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("👂 Listening for up to 3 seconds...") # Debug 3
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
            
            print("📡 Audio captured! Sending to Google API...") # Debug 4
            spoken_text = recognizer.recognize_google(audio).lower()
            
            print(f"🗣️ SUCCESS! User said: '{spoken_text}'")
            
            for word, filename in video_database.items():
                if word in spoken_text:
                    print(f"✅ MATCH FOUND: {word}")
                    return jsonify({"status": "success", "word": word, "video_url": f"/videos/{filename}"})
                    
            print("❌ Word spoken is not in the video_database.")
            return jsonify({"status": "not_found", "message": "Word not in dictionary."})
            
    except sr.WaitTimeoutError:
        print("⚠️ TIMEOUT: No speech was detected in 3 seconds.")
        return jsonify({"status": "error", "message": "No speech detected."})
    except sr.UnknownValueError:
        print("⚠️ UNKNOWN VALUE: Google heard something, but couldn't understand the words.")
        return jsonify({"status": "error", "message": "Could not understand audio."})
    except sr.RequestError as e:
        print(f"⚠️ API ERROR: Could not reach Google. Check Wi-Fi. Error: {e}")
        return jsonify({"status": "error", "message": "API unavailable. Check internet."})
    except Exception as e:
        print(f"🔥 FATAL ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)