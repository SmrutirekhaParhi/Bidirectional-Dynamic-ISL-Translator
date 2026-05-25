# Bidirectional Dynamic Indian Sign Language (ISL) Translator

A real-time, fully bidirectional communication system designed to bridge the gap between the Deaf and Hard-of-Hearing (DHH) community and the hearing population. 

Unlike traditional systems that rely on unidirectional translation or computationally heavy 3D avatars, this project utilizes a **Stacked LSTM** model for dynamic gesture recognition and **Natural Language Processing (NLP)** to instantly retrieve real human ISL video demonstrations natively within a unified Web interface.

## Key Features
* **Sign-to-Text (Vision Pipeline):** Captures live webcam feeds, extracts 178-D spatiotemporal keypoints using MediaPipe Holistic, and processes them through a custom Stacked LSTM network to predict 38 dynamic ISL gestures.
* **Speech-to-Sign (Audio Pipeline):** Utilizes the Google Speech Recognition API to capture spoken English, applies NLP text normalization, and instantly maps the speech to a local dictionary of 68 pre-recorded human ISL `.mp4` videos from the INCLUDE dataset.
* **Heuristic Temporal Smoothing:** An engineered stabilization algorithm that requires a 5-frame consecutive consistency check (at ≥85% confidence) to prevent erratic text flickering during gesture transitions.
* **Edge-Computing Architecture:** Decoupled ML models running locally via a Flask REST API and WebRTC/MJPEG streaming, achieving sub-40ms latency without relying on expensive cloud GPUs.
* **Unified Web Interface:** A sleek HTML/CSS/JS frontend featuring a dark/light mode toggle, dynamic listening overlays, and seamless context-switching between camera and video playback.

## Tech Stack
* **Machine Learning & Vision:** TensorFlow, Keras, OpenCV, MediaPipe
* **Audio & NLP:** SpeechRecognition, PyAudio
* **Backend:** Python, Flask, REST API
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **Data Processing:** NumPy, Pandas, Matplotlib

## Dataset & Acknowledgements

The Speech-to-Sign module of this project was trained and tested using the **INCLUDE (A Large Scale Dataset for Indian Sign Language Recognition)**. 

Due to file size constraints and standard repository practices, the raw video files are not hosted in this repository. 

**To run the Speech-to-Sign pipeline locally:**
1. Request or download the INCLUDE dataset from its official source.
2. Place the required `.mp4` human demonstration videos directly into the `videos/` directory of this project.
3. Ensure the video filenames match the corresponding vocabulary terms used in the application.

*Reference: Sridhar, A., Ganesan, S. G., Kumar, P., & Khapra, M. M. (2020). INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. In Proceedings of the 28th ACM International Conference on Multimedia.*

## Model Performance
* **Architecture:** 3-Layer Stacked LSTM (128 -> 256 -> 128 nodes) with Dropout (0.2).
* **Dataset:** Hybrid dataset (Public HD Corpora + Local Webcam Recordings), augmented 4x using Gaussian noise injection and geometric mirroring.
* **Accuracy:** Reached an unseen validation accuracy of **98.66%** across 38 classes.

## Installation & Setup

### Prerequisites
* Python 3.8 or higher
* A working webcam and microphone

### Step-by-Step Setup
**1. Clone the repository**

git clone https://github.com/SmrutirekhaParhi/Bidirectional-Dynamic-ISL-Translator.git

cd Bidirectional-Dynamic-ISL-Translator

**2. Create and activate a virtual environment**

For Windows:
python -m venv myenv
myenv\Scripts\activate

For Mac/Linux:
python3 -m venv myenv
source myenv/bin/activate

**3. Install dependencies**

pip install -r requirements.txt

*(Note: If you encounter issues with PyAudio on Windows, you may need to install it via pipwin: pip install pipwin -> pipwin install pyaudio)*

**4. Run the application**

python app.py

**5. Access the Web UI**

Open your web browser and navigate to: http://localhost:5000

## Repository Structure

```text
 Bidirectional-Dynamic-ISL-Translator
 ┣ 📂 data/                 # Dataset directory
 ┃ ┣ 📂 videos/             # Directory for ISL .mp4 dataset files (ignored in repo)
 ┃ ┃ ┗  placeholder.txt
 ┃ ┗  .gitkeep
 ┣ 📂 models/               # Saved model weights and metrics
 ┃ ┣  action.h5           # Trained Stacked LSTM model
 ┃ ┣  actions.npy         # NumPy array of the 38 ISL class labels
 ┃ ┗  training_history.npy
 ┣ 📂 results/              # Evaluation metrics and performance graphs
 ┃ ┣  classification_report.txt
 ┃ ┣  graph_class_distribution.png
 ┃ ┣  graph_confusion_matrix.png
 ┃ ┣  graph_per_class_accuracy_vertical.png
 ┃ ┗  training_curves.png
 ┣ 📂 src/                  # Core ML pipeline scripts
 ┃ ┣  evaluate_model.py
 ┃ ┣  generate_chart.py
 ┃ ┣  process_data.py
 ┃ ┗  train_model.py
 ┣ 📂 static/               # CSS and JS files for the frontend
 ┃ ┣  main.js
 ┃ ┗  style.css
 ┣ 📂 templates/            # HTML templates for Flask
 ┃ ┗  index.html
 ┣ 📂 utils/                # Helper functions and debugging scripts
 ┃ ┣  check_data.py
 ┃ ┣  check_setup.py
 ┃ ┣  debug_path.py
 ┃ ┣  test_opencv.py
 ┃ ┣  test_realtime.py
 ┃ ┗  verify_math.py
 ┣  .gitignore            # Ignored files and virtual environments
 ┣  LICENSE               # MIT License
 ┣  README.md             # Project documentation
 ┣  app.py                # Main Flask server and API endpoints
 ┗  requirements.txt      # Python dependencies
```
*(Note: Training datasets (`/data`), raw keypoints (`/keypoints`), and video files (`/videos`) have been excluded from this repository due to file size constraints.)*

## Usage Instructions
1. **Sign-to-Text:** Simply stand in front of the webcam and perform any of the 38 trained ISL gestures. The predicted translation will appear on the top banner.
2. **Speech-to-Sign:** Press the **[A]** key on your keyboard or click the "Activate Audio Mode" button on the UI. The webcam will dim, and the system will actively listen to your microphone. Speak a supported word (e.g., "Hello", "Good morning", "Thank you"), and the corresponding ISL video will play instantly.

## License
This project is licensed under the MIT License.
