
🚨 Offline AI CCTV — Smart Surveillance Dashboard
Local-first AI CCTV system for real-time person detection, live people counting, snapshots, and video recording.
<img width="1024" height="1024" alt="image1" src="https://github.com/user-attachments/assets/00232b3b-07a6-4d86-9697-f2da8932b521" />
<img width="544" height="393" alt="image2" src="https://github.com/user-attachments/assets/503a426f-0780-4fc7-95e8-9f94633eff78" />
<img width="1320" height="826" alt="image3" src="https://github.com/user-attachments/assets/f00e4bbd-8023-4698-8b91-875722dba8fa" />




📸 Dashboard Preview
Important: Put your screenshot inside docs/images/dashboard.png.
Then this Markdown will display it automatically:
 
Other screenshots
Put your other screenshots in the same folder:
docs/
└── images/
    ├── dashboard.png
    ├── live-detection.png
    ├── recording.png
    └── alerts.png
Then add them to this README:
## Live Detection

![Live Person Detection](docs/images/live-detection.png)

## Video Recording

![Video Recording](docs/images/recording.png)

## Alert Gallery

![Alert Gallery](docs/images/alerts.png)
🎯 Project Overview
This project is an offline-first AI CCTV monitoring application built with Python and Streamlit.
The system uses a webcam/camera and a locally downloaded YOLOS-Tiny object detection model to detect people in the camera view.
It provides a modern dashboard for:
- 🎥 Live camera monitoring
- 👤 Person detection
- 🔢 Live people count
- 📊 Detection confidence
- 🔴 Video recording
- 📸 Automatic snapshots
- 📷 Manual snapshots
- 🗂️ Recording library
- 🖼️ Alert gallery
- ⚙️ Local system configuration
The core camera inference runs locally after the required model has been downloaded.
✨ Features
🎥 Live CCTV Monitoring
- Start/stop camera
- Live camera feed
- Person bounding boxes
- Detection confidence
- Live FPS information
- Current people count
👤 Person Detection
The YOLOS-Tiny model detects the person object category.
Example:
Person detected
Confidence: 98%
People in frame: 2
🔴 Video Recording
Users can start and stop recording directly from the dashboard.
Recordings are stored locally:
recordings/
└── cctv_recording_YYYYMMDD_HHMMSS.mp4
📸 Smart Snapshots
The application can save snapshots when people are detected.
Snapshots are stored in:
alerts/
└── person_YYYYMMDD_HHMMSS.jpg
Manual snapshots are also supported.
📊 Modern Dashboard
The Streamlit interface contains:
- Live camera feed
- People counter
- Camera status
- Recording status
- Recording duration
- Saved video count
- Recent events
- Quick actions
- Alert gallery
🏗️ System Architecture
                 ┌──────────────────┐
                 │   CCTV / Webcam  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   OpenCV Camera  │
                 │     Capture      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  YOLOS-Tiny AI   │
                 │ Person Detection │
                 └────────┬─────────┘
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   Live   │ │ Snapshot │ │  Video   │
        │ Dashboard│ │  Alerts  │ │ Recording│
        └──────────┘ └──────────┘ └──────────┘
              │           │            │
              └───────────┼────────────┘
                          ▼
                 ┌──────────────────┐
                 │ Local File System │
                 │ alerts/           │
                 │ recordings/       │
                 └──────────────────┘
🛠️ Technology Stack
Technology	Purpose
Python 3.11	Core programming
Streamlit	Web dashboard
OpenCV	Camera and video processing
PyTorch	AI model runtime
Transformers	Object detection pipeline
YOLOS-Tiny	Person detection
Pillow	Image processing
Hugging Face Hub	Model download


📁 Project Structure
offline-ai-cctv/
│
├── streamlit_app.py
├── live_cctv.py
├── model_test.py
├── webcam_test.py
├── download_models.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── yolos-tiny/
│
├── alerts/
│   └── *.jpg
│
├── recordings/
│   └── *.mp4
│
└── docs/
    └── images/
        ├── dashboard.png
        ├── live-detection.png
        ├── recording.png
        └── alerts.png
models/, alerts/, and recordings/ should normally be excluded from Git because models and CCTV footage can be large or private.

💻 Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/offline-ai-cctv.git
cd offline-ai-cctv
2. Create virtual environment
Windows:
py -3.11 -m venv venv
Activate:
.\venv\Scripts\Activate.ps1
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
🤖 Download AI Model
Connect to the internet for the initial model download:
python download_models.py
The model will be downloaded to:
models/yolos-tiny/
After the model is downloaded, person detection can run locally.
📹 Test Webcam
Run:
python webcam_test.py
If the camera opens successfully, continue to the AI test.
🧠 Test AI Detection
Run:
python model_test.py
Example:
Label: person
Confidence: 0.998
Box: {'xmin': 107, 'ymin': 115, 'xmax': 605, 'ymax': 478}

AI model test completed.
🚀 Start Dashboard
Run:
python -m streamlit run streamlit_app.py
Open:
http://localhost:8501
🎮 Dashboard Usage
1. Start Camera
Click:
▶ Start camera
The live camera feed will appear.
2. Detect People
The AI will draw bounding boxes around detected people.
3. Start Recording
Click:
🔴 Start recording
The recording will be saved to:
recordings/
4. Stop Recording
Click:
■ Stop recording
The MP4 file is finalized and stored locally.
5. Take Snapshot
Click:
📸 Take snapshot now
The image is saved to:
alerts/
📊 Current Limitations
This is currently a demo/learning project, not a certified security system.
Current limitations include:
- Single camera configuration
- CPU inference by default
- Current-frame people count rather than unique-person tracking
- No persistent database for events
- No authentication system
- No multi-camera management
- Detection accuracy depends on lighting and camera position
- No face recognition
- No reliable entry/exit tracking yet
🚀 Future Roadmap
Phase 1 — Smart Alerts
- Telegram notifications
- Snapshot sent when a person enters camera view
- Alert cooldown
- Alert delivery status
Phase 2 — Better AI
- Object tracking
- Unique person counting
- Entry/exit detection
- Restricted-area detection
- Line crossing detection
Phase 3 — Multi-Camera
Camera 01 ──┐
Camera 02 ──┤
Camera 03 ──┼──► AI Processing ──► Dashboard
Camera 04 ──┘
Phase 4 — Client Dashboard
- User login
- Role-based access
- Multiple clients
- Multiple cameras
- Camera health monitoring
- Event search
- Analytics
- PDF/CSV reports
Phase 5 — Production
- PostgreSQL/SQLite event database
- Background workers
- GPU acceleration
- Automatic recording retention
- Disk-space monitoring
- Docker deployment
- HTTPS
- Authentication and audit logs
🔐 Privacy & Security
This project processes camera footage, so privacy should be considered from the beginning.
- Only use cameras where you have permission to record.
- Do not commit private CCTV images or videos to GitHub.
- Never commit API keys, passwords, bot tokens, or .env files.
- Secure access to recorded footage.
- Define an appropriate data-retention period.
- YOLOS-Tiny detects people; it does not identify people by name.
⚡ Performance
The current implementation uses CPU inference.
Performance can be improved by:
- Reducing input resolution
- Running inference every few frames
- Using a smaller/faster detector
- Using GPU acceleration when available
- Separating camera capture and AI inference
- Moving notification sending to a background worker
- Using a dedicated streaming component instead of frequent UI reruns
📜 License
MIT License
See the LICENSE file for the full license text.
👨‍💻 Author
Amit Saini
AI / ML Developer
Python · Computer Vision · Generative AI · Backend Development
⭐ If you find this project useful, consider giving the repository a star.
