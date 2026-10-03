import os
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from transformers import pipeline

# ---------------------------- Configuration ----------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "yolos-tiny"
ALERT_DIR = APP_DIR / "alerts"
VIDEO_DIR = APP_DIR / "recordings"
ALERT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
FRAME_WIDTH = 960
DETECT_EVERY_N_FRAMES = 3
DEFAULT_CONFIDENCE = 0.70

# The app uses already-downloaded local models only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.set_page_config(
    page_title="AI CCTV | Command Center",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Modern UI styling ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root { --bg:#07111f; --panel:#0e1b2c; --line:#203650; --text:#eef5ff;
            --muted:#91a5bf; --blue:#2878ff; --green:#20d879; }
    html, body, [class*="css"] { font-family:'Inter',sans-serif; }
    .stApp { background:radial-gradient(circle at 80% 0%,#10233c 0%,#07111f 45%,#050c16 100%); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(5,12,22,.7); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1727,#07111d); border-right:1px solid #1d3048; }
    [data-testid="stSidebar"] * { color:#dce8f8; }
    .block-container { padding-top:1.35rem; padding-bottom:2rem; max-width:1600px; }
    h1,h2,h3 { letter-spacing:-.035em; color:#f2f7ff; }
    h1 { font-size:2rem !important; font-weight:800 !important; }
    .eyebrow { color:#7f9bbb; font-size:.78rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
    .muted { color:#91a5bf; }
    .panel { background:linear-gradient(145deg,rgba(18,36,58,.96),rgba(11,25,42,.96)); border:1px solid #203650; border-radius:17px; padding:17px 19px; }
    .status-pill { display:inline-flex; gap:8px; align-items:center; border:1px solid #176344; background:rgba(32,216,121,.09); color:#58efa0; padding:7px 11px; border-radius:100px; font-size:.8rem; font-weight:700; }
    .dot { height:8px; width:8px; background:#20d879; border-radius:50%; display:inline-block; box-shadow:0 0 12px rgba(32,216,121,.8); }
    .section-title { font-size:1.02rem; font-weight:750; color:#edf5ff; margin-bottom:3px; }
    .section-sub { color:#8da3bd; font-size:.79rem; margin-bottom:13px; }
    .small-note { color:#7f94ae; font-size:.76rem; }
    div.stButton > button { border-radius:10px; border:1px solid #2a4564; background:#142942; color:#eff6ff; font-weight:650; min-height:2.6rem; }
    div.stButton > button:hover { border-color:#3987ff; color:white; background:#193858; }
    div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#2878ff,#1758d4); border:0; color:white; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#12243a,#0d1b2d); border:1px solid #203650; border-radius:14px; padding:15px 17px; }
    div[data-testid="stMetricLabel"] { color:#91a5bf; }
    div[data-testid="stMetricValue"] { color:#f2f7ff; }
    hr { border-color:#203650; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Shared camera worker ----------------------------
class CameraManager:
    """One background camera loop shared by Streamlit reruns."""

    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.running = False
        self.recording_requested = False
        self.recording_active = False
        self.frame_rgb = None
        self.people_count = 0
        self.total_detection_frames = 0
        self.last_confidence = 0.0
        self.last_message = "Camera stopped"
        self.last_snapshot_time = 0.0
        self.recording_path = None
        self.current_video_path = None
        self.recording_started = None
        self.events = []
        self.fps = 0.0

    def start(self, detector, threshold, auto_snapshots=True):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.recording_requested = False
            self.last_message = "Starting camera…"
            self.thread = threading.Thread(
                target=self._camera_loop,
                args=(detector, threshold, auto_snapshots),
                daemon=True,
            )
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
            self.recording_requested = False
        thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=3)
        with self.lock:
            self.last_message = "Camera stopped"
            self.recording_active = False

    def start_recording(self):
        with self.lock:
            if self.running:
                self.recording_requested = True
                self.last_message = "Starting video recording…"

    def stop_recording(self):
        with self.lock:
            self.recording_requested = False

    def snapshot(self):
        with self.lock:
            frame = self.frame_rgb.copy() if self.frame_rgb is not None else None
        if frame is None:
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ALERT_DIR / f"manual_{stamp}.jpg"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(str(path), bgr):
            with self.lock:
                self.events.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "count": self.people_count,
                    "confidence": self.last_confidence,
                    "file": path.name,
                    "type": "Manual snapshot",
                })
                self.events = self.events[:100]
            return path
        return None

    def _camera_loop(self, detector, threshold, auto_snapshots):
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            with self.lock:
                self.running = False
                self.last_message = "ERROR: Could not open webcam"
            camera.release()
            return

        writer = None
        writer_path = None
        frame_count = 0
        last_frame_time = time.time()
        latest_people = []

        try:
            while True:
                with self.lock:
                    should_run = self.running
                    want_recording = self.recording_requested
                if not should_run:
                    break

                ok, frame = camera.read()
                if not ok:
                    with self.lock:
                        self.last_message = "Camera frame unavailable"
                    time.sleep(0.1)
                    continue

                h, w = frame.shape[:2]
                new_h = max(1, int(h * FRAME_WIDTH / w))
                frame = cv2.resize(frame, (FRAME_WIDTH, new_h))

                if frame_count % DETECT_EVERY_N_FRAMES == 0:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = detector(Image.fromarray(rgb), threshold=threshold)
                        latest_people = []
                        for item in results:
                            if str(item.get("label", "")).lower() != "person":
                                continue
                            box = item["box"]
                            latest_people.append((
                                max(0, int(box["xmin"])),
                                max(0, int(box["ymin"])),
                                min(frame.shape[1], int(box["xmax"])),
                                min(frame.shape[0], int(box["ymax"])),
                                float(item["score"]),
                            ))
                    except Exception as exc:
                        with self.lock:
                            self.last_message = f"Detection error: {exc}"

                annotated = frame.copy()
                for x1, y1, x2, y2, score in latest_people:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 216, 121), 2)
                    label = f"PERSON  {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    top = max(0, y1 - th - 12)
                    cv2.rectangle(annotated, (x1, top), (x1 + tw + 12, y1), (32, 216, 121), -1)
                    cv2.putText(annotated, label, (x1 + 6, max(15, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (4, 20, 18), 2, cv2.LINE_AA)

                now = time.time()
                fps = 1 / max(now - last_frame_time, 0.0001)
                last_frame_time = now
                cv2.putText(annotated, f"PEOPLE: {len(latest_people)}  |  FPS: {fps:.1f}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 248, 255), 2, cv2.LINE_AA)

                # Start the video writer on demand; every saved video includes the detection overlay.
                if want_recording and writer is None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    writer_path = VIDEO_DIR / f"cctv_recording_{stamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(writer_path), fourcc, 20.0,
                                             (annotated.shape[1], annotated.shape[0]))
                    if writer.isOpened():
                        with self.lock:
                            self.recording_active = True
                            self.current_video_path = str(writer_path)
                            self.recording_started = time.time()
                            self.last_message = "Recording video"
                    else:
                        writer.release()
                        writer = None
                        with self.lock:
                            self.recording_requested = False
                            self.recording_active = False
                            self.last_message = "Could not initialize MP4 writer; try another codec"

                if writer is not None:
                    if want_recording:
                        writer.write(annotated)
                    else:
                        writer.release()
                        writer = None
                        with self.lock:
                            self.recording_active = False
                            self.recording_path = str(writer_path) if writer_path else None
                            self.last_message = "Recording saved" if writer_path else "Camera running"
                            self.events.insert(0, {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "count": len(latest_people),
                                "confidence": max((p[4] for p in latest_people), default=0.0),
                                "file": Path(writer_path).name if writer_path else "",
                                "type": "Video recording saved",
                            })
                            self.events = self.events[:100]
                        writer_path = None

                if auto_snapshots and latest_people and now - self.last_snapshot_time >= 30:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = ALERT_DIR / f"person_{stamp}.jpg"
                    cv2.imwrite(str(snap_path), annotated)
                    self.last_snapshot_time = now
                    with self.lock:
                        self.events.insert(0, {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "count": len(latest_people),
                            "confidence": max((p[4] for p in latest_people), default=0.0),
                            "file": snap_path.name,
                            "type": "Person detected",
                        })
                        self.events = self.events[:100]

                with self.lock:
                    self.frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    self.people_count = len(latest_people)
                    self.total_detection_frames += 1
                    self.last_confidence = max((p[4] for p in latest_people), default=0.0)
                    self.fps = fps
                    if not self.recording_active:
                        self.last_message = "Camera running"

                frame_count += 1
        finally:
            camera.release()
            if writer is not None:
                writer.release()
                with self.lock:
                    self.recording_active = False
                    self.recording_path = str(writer_path) if writer_path else None
                    self.last_message = "Recording saved" if writer_path else "Camera stopped"
                    if writer_path:
                        self.events.insert(0, {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "count": self.people_count,
                            "confidence": self.last_confidence,
                            "file": Path(writer_path).name,
                            "type": "Video recording saved",
                        })
                        self.events = self.events[:100]
            with self.lock:
                self.running = False
                self.recording_active = False
                if self.last_message not in ("ERROR: Could not open webcam",):
                    self.last_message = "Camera stopped"


if "camera_manager" not in st.session_state:
    # The object survives Streamlit reruns in this browser session.
    st.session_state.camera_manager = CameraManager()
manager = st.session_state.camera_manager

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;padding:5px 0 22px;">
          <div style="background:linear-gradient(135deg,#2878ff,#154ec4);border-radius:13px;width:46px;height:46px;display:flex;align-items:center;justify-content:center;font-size:24px;">◉</div>
          <div><div style="font-size:1.25rem;font-weight:800;color:#f4f8ff;">AI CCTV</div>
          <div style="font-size:.73rem;color:#8199b5;">LOCAL SECURITY SUITE</div></div>
        </div>
        """, unsafe_allow_html=True,
    )
    page = st.radio("NAVIGATION", ["Live Overview", "People & Events", "Recordings", "Alert Gallery", "System Settings", "About"])
    st.markdown("---")
    st.markdown(
        """
        <div style="border:1px solid #176344;background:#0c2b26;border-radius:13px;padding:14px;">
          <div style="color:#58efa0;font-weight:750;font-size:.9rem;">● Offline Mode</div>
          <div style="color:#91b9ad;font-size:.76rem;margin-top:5px;">Local model · No cloud inference</div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.caption("YOLOS-Tiny · CPU inference")
    st.caption(f"Videos: {VIDEO_DIR.name}/")
    st.caption(f"Snapshots: {ALERT_DIR.name}/")

# ---------------------------- Header ----------------------------
with st.container():
    hleft, hright = st.columns([2.5, 1.5])
    with hleft:
        st.markdown("<div class='eyebrow'>SECURITY COMMAND CENTER</div>", unsafe_allow_html=True)
        st.title("Live Overview" if page == "Live Overview" else page)
        st.markdown("<div class='muted'>Live person count, camera recording, and local event storage.</div>", unsafe_allow_html=True)
    with hright:
        with manager.lock:
            running = manager.running
            recording = manager.recording_active
        pill = "● RECORDING" if recording else ("● CAMERA LIVE" if running else "● SYSTEM READY")
        color = "#ff647c" if recording else "#58efa0"
        st.markdown(
            f"<div style='text-align:right;padding-top:15px;'><span style='display:inline-block;border:1px solid {color};color:{color};padding:7px 11px;border-radius:100px;font-size:.8rem;font-weight:700;'>{pill}</span>"
            f"<div class='small-note' style='margin-top:9px;'>{datetime.now().strftime('%a, %d %b %Y · %I:%M:%S %p')}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------------------------- Helpers ----------------------------
def load_detector():
    if "detector" in st.session_state and st.session_state.detector is not None:
        return st.session_state.detector
    if not MODEL_DIR.is_dir():
        st.error(f"Model folder not found: `{MODEL_DIR}`. Run `python download_models.py` first.")
        return None
    with st.spinner("Loading local YOLOS-Tiny model…"):
        try:
            detector = pipeline("object-detection", model=str(MODEL_DIR), device=-1)
            st.session_state.detector = detector
            return detector
        except Exception as exc:
            st.error(f"Could not load the local model: {exc}")
            return None


def get_files(folder, suffix):
    return sorted([p for p in folder.glob(f"*{suffix}") if p.is_file()],
                  key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------- Live page ----------------------------
if page == "Live Overview":
    threshold = st.slider("Detection confidence threshold", 0.30, 0.95, DEFAULT_CONFIDENCE, 0.05)
    auto_snapshots = st.checkbox("Automatically save a snapshot when a person is detected (30-second cooldown)", value=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    if c1.button("▶  Start camera", type="primary", use_container_width=True):
        detector = load_detector()
        if detector is not None:
            manager.start(detector, threshold, auto_snapshots)
    if c2.button("■  Stop camera", use_container_width=True):
        manager.stop()
        st.rerun()
    if c3.button("🔴  Start recording", use_container_width=True):
        with manager.lock:
            camera_running = manager.running
        if camera_running:
            manager.start_recording()
            st.toast("Recording will start on the next camera frame.")
        else:
            st.warning("Start the camera before recording.")
    if c4.button("■  Stop recording", use_container_width=True):
        manager.stop_recording()
        st.toast("Recording will be finalized and saved.")

    with manager.lock:
        running = manager.running
        recording = manager.recording_active
        count = manager.people_count
        fps = manager.fps
        confidence = manager.last_confidence
        message = manager.last_message
        frame = manager.frame_rgb.copy() if manager.frame_rgb is not None else None
        video_path = manager.current_video_path
        started = manager.recording_started
        event_count = len(manager.events)

    video_files = get_files(VIDEO_DIR, ".mp4")
    snapshot_files = get_files(ALERT_DIR, ".jpg")
    elapsed = int(time.time() - started) if recording and started else 0
    elapsed_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("People in live frame", count if running else "—")
    m2.metric("Camera status", "LIVE" if running else "OFF")
    m3.metric("Recording status", "REC" if recording else "IDLE")
    m4.metric("Recording duration", elapsed_str if recording else "00:00:00")
    m5.metric("Saved videos", len(video_files))

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>◉  Live camera feed</div><div class='section-sub'>Camera 01 · Person detection overlay</div>", unsafe_allow_html=True)
        live_placeholder = st.empty()
        if frame is not None:
            live_placeholder.image(frame, use_container_width=True)
        else:
            live_placeholder.markdown(
                """<div style="height:390px;border:1px dashed #2a4564;border-radius:12px;background:radial-gradient(circle at 50% 45%,#152d49,#081422 70%);display:flex;align-items:center;justify-content:center;text-align:center;"><div><div style="font-size:3rem;margin-bottom:12px;">📹</div><div style="font-size:1.1rem;font-weight:750;color:#dce9fb;">Camera feed is paused</div><div style="color:#8198b4;font-size:.82rem;margin-top:7px;">Click Start camera to begin continuous capture.</div></div></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='small-note' style='margin-top:9px;'>Status: {message} · FPS: {fps:.1f} · Latest confidence: {confidence:.0%}</div>", unsafe_allow_html=True)
        if recording and video_path:
            st.markdown(f"<div style='color:#ff7187;font-size:.8rem;margin-top:5px;'>● Recording to: {Path(video_path).name}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚡ Quick actions</div><div class='section-sub'>Capture evidence and control recording</div>", unsafe_allow_html=True)
        if st.button("📸  Take snapshot now", use_container_width=True):
            saved = manager.snapshot()
            if saved:
                st.success(f"Snapshot saved: {saved.name}")
                st.rerun()
            else:
                st.warning("Start the camera first to take a snapshot.")
        st.markdown("---")
        st.markdown(f"<div class='small-note'>Snapshot folder</div><div style='font-weight:650;margin-bottom:10px;'>{ALERT_DIR.name}/</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-note'>Video folder</div><div style='font-weight:650;margin-bottom:10px;'>{VIDEO_DIR.name}/</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-note'>Recording format: MP4 · Codec: mp4v</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    bottom1, bottom2 = st.columns([1.4, 1], gap="large")
    with bottom1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔔 Recent events</div><div class='section-sub'>Latest detection and recording activity</div>", unsafe_allow_html=True)
        with manager.lock:
            recent = list(manager.events[:6])
        if recent:
            for event in recent:
                e1, e2, e3 = st.columns([1, 1.7, 1.2])
                e1.markdown(f"**{event['time']}**")
                e2.markdown(f"{event['type']} · {event['count']} person(s)")
                e3.markdown(f"<span style='color:#58efa0'>{event['confidence']:.0%}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='small-note'>No events yet. Start the camera to see activity.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with bottom2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎞️ Latest recordings</div><div class='section-sub'>Saved MP4 files on this device</div>", unsafe_allow_html=True)
        latest_videos = video_files[:3]
        if latest_videos:
            for path in latest_videos:
                st.markdown(f"**{path.name}**")
                st.caption(f"{path.stat().st_size / (1024 * 1024):.2f} MB")
        else:
            st.markdown("<div class='small-note'>No videos saved yet. Start recording to create an MP4.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Refresh only the visual section periodically while the camera thread runs.
    # The camera itself continues recording in its background thread between refreshes.
    if running:
        time.sleep(0.5)
        st.rerun()

elif page == "People & Events":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>People & Events</div><div class='section-sub'>Events detected during this app session</div>", unsafe_allow_html=True)
    with manager.lock:
        events = list(manager.events)
    if events:
        st.dataframe(
            [{"Time": e["time"], "Event": e["type"], "People in frame": e["count"],
              "Confidence": f"{e['confidence']:.1%}", "File": e["file"]} for e in events],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No events recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("YOLOS-Tiny detects the object category 'person'; it does not identify a person's name or identity.")

elif page == "Recordings":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎞️ Video recordings</div><div class='section-sub'>Saved locally in the recordings folder</div>", unsafe_allow_html=True)
    videos = get_files(VIDEO_DIR, ".mp4")
    if not videos:
        st.info("No saved videos yet. Go to Live Overview, start the camera, then click Start recording.")
    else:
        for path in videos:
            col1, col2, col3 = st.columns([2.3, 1, 1])
            col1.markdown(f"**{path.name}**")
            col1.caption(f"Modified: {datetime.fromtimestamp(path.stat().st_mtime).strftime('%d %b %Y, %I:%M:%S %p')}")
            col2.markdown(f"{path.stat().st_size / (1024 * 1024):.2f} MB")
            with open(path, "rb") as f:
                col3.download_button("Download MP4", f.read(), file_name=path.name, mime="video/mp4", key=f"video_{path.name}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.code(str(VIDEO_DIR), language="text")

elif page == "Alert Gallery":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🖼️ Alert gallery</div><div class='section-sub'>Person detection and manual snapshots</div>", unsafe_allow_html=True)
    images = get_files(ALERT_DIR, ".jpg")
    if not images:
        st.info("No snapshots saved yet.")
    else:
        cols = st.columns(3)
        for i, path in enumerate(images):
            with cols[i % 3]:
                st.image(str(path), use_container_width=True)
                st.caption(path.name)
                with open(path, "rb") as f:
                    st.download_button("Download image", f.read(), file_name=path.name, mime="image/jpeg", key=f"img_{path.name}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.code(str(ALERT_DIR), language="text")

elif page == "System Settings":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### System settings")
    st.write("**Local model directory**")
    st.code(str(MODEL_DIR))
    st.write("**Video recording directory**")
    st.code(str(VIDEO_DIR))
    st.write("**Snapshot directory**")
    st.code(str(ALERT_DIR))
    st.write("**Model availability**")
    if MODEL_DIR.is_dir():
        st.success("Local YOLOS-Tiny model folder found.")
    else:
        st.error("Model folder missing. Run `python download_models.py` first.")
    st.info("Inference runs locally using the downloaded model. Videos and snapshots are saved on this computer.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### About AI CCTV")
    st.write("Local-first person detection using Streamlit, OpenCV, and YOLOS-Tiny.")
    st.markdown("- Continuous background webcam capture\n- Live person count and bounding boxes\n- MP4 video recording with detection overlays\n- Automatic and manual JPEG snapshots\n- Local recordings and alert gallery")
    st.warning("This is a demo system, not a certified security product. Detection performance depends on lighting, camera angle, and hardware. Use cameras only where you have permission to record.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div style='border-top:1px solid #1d3048;margin-top:28px;padding-top:13px;display:flex;justify-content:space-between;color:#6f86a3;font-size:.74rem;'><span>AI CCTV · LOCAL DEMO BUILD</span><span>Inference and recording stay on this device</span></div>",
    unsafe_allow_html=True,
)
