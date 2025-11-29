import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(
    page_title="AV Perception System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Command Center" Look
st.markdown("""
    <style>
        /* Dark Background */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #262730;
        }
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-family: "Source Code Pro", monospace;
            color: #00FF41; /* Hacker Green */
        }
        /* Buttons */
        .stButton button {
            border: 1px solid #00FF41;
            color: #00FF41;
            background-color: transparent;
            font-family: "Source Code Pro", monospace;
        }
        .stButton button:hover {
            background-color: #00FF41;
            color: #000000;
        }
        /* Headers */
        h1, h2, h3 {
            font-family: "Source Code Pro", monospace;
            color: #E0E0E0;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. IMPORT MOVIEPY ---
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    st.error("System Error: MoviePy library not found. Please update requirements.txt.")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        st.error(f"Critical Error: Model file '{model_path}' not found.")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

def convert_video_to_h264(input_path, output_path):
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None, preset="ultrafast")
        return True
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        return False

# --- 4. MAIN APP LOGIC ---
def main():
    model = load_model()
    
    # --- SIDEBAR: SYSTEM STATUS & METRICS ---
    st.sidebar.header("📡 SYSTEM STATUS")
    
    # Hardcoded Metrics from your Results.csv (Final Epoch)
    # Precision: 0.699, Recall: 0.45, mAP50: 0.50
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Precision", "69.9%", "1.2%")
    col2.metric("Recall", "45.1%", "0.5%")
    st.sidebar.metric("Mean Average Precision (mAP)", "50.0%")
    
    st.sidebar.markdown("---")
    
    # Input Mode
    st.sidebar.header("🎮 CONTROL PANEL")
    mode = st.sidebar.selectbox("Operation Mode", ["Image Inference", "Video Feed Analysis", "System Documentation"])
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.header("⚙️ PARAMETERS")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    # --- MAIN INTERFACE ---
    st.title("👁️ AV PERCEPTION MODULE")
    st.caption("YOLOv11n | BDD100K Dataset | Real-Time Inference")
    st.markdown("---")

    # --- MODE 1: IMAGE ---
    if mode == "Image Inference":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📥 Input Source")
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("📤 Telemetry & Output")
            if uploaded_file and model:
                if st.button("INITIATE SCAN", type="primary"):
                    with st.spinner("Processing neural network..."):
                        results = model.predict(image, conf=conf_threshold)
                        res_plotted = results[0].plot()
                        res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        st.image(res_image, use_column_width=True)
                        
                        # Detection Statistics
                        count = len(results[0].boxes)
                        st.success(f"Scan Complete: {count} Objects Identified")
                        
                        # Detailed Breakdown
                        if count > 0:
                            classes = results[0].boxes.cls.cpu().numpy()
                            names = results[0].names
                            class_counts = {}
                            for c in classes:
                                name = names[int(c)]
                                class_counts[name] = class_counts.get(name, 0) + 1
                            
                            st.write("### Object Classification:")
                            st.json(class_counts)

    # --- MODE 2: VIDEO ---
    elif mode == "Video Feed Analysis":
        uploaded_video = st.file_uploader("Upload Dashcam Footage", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video and model:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

            try:
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"Stream Loaded: {width}x{height} @ {fps}FPS")
                
                if st.button("🚀 EXECUTE PIPELINE", type="primary"):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        results = model.predict(frame, conf=conf_threshold, verbose=False)
                        res_plotted = results[0].plot()
                        out.write(res_plotted)
                        
                        frame_count += 1
                        if total_frames > 0:
                            progress_bar.progress(min(frame_count / total_frames, 1.0))
                            status_text.text(f"Analyzing Frame: {frame_count}/{total_frames}")

                    cap.release()
                    out.release()
                    
                    status_text.text("Compressing stream for web transmission...")
                    if convert_video_to_h264(raw_path, final_path):
                        status_text.success("Pipeline Executed Successfully.")
                        st.video(final_path)
                    else:
                        st.error("Compression Failed.")

            except Exception as e:
                st.error(f"Runtime Error: {e}")
            finally:
                # Clean up raw files
                if os.path.exists(video_path): os.remove(video_path)
                if os.path.exists(raw_path): os.remove(raw_path)

    # --- MODE 3: DOCUMENTATION ---
    elif mode == "System Documentation":
        st.markdown("## 🛠️ Project Specifications")
        
        st.markdown("""
        ### **Autonomous Vehicle Perception System**
        This system utilizes a **YOLOv11n** neural network trained on the **BDD100K** dataset to detect critical road objects in real-time.
        
        #### **Technical Stack:**
        - **Model Architecture:** YOLOv11 (Nano)
        - **Training Dataset:** Berkeley DeepDrive (BDD100K)
        - **Training Epochs:** 50
        - **Deployment:** Streamlit Cloud + OpenCV
        
        #### **Performance Metrics:**
        - **mAP@50:** 50.0%
        - **Precision:** 69.9%
        - **Recall:** 45.1%
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Engineering Team")
        
        team = [
            "Omar Salem",
            "Mohammed Sallal",
            "Ziad Medhat",
            "Refaat Elia",
            "Shahd Farid"
        ]
        
        cols = st.columns(len(team))
        for i, member in enumerate(team):
            with cols[i]:
                st.markdown(f"**{member}**")
                st.caption("AI Engineer")

if __name__ == "__main__":
    main()
