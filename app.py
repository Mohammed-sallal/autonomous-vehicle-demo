import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. IMPORT MOVIEPY ---
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    st.error("System Error: MoviePy library not found. Please update requirements.txt.")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AV Perception System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS (DARK PROFESSIONAL THEME) ---
st.markdown("""
    <style>
        /* Import Custom Font */
        @import url('https://fonts.cdnfonts.com/css/stella-aesta');

        /* 1. MAIN BACKGROUND */
        .stApp {
            background-color: #0b0c10; /* Deep Carbon Black */
            color: #c5c6c7; /* Light Grey Text */
        }
        
        /* 2. SIDEBAR (CONTROL PANEL) */
        [data-testid="stSidebar"] {
            background-color: #1f2833; /* Dark Slate */
            border-right: 1px solid #45a29e; /* Thin Teal Border */
        }
        
        /* Sidebar Headers */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #66fcf1; /* Neon Teal */
            font-family: 'Helvetica Neue', sans-serif;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            font-weight: 700;
        }
        
        /* Sidebar Labels */
        [data-testid="stSidebar"] label {
            color: #c5c6c7 !important;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
        }

        /* 3. TYPOGRAPHY */
        /* Main Title with Custom Font */
        h1 {
            font-family: 'Stella Aesta', sans-serif !important;
            color: #66fcf1; 
            font-weight: normal;
            font-size: 3.5rem !important;
            text-shadow: 0px 0px 10px rgba(102, 252, 241, 0.3);
        }
        
        h2, h3, h4 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #ffffff;
            font-weight: 600;
        }
        
        /* 4. PROFESSIONAL BUTTONS */
        .stButton button {
            background-color: transparent;
            color: #66fcf1;
            border: 1px solid #45a29e;
            border-radius: 0px; 
            padding: 0.6rem 1.5rem;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 0.1em;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton button:hover {
            background-color: #45a29e;
            color: #0b0c10;
            box-shadow: 0 0 15px rgba(69, 162, 158, 0.7);
            border-color: #66fcf1;
        }

        /* 5. FILE UPLOADER STYLE */
        [data-testid="stFileUploader"] {
            background-color: #1f2833;
            border: 1px dashed #45a29e;
            border-radius: 4px;
            padding: 20px;
        }
        
        /* 6. METRICS */
        [data-testid="stMetricLabel"] {
            color: #45a29e;
            font-size: 0.8rem;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-family: 'Courier New', monospace;
        }

        /* 7. RADIO BUTTONS */
        [data-testid="stRadio"] > label {
            color: #c5c6c7 !important;
            font-weight: bold;
        }
        div[role="radiogroup"] > label > div:first-child {
            background-color: #66fcf1 !important;
            border-color: #45a29e !important;
        }

    </style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE INITIALIZATION ---
if 'view_summary' not in st.session_state:
    st.session_state.view_summary = False

# --- 5. MODEL LOADING ---
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
        if clip.h > 720:
            clip = clip.resize(height=720)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None, preset="ultrafast")
        clip.close()
        return True
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        return False

# --- 6. MAIN APP LOGIC ---
def main():
    model = load_model()
    
    # --- SIDEBAR: CONTROL PANEL ---
    st.sidebar.header("Control Panel")
    
    # Input Selection
    st.sidebar.subheader("System Input")
    input_type = st.sidebar.radio("Data Source", ["Image", "Video"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    
    # Model Parameters
    st.sidebar.subheader("Sensitivity")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    st.sidebar.markdown("---")
    
    # Project Summary Button (Toggle)
    if st.sidebar.button("System Info"):
        st.session_state.view_summary = not st.session_state.view_summary

    st.sidebar.markdown("---")
    
    # Status Indicator
    st.sidebar.caption("Status: System Online")
    st.sidebar.caption("v1.0 | YOLOv11n")

    # --- MAIN CONTENT AREA ---
    
    # 1. VIEW: PROJECT SUMMARY
    if st.session_state.view_summary:
        st.title("Project Summary")
        st.markdown("### Autonomous Vehicle Perception System")
        
        st.markdown("""
        This system utilizes a **YOLOv11n** neural network trained on the **BDD100K** dataset to detect critical road objects in real-time.
        
        #### Technical Specifications
        - **Model Architecture:** YOLOv11 (Nano)
        - **Dataset:** Berkeley DeepDrive (BDD100K)
        - **Training Metrics:** mAP@50: 50.0% | Precision: 69.9% | Recall: 45.1%
        
        #### Engineering Team
        - Omar Salem
        - Mohammed Sallal
        - Ziad Medhat
        - Refaat Elia
        - Shahd Farid
        """)
        
        # Display Metrics in columns
        st.markdown("#### Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Average Precision", "50.0%")
        col2.metric("Precision", "69.9%")
        col3.metric("Recall", "45.1%")
        
        if st.button("Return to Operation"):
            st.session_state.view_summary = False
            st.rerun()

    # 2. VIEW: DETECTION INTERFACE (Image/Video)
    else:
        st.title("Autonomous Vehicle Object Detection")
        
        # --- IMAGE LOGIC ---
        if input_type == "Image":
            st.subheader("Image Analysis Module")
            uploaded_file = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file and model:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption("INPUT FEED")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.caption("ANALYSIS OUTPUT")
                    if st.button("Execute Inference", type="primary"):
                        with st.spinner("Processing neural network..."):
                            results = model.predict(image, conf=conf_threshold)
                            res_plotted = results[0].plot()
                            res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                            
                            st.image(res_image, use_column_width=True)
                            
                            count = len(results[0].boxes)
                            st.info(f"Targets Identified: {count}")

        # --- VIDEO LOGIC ---
        elif input_type == "Video":
            st.subheader("Video Analysis Module")
            uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_video and model:
                # Initialize variables to avoid UnboundLocalError
                video_path = None
                raw_path = None
                final_path = None
                cap = None # Initialize cap to None

                try:
                    # Create temp files
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name
                    
                    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening video stream.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        st.caption(f"Stream Data: {width}x{height} | {fps} FPS | {total_frames} Frames")
                        
                        if st.button("Initiate Sequence", type="primary"):
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            stop_button = st.button("Abort Sequence")
                            
                            frame_count = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                if stop_button:
                                    status_text.warning("Sequence Aborted.")
                                    break
                                
                                results = model.predict(frame, conf=conf_threshold, verbose=False)
                                res_plotted = results[0].plot()
                                out.write(res_plotted)
                                
                                frame_count += 1
                                if total_frames > 0:
                                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                                    status_text.text(f"Scanning Frame {frame_count}/{total_frames}")

                            cap.release()
                            out.release()
                            
                            status_text.text("Encoding stream for playback...")
                            if convert_video_to_h264(raw_path, final_path):
                                status_text.success("Sequence Complete.")
                                
                                with open(final_path, 'rb') as v:
                                    video_bytes = v.read()
                                
                                st.video(video_bytes)
                                st.download_button("Export Data", video_bytes, "detection_log.mp4", "video/mp4")
                            else:
                                st.error("Encoding protocol failed.")

                except Exception as e:
                    st.error(f"Runtime Error: {e}")
                finally:
                    # Robust Cleanup
                    try: 
                        if cap is not None:
                            cap.release()
                    except: pass
                    
                    for path in [video_path, raw_path, final_path]:
                        # Only delete if path exists and is not None
                        if path and os.path.exists(path) and path != final_path:
                            try: os.remove(path)
                            except: pass

if __name__ == "__main__":
    main()
