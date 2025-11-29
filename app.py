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

# --- 3. CUSTOM CSS (DARK THEME + NEON ACCENTS) ---
st.markdown("""
    <style>
        @import url('https://fonts.cdnfonts.com/css/stella-aesta');

        /* Main Background */
        .stApp {
            background-color: #0b0c10;
            color: #c5c6c7;
        }
        
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #1f2833;
            border-right: 1px solid #45a29e;
        }
        
        /* Sidebar Text */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
            color: #66fcf1 !important;
            font-family: 'Helvetica Neue', sans-serif;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.05em;
        }
        
        /* Main Title Font */
        h1 {
            font-family: 'Stella Aesta', sans-serif !important;
            color: #66fcf1;
            font-weight: normal;
            font-size: 3.5rem !important;
            text-shadow: 0px 0px 10px rgba(102, 252, 241, 0.3);
        }
        
        /* Buttons */
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

        /* Metrics */
        [data-testid="stMetricLabel"] { color: #45a29e; }
        [data-testid="stMetricValue"] { color: #ffffff; font-family: 'Courier New', monospace; }

        /* --- SLIDER STYLING (FIXED) --- */
        div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
            background-color: #66fcf1 !important;
            box-shadow: 0 0 10px rgba(102, 252, 241, 0.5);
            border: none;
        }
        div.stSlider > div[data-baseweb="slider"] > div > div {
            background: #45a29e !important;
        }
        div[data-testid="stMarkdownContainer"] p {
            color: #66fcf1 !important;
        }
        
        /* Radio Buttons */
        [data-testid="stRadio"] label {
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
    
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("System Config")
    
    # Input Type
    st.sidebar.subheader("Input Source")
    input_type = st.sidebar.radio("Select Type", ["Image", "Video"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    
    # Model Parameters
    st.sidebar.subheader("Sensitivity")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    st.sidebar.markdown("---")
    
    # Project Summary Button
    if st.sidebar.button("Project Summary"):
        st.session_state.view_summary = not st.session_state.view_summary

    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: Online | v1.0")

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
        
        st.markdown("#### Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("mAP@50", "50.0%")
        col2.metric("Precision", "69.9%")
        col3.metric("Recall", "45.1%")
        
        if st.button("Back to Detection"):
            st.session_state.view_summary = False
            st.rerun()

    # 2. VIEW: DETECTION INTERFACE
    else:
        st.title("Autonomous Vehicle Object Detection")
        
        # --- IMAGE MODE ---
        if input_type == "Image":
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file and model:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Input Data", use_column_width=True)
                
                with col2:
                    if st.button("Run Inference", type="primary"):
                        with st.spinner("Processing..."):
                            results = model.predict(image, conf=conf_threshold)
                            res_plotted = results[0].plot()
                            res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                            
                            st.image(res_image, caption="Detection Result", use_column_width=True)
                            
                            count = len(results[0].boxes)
                            st.info(f"Objects Detected: {count}")

        # --- VIDEO MODE ---
        elif input_type == "Video":
            uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_video and model:
                # Variable Initialization
                video_path = None
                raw_path = None
                final_path = None
                cap = None

                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name
                    
                    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening video.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        st.caption(f"Video: {width}x{height} | {fps} FPS | {total_frames} Frames")
                        
                        if st.button("Start Analysis", type="primary"):
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            stop_button = st.button("Stop Analysis")
                            
                            frame_count = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                if stop_button:
                                    status_text.warning("Analysis stopped.")
                                    break
                                
                                results = model.predict(frame, conf=conf_threshold, verbose=False)
                                res_plotted = results[0].plot()
                                out.write(res_plotted)
                                
                                frame_count += 1
                                if total_frames > 0:
                                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

                            cap.release()
                            out.release()
                            
                            status_text.text("Optimizing video for web...")
                            if convert_video_to_h264(raw_path, final_path):
                                status_text.success("Complete.")
                                
                                with open(final_path, 'rb') as v:
                                    video_bytes = v.read()
                                
                                st.video(video_bytes)
                                st.download_button("Download Result", video_bytes, "result.mp4", "video/mp4")
                            else:
                                st.error("Video optimization failed.")

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    # Cleanup
                    try: 
                        if cap is not None: cap.release()
                    except: pass
                    
                    for path in [video_path, raw_path, final_path]:
                        if path and os.path.exists(path) and path != final_path:
                            try: os.remove(path)
                            except: pass

if __name__ == "__main__":
    main()
