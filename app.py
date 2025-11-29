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

# --- 3. CUSTOM CSS (PROFESSIONAL THEME) ---
st.markdown("""
    <style>
        /* Professional Light Theme adjustments */
        .stApp {
            background-color: #FFFFFF;
            color: #333333;
        }
        
        /* Sidebar Styling - Light Grey */
        [data-testid="stSidebar"] {
            background-color: #F0F2F6;
            border-right: 1px solid #D1D5DB;
        }
        
        /* Typography */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #0F172A;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #2563EB;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton button:hover {
            background-color: #1D4ED8;
            color: white;
        }
        
        /* Metrics */
        [data-testid="stMetricLabel"] {
            color: #64748B;
        }
        [data-testid="stMetricValue"] {
            color: #0F172A;
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
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None, preset="ultrafast")
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
    st.sidebar.subheader("Input Source")
    input_type = st.sidebar.radio("Select Data Type", ["Image", "Video"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    
    # Model Parameters
    st.sidebar.subheader("Parameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    st.sidebar.markdown("---")
    
    # Project Summary Button (Toggle)
    if st.sidebar.button("Project Summary"):
        st.session_state.view_summary = not st.session_state.view_summary

    st.sidebar.markdown("---")
    st.sidebar.caption("System Version 1.0")

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
        
        if st.button("Return to Detection"):
            st.session_state.view_summary = False
            st.rerun()

    # 2. VIEW: DETECTION INTERFACE (Image/Video)
    else:
        st.title("Autonomous Vehicle Object Detection")
        
        # --- IMAGE LOGIC ---
        if input_type == "Image":
            st.subheader("Image Analysis")
            uploaded_file = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
            
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
                            
                            st.image(res_image, caption="Analysis Result", use_column_width=True)
                            
                            count = len(results[0].boxes)
                            st.info(f"Objects Detected: {count}")

        # --- VIDEO LOGIC ---
        elif input_type == "Video":
            st.subheader("Video Analysis")
            uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
            
            if uploaded_video and model:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                video_path = tfile.name
                
                raw_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                try:
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening video.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        st.caption(f"Metadata: {width}x{height} | {fps} FPS | {total_frames} Frames")
                        
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
                                status_text.success("Analysis Complete.")
                                
                                with open(final_output_path, 'rb') as v:
                                    video_bytes = v.read()
                                
                                st.video(video_bytes)
                                st.download_button("Download Result", video_bytes, "result.mp4", "video/mp4")
                            else:
                                st.error("Video conversion failed.")

                except Exception as e:
                    st.error(f"Runtime Error: {e}")
                finally:
                    for path in [video_path, raw_path, final_path]:
                        if os.path.exists(path) and path != final_path:
                            try: os.remove(path)
                            except: pass

if __name__ == "__main__":
    main()
