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
    page_title="Object Detection System",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM STYLING (THEME) ---
# Injecting CSS to enforce the specific colors requested
st.markdown("""
    <style>
        /* Main Background Color */
        .stApp {
            background-color: #55738D;
        }
        
        /* Sidebar Background Color */
        [data-testid="stSidebar"] {
            background-color: #96A7B6;
        }
        
        /* General Text Color (Main Area) */
        .stApp, p, label, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, li, span {
            color: #CBC8C4 !important;
        }
        
        /* Sidebar Text Color */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #F9FFCC !important;
        }

        /* Adjusting Headings specifically to ensure they take the color */
        h1, h2, h3 {
            color: #F9FFCC !important;
        }
        
        /* Optional: Making the file uploader look good on dark background */
        [data-testid="stFileUploader"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 4. MODEL LOADING ---
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

# --- 5. VIDEO CONVERSION HELPER ---
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
    
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("System Configuration")
    mode = st.sidebar.selectbox("Select Data Source", ["Image Analysis", "Video Analysis"])
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Model Parameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    
    st.sidebar.divider()
    st.sidebar.caption("Autonomous Vehicle Perception Module v1.0")

    # --- MAIN INTERFACE ---
    st.title("Autonomous Vehicle Object Detection")
    st.markdown("### Real-time Perception System")
    st.markdown("---")

    # --- IMAGE MODE ---
    if mode == "Image Analysis":
        uploaded_file = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file and model:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Data")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                if st.button("Run Inference", type="primary"):
                    with st.spinner("Processing image data..."):
                        results = model.predict(image, conf=conf_threshold)
                        res_plotted = results[0].plot()
                        res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        st.image(res_image, use_column_width=True)
                        obj_count = len(results[0].boxes)
                        st.info(f"Objects Detected: {obj_count}")

    # --- VIDEO MODE ---
    elif mode == "Video Analysis":
        uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video and model:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            raw_output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            raw_output_path = raw_output_tfile.name

            final_output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            final_output_path = final_output_tfile.name

            try:
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Error: Could not open video source.")
                else:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.caption(f"Source Metadata: {width}x{height} | {fps} FPS | {total_frames} Frames")
                    
                    if st.button("Start Analysis", type="primary"):
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        stop_button = st.button("Stop Analysis")
                        
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if stop_button:
                                status_text.warning("Analysis interrupted by user.")
                                break
                            
                            results = model.predict(frame, conf=conf_threshold, verbose=False)
                            res_plotted = results[0].plot()
                            out.write(res_plotted)
                            
                            frame_count += 1
                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                                status_text.text(f"Processing frame {frame_count} of {total_frames}...")

                        cap.release()
                        out.release()
                        
                        status_text.text("Optimizing video codec for web playback...")
                        success = convert_video_to_h264(raw_output_path, final_output_path)
                        
                        if success:
                            status_text.success("Processing Completed Successfully.")
                            st.subheader("Output Stream")
                            
                            with open(final_output_path, 'rb') as v:
                                video_bytes = v.read()
                            
                            st.video(video_bytes)
                            
                            st.download_button(
                                label="Download Output Video",
                                data=video_bytes,
                                file_name="detection_output.mp4",
                                mime="video/mp4"
                            )
                        else:
                            st.error("Video encoding failed.")

            except Exception as e:
                st.error(f"Runtime Error: {e}")
            finally:
                for path in [video_path, raw_output_path, final_output_path]:
                    if os.path.exists(path) and path != final_output_path:
                        try:
                            os.remove(path)
                        except:
                            pass

if __name__ == "__main__":
    main()


