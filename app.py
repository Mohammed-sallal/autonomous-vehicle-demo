import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. IMPORT MOVIEPY FOR VIDEO CONVERSION ---
# This is crucial for making the video play on the web and reducing file size
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    st.error("MoviePy is not installed. Please add 'moviepy' to requirements.txt")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autonomous Vehicle Vision",
    page_icon="🚘",
    layout="wide"
)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model(model_path="best.pt"):
    """
    Loads the YOLO model securely. Caches the model to prevent reloading.
    """
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found! Please upload '{model_path}' to your project folder.")
        return None
    
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 4. VIDEO CONVERSION HELPER ---
def convert_video_to_h264(input_path, output_path):
    """
    Converts video to H.264 (Web Compatible) using MoviePy.
    This fixes the "Black Screen" issue and compresses the file size.
    """
    try:
        clip = VideoFileClip(input_path)
        # write_videofile uses libx264 by default which works in all browsers
        # 'preset="ultrafast"' speeds up compression
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None, preset="ultrafast")
        return True
    except Exception as e:
        st.error(f"Error converting video: {e}")
        return False

# --- 5. MAIN APP LOGIC ---
def main():
    # Load Model
    model = load_model()
    
    # Sidebar
    st.sidebar.title("⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.info("Select **Image** or **Video** mode.")

    # Main Title
    st.title("🚘 Autonomous Vehicle Object Detection")
    st.markdown("Analyze road scenes for cars, pedestrians, signs, and more.")

    # Mode Selector
    mode = st.radio("Select Input Type:", ["🖼️ Image", "🎥 Video"], horizontal=True)

    # ==========================================
    #               IMAGE MODE
    # ==========================================
    if mode == "🖼️ Image":
        uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file and model:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Detect Objects", type="primary"):
                with col2:
                    st.subheader("Detection Result")
                    with st.spinner("Analyzing..."):
                        results = model.predict(image, conf=conf_threshold)
                        
                        # Plot returns BGR numpy array
                        res_plotted = results[0].plot()
                        
                        # Convert BGR to RGB for Streamlit display
                        res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        st.image(res_image, use_column_width=True)
                        st.success(f"✅ Found {len(results[0].boxes)} objects.")

    # ==========================================
    #               VIDEO MODE
    # ==========================================
    elif mode == "🎥 Video":
        uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_video and model:
            # Create temp files
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            # Temporary "Raw" output (Fast to write, but big & incompatible)
            raw_output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            raw_output_path = raw_output_tfile.name

            # Final "Web" output (Compressed & Web-ready)
            final_output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            final_output_path = final_output_tfile.name

            try:
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Error opening video file.")
                else:
                    # Video Metadata
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.info(f"Loaded: {width}x{height} @ {fps} FPS | {total_frames} Frames")
                    
                    if st.button("▶️ Start Processing", type="primary"):
                        # Use 'mp4v' for speed during the frame-by-frame detection step
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        stop_button = st.button("⏹️ Stop Processing")
                        
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if stop_button:
                                status_text.warning("Processing stopped by user.")
                                break
                            
                            # YOLO Prediction
                            results = model.predict(frame, conf=conf_threshold, verbose=False)
                            res_plotted = results[0].plot()
                            
                            # Write BGR frame to RAW video
                            out.write(res_plotted)
                            
                            # UI Updates
                            frame_count += 1
                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                                status_text.text(f"Processing Frame {frame_count}/{total_frames}")

                        # 1. Close the OpenCV Writer and Reader
                        cap.release()
                        out.release()
                        
                        # 2. OPTIMIZATION STEP: Convert to Web-Ready Format
                        status_text.text("Optimize video for web (Compressing)... this may take a moment.")
                        success = convert_video_to_h264(raw_output_path, final_output_path)
                        
                        if success:
                            status_text.success("✅ Analysis & Compression Complete!")
                            st.subheader("Processed Video")
                            
                            # Read the optimized file
                            with open(final_output_path, 'rb') as v:
                                video_bytes = v.read()
                            
                            # Display it
                            st.video(video_bytes)
                            
                            # Download Button
                            st.download_button(
                                label="⬇️ Download Optimized Result",
                                data=video_bytes,
                                file_name="autonomous_result.mp4",
                                mime="video/mp4"
                            )
                        else:
                            st.error("Video conversion failed. Please check if 'moviepy' is in requirements.txt")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Cleanup ALL temp files to keep the server clean
                for path in [video_path, raw_output_path, final_output_path]:
                    if os.path.exists(path):
                        try:
                            # We keep final output briefly for download/viewing if needed, 
                            # but usually Streamlit handles the buffer in memory for the button.
                            # We delete raw files immediately.
                            if path != final_output_path: 
                                os.remove(path)
                        except:
                            pass

if __name__ == "__main__":
    main()
