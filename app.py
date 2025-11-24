import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autonomous Vehicle Vision",
    page_icon="🚘",
    layout="wide"
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    """
    Loads the YOLO model from the 'best.pt' file.
    """
    model_path = "best.pt"  # Assumes best.pt is in the same folder
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found! Please upload '{model_path}' to your project folder.")
        return None
    
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- 3. SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Detection Settings")

# Confidence Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.35, 
    step=0.05,
    help="Higher values reduce false positives but might miss some objects."
)

st.sidebar.markdown("---")
st.sidebar.info("Select **Image** or **Video** mode to begin analysis.")

# --- 4. MAIN INTERFACE ---
st.title("🚘 Autonomous Vehicle Object Detection")
st.markdown("Analyze road scenes for cars, pedestrians, signs, and more.")

# --- 5. SELECT MODE ---
mode = st.radio("Select Input Type:", ["🖼️ Image", "🎥 Video"], horizontal=True)

# ==========================================
#               IMAGE MODE
# ==========================================
if mode == "🖼️ Image":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        # Load and display original
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Input", use_column_width=True)
            
        # Inference Button
        if st.button("🔍 Detect Objects", type="primary"):
            with col2:
                st.subheader("Detection Result")
                with st.spinner("Analyzing..."):
                    # Predict
                    results = model.predict(image, conf=conf_threshold)
                    
                    # Plot
                    res_plotted = results[0].plot()
                    res_image = res_plotted[:, :, ::-1] # BGR to RGB
                    
                    st.image(res_image, caption="Predictions", use_column_width=True)
                    
                    # Stats
                    st.success(f"✅ Found {len(results[0].boxes)} objects.")

# ==========================================
#               VIDEO MODE
# ==========================================
elif mode == "🎥 Video":
    uploaded_video = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video and model:
        # 1. Save input video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            # Video Stats
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Video Loaded: {width}x{height} @ {fps}fps | {total_frames} frames")
            
            if st.button("▶️ Start Processing", type="primary"):
                # 2. Prepare Output Video
                output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                output_path = output_tfile.name
                
                # --- FIX 1: Try 'avc1' (H.264) for browser support ---
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Fallback for Windows local run
                    
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # 3. Processing Loop
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Processing video frames... please wait.")
                
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run YOLO prediction
                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    
                    # Plot detections on the frame
                    res_plotted = results[0].plot()
                    
                    # Write frame to output video
                    out.write(res_plotted)
                    
                    # Update Progress
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                cap.release()
                out.release()
                
                # 4. Display Result
                status_text.success("✅ Analysis Complete!")
                st.subheader("Processed Video")
                
                # --- FIX 2: Read as binary for Streamlit Player ---
                with open(output_path, 'rb') as v:
                    video_bytes = v.read()
                
                st.video(video_bytes)
                
                # --- FIX 3: Add Download Button ---
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="autonomous_vehicle_result.mp4",
                    mime="video/mp4"
                )
                
                # Cleanup input temp file
                os.remove(video_path)
                # We keep output_path briefly so the download button works
