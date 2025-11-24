import streamlit as st
import cv2
import tempfile
import os
import subprocess
from ultralytics import YOLO
from PIL import Image

# ==============================
# 1) PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Autonomous Vehicle Vision",
    page_icon="🚘",
    layout="wide"
)

# ==============================
# 2) UTILITIES
# ==============================
def ffmpeg_available() -> bool:
    """Check if ffmpeg exists on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def reencode_with_ffmpeg(input_path: str) -> str:
    """
    Re-encode the OpenCV output using FFmpeg to ensure H.264 + yuv420p + faststart for browser compatibility.
    Returns the path to the re-encoded file or raises if encoding fails.
    """
    base, ext = os.path.splitext(input_path)
    final_output_path = f"{base}_encoded.mp4"

    # Use libx264, ensure yuv420p pixel format, and move moov atom to the start for streaming
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # no audio; remove if you want to preserve audio
        final_output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return final_output_path

def safe_fps(value: float) -> float:
    """Ensure fps is a sensible positive value; provide a default if zero/invalid."""
    try:
        v = float(value)
        return v if v and v > 0 else 30.0
    except Exception:
        return 30.0

# ==============================
# 3) MODEL LOADING
# ==============================
@st.cache_resource
def load_model():
    """
    Loads the YOLO model from the 'best.pt' file.
    """
    model_path = "best.pt"  # Assumes best.pt is in the same folder
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found! Please upload '{model_path}' to your project folder.")
        return None
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# ==============================
# 4) SIDEBAR SETTINGS
# ==============================
st.sidebar.title("⚙ Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.35, 
    step=0.05,
    help="Higher values reduce false positives but might miss some objects."
)

st.sidebar.markdown("---")
st.sidebar.info("Select Image or Video mode to begin analysis.")

# ==============================
# 5) MAIN UI
# ==============================
st.title("🚘 Autonomous Vehicle Object Detection")
st.markdown("Analyze road scenes for cars, pedestrians, signs, and more.")

mode = st.radio("Select Input Type:", ["🖼 Image", "🎥 Video"], horizontal=True)

# ==========================================
#               IMAGE MODE
# ==========================================
if mode == "🖼 Image":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        # Load and display original
        image = Image.open(uploaded_file).convert("RGB")  # ensure RGB for consistency
        
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
                    results = model.predict(image, conf=conf_threshold, verbose=False)
                    
                    # Plot (Ultralytics returns BGR ndarray)
                    res_plotted = results[0].plot()
                    res_image = res_plotted[:, :, ::-1]  # BGR -> RGB for Streamlit display
                    
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
        tfile.flush()
        video_path = tfile.name

        # Open capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            # Video Stats
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            fps = safe_fps(fps_raw)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Video Loaded: {width}x{height} @ {fps:.2f}fps | {total_frames} frames")
            
            if st.button("▶ Start Processing", type="primary"):
                # 2. Prepare Output Video
                output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                output_path = output_tfile.name

                # Try H.264 ('avc1') for direct browser support; fall back to 'mp4v'
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                except Exception:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    st.error("Failed to open VideoWriter. Your OpenCV build may lack the required codec.")
                    cap.release()
                    os.remove(video_path)
                else:
                    # 3. Processing Loop
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Processing video frames... please wait.")
                    
                    frame_count = 0

                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Run YOLO prediction
                            results = model.predict(frame, conf=conf_threshold, verbose=False)
                            
                            # Plot detections on the frame
                            # Ultralytics plot() returns BGR ndarray compatible with cv2.VideoWriter
                            res_plotted = results[0].plot()

                            # Ensure output size matches (guard for any shape mismatch)
                            if res_plotted.shape[1] != width or res_plotted.shape[0] != height:
                                res_plotted = cv2.resize(res_plotted, (width, height), interpolation=cv2.INTER_LINEAR)

                            # Write frame to output video
                            out.write(res_plotted)
                            
                            # Update Progress
                            frame_count += 1
                            if total_frames and total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                            else:
                                # Unknown total frames: approximate progress (optional)
                                progress_bar.progress(min(frame_count / max(frame_count, 1), 1.0))
                    finally:
                        cap.release()
                        out.release()

                    # 4. Display Result
                    status_text.success("✅ Analysis Complete!")

                    # 5. Re-encode via FFmpeg for browser compatibility (if available)
                    final_output_path = output_path
                    used_ffmpeg = False
                    if ffmpeg_available():
                        try:
                            with st.spinner("Re-encoding to H.264 for browser compatibility..."):
                                final_output_path = reencode_with_ffmpeg(output_path)
                                used_ffmpeg = True
                        except Exception as e:
                            st.warning(f"FFmpeg re-encode failed, using raw OpenCV output: {e}")

                    st.subheader("Processed Video")

                    # Prefer serving by path to avoid loading entire file into memory
                    st.video(final_output_path)

                    # Download Button
                    # Read as binary for consistent download behavior
                    with open(final_output_path, 'rb') as v:
                        video_bytes = v.read()

                    st.download_button(
                        label="⬇ Download Processed Video",
                        data=video_bytes,
                        file_name="autonomous_vehicle_result.mp4",
                        mime="video/mp4"
                    )

                    # Cleanup temp input file
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

                    # If we produced an FFmpeg version, remove the intermediate OpenCV file
                    if used_ffmpeg:
                        try:
                            os.remove(output_path)
                        except Exception:
                            pass
