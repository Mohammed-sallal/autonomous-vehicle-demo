import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autonomous Vehicle Vision",
    page_icon="🚘",
    layout="wide",
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model(path: str = "best.pt"):
    """Load YOLO model from disk or return None with an error message.

    Keeps model in cache_resource so it isn't reloaded on every interaction.
    """
    if not os.path.exists(path):
        return None, f"Model file not found at '{path}'. Please upload or place the file in the app folder."

    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

# Path to model file (can be replaced by user-upload in the UI)
DEFAULT_MODEL_PATH = "best.pt"

model, model_err = load_model(DEFAULT_MODEL_PATH)

# If model not present, allow user to upload it
if model is None:
    st.warning(model_err)
    uploaded_model = st.file_uploader("Upload YOLO model (.pt)", type=["pt"])
    if uploaded_model is not None:
        # Save uploaded model to temp file and reload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as mf:
            mf.write(uploaded_model.read())
            mf.flush()
            model, model_err = load_model(mf.name)
            if model is None:
                st.error(model_err)

# --- 3. SIDEBAR SETTINGS ---
st.sidebar.title("⚙ Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01,
    help="Higher values reduce false positives but might miss some objects.",
)

process_stride = st.sidebar.slider(
    "Process every N-th frame (video)",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="Increase to speed up processing at the cost of temporal resolution.",
)

st.sidebar.markdown("---")
st.sidebar.info("Select Image or Video mode to begin analysis.")

# --- 4. MAIN INTERFACE ---
st.title("🚘 Autonomous Vehicle Object Detection")
st.markdown("Analyze road scenes for cars, pedestrians, signs, and more.")

mode = st.radio("Select Input Type:", ["🖼 Image", "🎥 Video"], horizontal=True)

# Helper: safe image display from numpy array (handles RGB/BGR confusion)
def np_to_st_image(img: np.ndarray):
    """Return an image that Streamlit can display (RGB or PIL Image).

    If input is BGR (as OpenCV), convert to RGB first.
    """
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Heuristic: if channel order is BGR (common for cv2), convert to RGB for display
        # We'll assume arrays from cv2 are BGR, but arrays from results[0].plot() are RGB.
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(img)

# IMAGE MODE
if mode == "🖼 Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file and model:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Input", use_column_width=True)

        if st.button("🔍 Detect Objects", type="primary"):
            with col2:
                st.subheader("Detection Result")
                with st.spinner("Analyzing..."):
                    # YOLO accepts PIL images; result.plot() returns RGB numpy array
                    results = model.predict(np.array(image), conf=conf_threshold, verbose=False)

                    res_rgb = results[0].plot()

                    # Convert RGB->BGR only if you need to save with cv2; for display convert to PIL
                    res_pil = Image.fromarray(res_rgb)

                    st.image(res_pil, caption="Predictions", use_column_width=True)
                    st.success(f"✅ Found {len(results[0].boxes)} objects.")

# VIDEO MODE
elif mode == "🎥 Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video and model:
        # Save input to temp file
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        try:
            in_tmp.write(uploaded_video.read())
            in_tmp.flush()
            in_tmp.close()

            cap = cv2.VideoCapture(in_tmp.name)

            if not cap.isOpened():
                st.error("Error opening video file.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 25.0  # fallback
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                st.info(f"Video Loaded: {width}x{height} @ {int(fps)}fps | {total_frames} frames")

                if st.button("▶ Start Processing", type="primary"):
                    # Prepare output file
                    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    out_path = out_tmp.name
                    out_tmp.close()

                    # Use most compatible codec
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                    # We will determine frame size dynamically from the plotted frames because plotting
                    # may change the canvas size. But VideoWriter needs a fixed size; so we'll use input size
                    # and make sure plotted frames are resized to it.
                    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    status_text.text("Processing video frames...")

                    frame_count = 0
                    processed_frames = 0

                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_count += 1

                            # Optionally skip frames to speed up processing
                            if (frame_count - 1) % process_stride != 0:
                                # write the original frame (or write last annotated) to keep timing consistent
                                out.write(frame)
                                if total_frames > 0:
                                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                                continue

                            # Run prediction on the BGR frame (YOLO can accept numpy arrays)
                            results = model.predict(frame, conf=conf_threshold, verbose=False)

                            # results[0].plot() -> RGB numpy array
                            res_rgb = results[0].plot()

                            # Ensure we have same size as original video
                            if res_rgb.shape[1] != width or res_rgb.shape[0] != height:
                                res_rgb = cv2.resize(res_rgb, (width, height))

                            # Convert RGB -> BGR for VideoWriter
                            res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

                            out.write(res_bgr)

                            processed_frames += 1

                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))

                        status_text.success("✅ Analysis Complete!")

                        # Close writers and capture
                        cap.release()
                        out.release()

                        # Read output and show
                        with open(out_path, 'rb') as f:
                            video_bytes = f.read()

                        st.subheader("Processed Video")
                        st.video(video_bytes)

                        st.download_button(
                            label="⬇ Download Processed Video",
                            data=video_bytes,
                            file_name="autonomous_vehicle_result.mp4",
                            mime="video/mp4",
                        )

                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                        try:
                            cap.release()
                            out.release()
                        except:
                            pass
                    finally:
                        # Cleanup temp files
                        if os.path.exists(in_tmp.name):
                            try:
                                os.remove(in_tmp.name)
                            except:
                                pass
                        if os.path.exists(out_path):
                            # keep the file for download button; remove if user wants
                            pass

        except Exception as e:
            st.error(f"Failed to save uploaded video: {e}")
        # don't remove in_tmp here if processing hasn't started

# Footer
st.markdown("---")
st.caption("Optimized Streamlit app: correct RGB/BGR handling, compatible codec (mp4v), optional frame skipping for speed.")
