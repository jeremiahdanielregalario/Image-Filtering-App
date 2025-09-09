# streamlit app for imagefilter

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
from PIL import Image
import io
import random
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import base64

# Helpers

PRIMARY = "#6C63FF"         # purple
ACCENT = "#00C2FF"          # cyan
CARD_BG = "rgba(255,255,255,0.03)"
TEXT = "#EDEFF6"
BG_GRADIENT = "linear-gradient(135deg, #0f172a 0%, #0b2545 40%, #08121f 100%)"

st.set_page_config(page_title="ImageFilter — Live & Upload", layout="wide", page_icon="🎨")

with open("images/logo.png", "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data).decode()
    

def inject_css():
    css = f"""
    <style>
    /* page background */
    .stApp {{
        background: {BG_GRADIENT};
        color: {TEXT};
    }}

    /* hero card */
    .hero {{
        border-radius: 16px;
        padding: 22px;
        background: linear-gradient(90deg, rgba(108,99,255,0.06), rgba(0,194,255,0.02));
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        margin-bottom: 12px;
    }}

    .cta-btn {{
        display:inline-block;
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
        color: white !important;
        padding: 10px 16px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
    }}

    .controls-card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 12px;
    }}

    .muted {{ color: #9aa6bf; font-size: 13px; }}
    button {{ border-radius: 10px; padding: 8px 12px; }}

    /* ---------- webrtc/video-specific styling ---------- */
    /* Make video & canvas elements follow the app design and fill their column */
    .stApp video,
    .stApp canvas,
    .stApp .stVideo > div > video,
    .stApp .stVideo > div > canvas,
    .stApp .streamlit-webrtc > div > video,
    .stApp .streamlit-webrtc > div > canvas {{
      width: 100% !important;
      max-width: 100% !important;
      height: auto !important;
      max-height: 520px !important;
      object-fit: cover !important;
      border-radius: 12px !important;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.06));
      box-shadow: 0 12px 30px rgba(0,0,0,0.45);
      display: block !important;
    }}

    /* Tidy up the video container */
    .stApp .streamlit-webrtc,
    .stApp .stVideo {{
      background: transparent !important;
      padding: 8px 0 0 0 !important;
    }}

    /* Hide any placeholder elements that use 'placeholder' in the class name (aggressive) */
    .stApp div[class*="placeholder"],
    .stApp div[class*="placeholder"] * {{
      display: none !important;
      visibility: hidden !important;
    }}

    /* If the webrtc container does NOT have a <video> element yet, show a custom polished placeholder.
       Uses :not(:has(video)) so it only appears when no video is present (modern browsers). */
    .stApp .streamlit-webrtc > div:not(:has(video)) {{
      min-height: 320px;
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.06));
      display: flex;
      align-items: center;
      justify-content: center;
      color: #bcd3ff;
      font-size: 18px;
      position: relative;
      box-shadow: 0 10px 28px rgba(0,0,0,0.45);
    }}

    /* Nice centered icon/text for the custom placeholder */
    .stApp .streamlit-webrtc > div:not(:has(video))::before {{
      content: "Camera not started — click START";
      font-weight: 600;
      letter-spacing: 0.2px;
      display: block;
    }}
    .stApp .streamlit-webrtc > div:not(:has(video))::after {{
      content: "";
      width: 64px;
      height: 42px;
      border-radius: 6px;
      border: 6px solid rgba(255,255,255,0.9);
      position: absolute;
      top: 36%;
      transform: translateY(-50%);
      box-sizing: border-box;
      opacity: 0.95;
    }}

    /* Style for the control bar that appears below the video (device selector / start) */
    .stApp .webrtc-control-panel,
    .stApp .webrtc-controls,
    .stApp .streamlit-webrtc .control-panel {{
      background: rgba(0,0,0,0.28) !important;
      border-radius: 0 0 12px 12px !important;
      padding: 8px 12px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: space-between !important;
      color: #fff !important;
    }}

    .stApp .webrtc-control-panel button,
    .stApp .webrtc-controls button,
    .stApp .streamlit-webrtc button {{
      border-radius: 10px !important;
      padding: 8px 12px !important;
      font-weight: 700 !important;
      text-transform: uppercase;
    }}

    /* Make device select float right and appear clean */
    .stApp select {{
      background: rgba(255,255,255,0.02) !important;
      color: white !important;
      border-radius: 8px !important;
      padding: 6px 10px !important;
    }}

    @media (max-width: 800px) {{
      .stApp video,
      .stApp canvas {{
        max-height: 320px !important;
      }}
      .stApp .streamlit-webrtc > div:not(:has(video))::before {{
        font-size: 15px;
      }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
def hero_section():
    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex; align-items:center; gap:14px; text-align: center;">
                <div style='flex:1'>
                    <img src="data:image/png;base64,{encoded}" width="600">
                    <p style="margin:0.25rem 0 8px 0; color:#bcd3ff; font-size:15px;">Apply fun filters to your Pokémons and experience a full Pokémon playground preserved.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def to_bytes_from_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_cv2(pil_image):
    arr = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# Filters/Transforms

def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_sepia(img, intensity=0.8):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    img_float = img.astype(np.float32) / 255.0
    sepia = cv2.transform(img_float, kernel)
    sepia = np.clip(sepia * intensity + img_float * (1 - intensity), 0, 1)
    return (sepia * 255).astype(np.uint8)


def apply_pencil_sketch(img, sigma=10, shade_factor=0.02):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigma)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    sketch_bgr = (sketch_bgr * (1 - shade_factor)).astype(np.uint8)
    return sketch_bgr


def apply_cartoonify(img, num_bilateral=7):
    img_color = img.copy()
    for _ in range(max(1, num_bilateral)):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    edges = cv2.adaptiveThreshold(img_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9, C=2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, edges)
    return cartoon


def apply_canny(img, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr


def apply_blur(img, ksize=5):
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_posterize(img: np.ndarray, levels: int = 4) -> np.ndarray:
    # Reduce number of color levels
    factor = 256 // levels
    poster = (img // factor) * factor + factor // 2
    poster = np.clip(poster, 0, 255).astype(np.uint8)
    return poster


def apply_negative(img: np.ndarray) -> np.ndarray:
    return 255 - img


def apply_emboss(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    embossed = cv2.filter2D(img, -1, kernel) + 128
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)
    return embossed


def transform_rotate(img, angle=0):
    if angle == 0:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def transform_scale(img, scale=1.0):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(img)
    start_x = max((w - new_w) // 2, 0)
    start_y = max((h - new_h) // 2, 0)
    end_x = start_x + min(new_w, w)
    end_y = start_y + min(new_h, h)
    crop_w = min(new_w, w)
    crop_h = min(new_h, h)
    canvas[start_y:end_y, start_x:end_x] = resized[0:crop_h, 0:crop_w]
    return canvas

# Live Cam Transforms

class OpenCVTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter_name = "None"
        self.rotate = 0
        self.scale = 1.0
        self.sepia_int = 0.8
        self.pencil_sigma = 10
        self.cartoon_bilateral = 7
        self.canny_t1 = 100
        self.canny_t2 = 200
        self.blur_k = 5
        self.proc_w = None
        self.proc_h = None
        self.posterize_level = 4
        self.performance_mode = False
        self.frame = None

        # Roulette control inside the transformer (for Live mode)
        self.roulette_enabled = False
        self.roulette_interval = 5.0
        self.roulette_choices = [
            "Grayscale", 
            "Sepia", 
            "Pencil Sketch", 
            "Cartoonify", 
            "Canny Edges", 
            "Gaussian Blur",
            "Emboss",
            "Posterize",
            "Negative"
        ]
        self.roulette_last_time = time.time()

    def transform(self, frame):
        # If roulette is enabled, possibly change filter based on time elapsed
        if self.roulette_enabled:
            try:
                now = time.time()
                if now - self.roulette_last_time >= float(self.roulette_interval):
                    choice = random.choice(self.roulette_choices)
                    # apply the choice (overrides current filter_name)
                    self.filter_name = choice
                    self.roulette_last_time = now
            except Exception:
                pass

        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        proc_w = self.proc_w if self.proc_w is not None else w
        proc_h = self.proc_h if self.proc_h is not None else h
        proc_w = min(proc_w, w)
        proc_h = min(proc_h, h)

        if (proc_w, proc_h) != (w, h):
            proc = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc = img.copy()

        cartoon_passes = max(1, int(self.cartoon_bilateral / (2 if self.performance_mode else 1)))
        blur_k = max(1, int(self.blur_k / (2 if self.performance_mode else 1)))
        pencil_sigma = max(1, int(self.pencil_sigma / (2 if self.performance_mode else 1)))

        try:
            if self.filter_name == "None":
                out_small = proc
            elif self.filter_name == "Grayscale":
                out_small = apply_grayscale(proc)
            elif self.filter_name == "Sepia":
                out_small = apply_sepia(proc, intensity=self.sepia_int)
            elif self.filter_name == "Pencil Sketch":
                out_small = apply_pencil_sketch(proc, sigma=pencil_sigma)
            elif self.filter_name == "Cartoonify":
                out_small = apply_cartoonify(proc, num_bilateral=cartoon_passes)
            elif self.filter_name == "Canny Edges":
                out_small = apply_canny(proc, threshold1=self.canny_t1, threshold2=self.canny_t2)
            elif self.filter_name == "Gaussian Blur":
                out_small = apply_blur(proc, ksize=blur_k)
            elif self.filter_name == "Emboss":
                out_small = apply_emboss(proc)
            elif self.filter_name == "Negative":
                out_small = apply_negative(proc)
            elif self.filter_name == "Posterize":
                out_small = apply_posterize(proc, levels=4)
            else:
                out_small = proc
        except Exception as e:
            print("Transformer error:", e)
            out_small = proc

        if (proc_w, proc_h) != (w, h):
            out = cv2.resize(out_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            out = out_small

        if self.rotate != 0:
            out = transform_rotate(out, self.rotate)
        if self.scale != 1.0:
            out = transform_scale(out, self.scale)

        self.frame = out.copy()
        return out

# Streamlit App ##############################################################################################################################

def main():
    st.set_page_config(page_title="Live Image Filters App", layout="wide")

    if 'input_mode' not in st.session_state:
        st.session_state['input_mode'] = 'Live'
    if 'upload_roulette_last' not in st.session_state:
        st.session_state['upload_roulette_last'] = 0.0
    if 'upload_roulette_filter' not in st.session_state:
        st.session_state['upload_roulette_filter'] = None

    # Sidebar (input mode buttons)
    with st.sidebar:
        st.header("Input Mode")
        b1 = st.button("📷 Live Webcam")
        b2 = st.button("🖼️ Upload Image")
        b3 = st.button("🕹️ Pokémon Filter")
        if b1:
            st.session_state['input_mode'] = 'Live'
        if b2:
            st.session_state['input_mode'] = 'Upload'
        if b3:
            st.session_state['input_mode'] = 'Pokemon'
            
    if st.session_state['input_mode'] == 'Pokemon':
        inject_css()    
        hero_section()


    # Two cols: left (video/upload preview), right (controls)
    left_col, right_col = st.columns([2, 1])

    # controls (right col) 
    with right_col:
        st.header("Filter & Transform Settings")

        selected_filter = st.selectbox("Choose filter", [
            "None", 
            "Grayscale", 
            "Sepia", 
            "Pencil Sketch", 
            "Cartoonify", 
            "Canny Edges", 
            "Gaussian Blur", 
            "Emboss", 
            "Posterize", 
            "Negative"
        ]) 

        # Rotate
        r_col1, r_col2 = st.columns([3,1])
        rotate_slider = r_col1.slider("Rotate (degrees)", -180, 180, 0, step=1)
        rotate_typed = r_col2.number_input("", min_value=-180, max_value=180, value=rotate_slider, step=1, format="%d")
        rotate_angle = int(rotate_typed)

        # Scale
        s_col1, s_col2 = st.columns([3,1])
        scale_slider = s_col1.slider("Scale (1.0 = original)", 0.2, 2.0, 1.0, step=0.05)
        scale_typed = s_col2.number_input("", min_value=0.2, max_value=2.0, value=float(scale_slider), step=0.01, format="%.2f")
        scale_val = float(scale_typed)

        # filter
        if selected_filter == "Sepia":
            sep_col1, sep_col2 = st.columns([3,1])
            sep_slider = sep_col1.slider("Sepia intensity", 0.0, 1.0, 0.8, step=0.01)
            sep_typed = sep_col2.number_input("", min_value=0.0, max_value=1.0, value=float(sep_slider), step=0.01, format="%.2f")
            sepia_int = float(sep_typed)
        else:
            sepia_int = 0.8

        if selected_filter == "Pencil Sketch":
            p_col1, p_col2 = st.columns([3,1])
            p_slider = p_col1.slider("Sketch blur sigma", 1, 50, 10)
            p_typed = p_col2.number_input("", min_value=1, max_value=200, value=int(p_slider), step=1, format="%d")
            pencil_sigma = int(p_typed)
        else:
            pencil_sigma = 10

        if selected_filter == "Canny Edges":
            c_col1, c_col2 = st.columns([3,1])
            t1_slider = c_col1.slider("Canny threshold1", 10, 300, 100)
            t1_typed = c_col2.number_input("", min_value=10, max_value=1000, value=int(t1_slider), step=1, format="%d")
            t1 = int(t1_typed)

            c2_col1, c2_col2 = st.columns([3,1])
            t2_slider = c2_col1.slider("Canny threshold2", 10, 400, 200)
            t2_typed = c2_col2.number_input("", min_value=10, max_value=2000, value=int(t2_slider), step=1, format="%d")
            t2 = int(t2_typed)
        else:
            t1, t2 = 100, 200

        if selected_filter == "Gaussian Blur":
            b_col1, b_col2 = st.columns([3,1])
            blur_slider = b_col1.slider("Blur kernel (odd) e.g. 3,5,7", 1, 31, 5, step=2)
            blur_typed = b_col2.number_input("", min_value=1, max_value=99, value=int(blur_slider), step=2, format="%d")
            blur_k = int(blur_typed)
        else:
            blur_k = 5

        if selected_filter == "Cartoonify":
            cart_col1, cart_col2 = st.columns([3,1])
            cart_slider = cart_col1.slider("Bilateral filter passes", 1, 20, 7)
            cart_typed = cart_col2.number_input("", min_value=1, max_value=50, value=int(cart_slider), step=1, format="%d")
            bilateral_count = int(cart_typed)
        else:
            bilateral_count = 7
            
        if selected_filter == "Posterize":
            cart_col1, cart_col2 = st.columns([3,1])
            cart_slider = cart_col1.slider("Posterize level", 1, 12, 4)
            cart_typed = cart_col2.number_input("", min_value=1, max_value=12, value=int(cart_slider), step=1, format="%d")
            posterize_level = int(cart_typed)
        else:
            posterize_level = 4

        st.markdown("---")
        # Live controls (resolution / perf / async)
        if st.session_state['input_mode'] == 'Live':
            st.subheader("Live webcam options")
            res_choice = st.selectbox("Requested camera resolution", ["High (1280×720)", "Medium (640×480)", "Low (320×240)"])
            perf_mode = st.checkbox("Performance mode (reduce heavy params for smoother FPS)")
            async_transform_ui = st.checkbox("Async transform (lower latency)", value=True)
        else:
            # defaults (not live)
            res_choice = "High (1280×720)"
            perf_mode = False
            async_transform_ui = True

        st.subheader("Filter Roulette")
        # manual roulette controls
        roulette_manual = st.checkbox("Enable manual Filter Roulette (checkbox)")
        roulette_next = st.button("Next random filter (manual)")

        # automatic roulette control
        auto_roulette = st.checkbox("Enable auto filter roulette (every n seconds)")
        roulette_interval = st.number_input("Interval (seconds)", min_value=1, max_value=3600, value=5, step=1)

        if roulette_manual and roulette_next:
            choices = [
                "Grayscale", 
                "Sepia", 
                "Pencil Sketch", 
                "Cartoonify", 
                "Canny Edges", 
                "Gaussian Blur", 
                "Negative", 
                "Posterize", 
                "Emboss"
            ]
            selected_filter = random.choice(choices)
            st.info(f"Roulette picked: {selected_filter}")

        st.markdown("---")
        st.write("Tip: Type exact values into the small boxes to the right of sliders for precise control.")

    # Resolution choice
    if res_choice.startswith("Low"):
        req_w, req_h = 320, 240
        proc_w, proc_h = 320, 240
    elif res_choice.startswith("Medium"):
        req_w, req_h = 640, 480
        proc_w, proc_h = 480, 360
    else:
        req_w, req_h = 1280, 720
        proc_w, proc_h = 640, 360

    # Choices list used by both modes
    choices = [
        "Grayscale", 
        "Sepia", 
        "Pencil Sketch", 
        "Cartoonify", 
        "Canny Edges", 
        "Gaussian Blur", 
        "Negative", 
        "Posterize", 
        "Emboss"
    ]

    # Left Col (video or uploader) 
    with left_col:

        if st.session_state['input_mode'] == 'Live':
            st.title("🎥 ImageFilter")
            st.header("Output / Video")
            media_constraints = {
                "video": {
                    "width": {"ideal": req_w},
                    "height": {"ideal": req_h},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            }

            try:
                rtc_config = eval(rtc_json)
            except Exception:
                rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

            webrtc_ctx = webrtc_streamer(
                key="live-filter",
                video_transformer_factory=OpenCVTransformer,
                media_stream_constraints=media_constraints,
                rtc_configuration=rtc_config,
                async_transform=async_transform_ui,
            )

            if webrtc_ctx.video_transformer is not None:
                tr = webrtc_ctx.video_transformer

                # auto_roulette is enabled
                tr.roulette_enabled = auto_roulette
                tr.roulette_interval = float(roulette_interval)
                tr.roulette_choices = choices

                # auto_roulette is NOT active
                if not auto_roulette:
                    tr.filter_name = selected_filter
                # update params
                tr.rotate = rotate_angle
                tr.scale = scale_val
                tr.sepia_int = sepia_int
                tr.pencil_sigma = pencil_sigma
                tr.cartoon_bilateral = bilateral_count
                tr.canny_t1 = t1
                tr.canny_t2 = t2
                tr.blur_k = blur_k
                tr.proc_w = proc_w
                tr.proc_h = proc_h
                tr.performance_mode = perf_mode
                tr.posterize_level = posterize_level

                if st.button("Take snapshot of processed frame"):
                    if tr.frame is not None:
                        data = to_bytes_from_bgr(tr.frame)
                        st.download_button("Download snapshot PNG", data=data, file_name="snapshot.png", mime="image/png")
                    else:
                        st.warning("No frame available yet — wait for the video to initialize.")

        elif st.session_state['input_mode'] == 'Upload':
            st.title("🎥 ImageFilter")
            st.header("Output / Video")
            uploaded = st.file_uploader("Upload an image (png/jpg)", type=["png", "jpg", "jpeg"]) 

            if uploaded is not None:
                img_cv = pil_to_cv2(Image.open(uploaded))

                # auto roulette is enabled in Upload mode
                if auto_roulette:
                    now = time.time()
                    last = float(st.session_state.get('upload_roulette_last', 0.0))
                    if now - last >= float(roulette_interval):
                        st.session_state['upload_roulette_filter'] = random.choice(choices)
                        st.session_state['upload_roulette_last'] = now
                        # trigger rerun 
                        st.rerun()

                    if st.session_state.get('upload_roulette_filter') is not None:
                        selected_filter = st.session_state['upload_roulette_filter']
                        

                # apply transforms/filters
                processed = img_cv.copy()
                processed = transform_rotate(processed, rotate_angle)
                processed = transform_scale(processed, scale_val)

                if selected_filter == "Grayscale":
                    processed = apply_grayscale(processed)
                elif selected_filter == "Sepia":
                    processed = apply_sepia(processed, intensity=sepia_int)
                elif selected_filter == "Pencil Sketch":
                    processed = apply_pencil_sketch(processed, sigma=pencil_sigma)
                elif selected_filter == "Cartoonify":
                    processed = apply_cartoonify(processed, num_bilateral=bilateral_count)
                elif selected_filter == "Canny Edges":
                    processed = apply_canny(processed, threshold1=t1, threshold2=t2)
                elif selected_filter == "Gaussian Blur":
                    processed = apply_blur(processed, ksize=blur_k)
                elif selected_filter == "Posterize":
                    processed = apply_posterize(processed, levels=posterize_level)
                elif selected_filter == "Negative":
                    processed = apply_negative(processed)
                elif selected_filter == "Emboss":
                    processed = apply_emboss(processed)

                st.subheader("Original")
                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), width=600)
                st.subheader("Processed")
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width=600)
                data = to_bytes_from_bgr(processed)
                st.download_button("Download processed image", data=data, file_name="processed.png", mime="image/png")

##############################################################################################################################################    
        elif st.session_state['input_mode'] == 'Pokemon':
            #Load data
            df = pd.read_csv(fr"pokemon.csv")
            # Ensure Type1/Type2 columns are combined into a single "Type" (list)
            df["Type"] = df[["Type 1", "Type 2"]].apply(
                lambda x: [t for t in x if pd.notna(t)], axis=1
            )

            #Pokemon list
            pokemons = ['Pikachu', 'Venusaur', 'Charizard', 'Blastoise']

            #Header
            col1, col2, col3 = st.columns([0.4, 1, 0.4])
            

            #Choose your pokemon
            pokemon = st.sidebar.selectbox("Choose your Pokemon", pokemons)

            
            #Image
            image = Image.open(fr'images/{pokemon}.png')
            img_cv = pil_to_cv2(image)

            #apply filters/transformations
            processed = img_cv.copy()
            processed = transform_rotate(processed, rotate_angle)
            processed = transform_scale(processed, scale_val)

            if selected_filter == "Grayscale":
                processed = apply_grayscale(processed)
            elif selected_filter == "Sepia":
                processed = apply_sepia(processed, intensity=sepia_int)
            elif selected_filter == "Pencil Sketch":
                processed = apply_pencil_sketch(processed, sigma=pencil_sigma)
            elif selected_filter == "Cartoonify":
                processed = apply_cartoonify(processed, num_bilateral=bilateral_count)
            elif selected_filter == "Canny Edges":
                processed = apply_canny(processed, threshold1=t1, threshold2=t2)
            elif selected_filter == "Gaussian Blur":
                processed = apply_blur(processed, ksize=blur_k)
            elif selected_filter == "Posterize":
                processed = apply_posterize(processed, levels=posterize_level)
            elif selected_filter == "Negative":
                processed = apply_negative(processed)
            elif selected_filter == "Emboss":
                processed = apply_emboss(processed)

            col1, col2, col3 = st.columns([0.8, 1, 0.8])

            with col2:
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width=300)
                if selected_filter == "None" and rotate_angle == 0 and scale_val == 1:
                    st.subheader(f"{pokemon}")
                elif selected_filter == "None" and not (rotate_angle == 0 and scale_val == 1):
                    st.subheader(f"R{rotate_angle}S{scale_val} {pokemon}")
                else:
                    st.subheader(f"R{rotate_angle}S{scale_val} {selected_filter} {pokemon}")

            # Pokeomon Stats

            def plot_pokemon_stats(df, pokemon_name, filter_name="None", rotate=0, scale=1.0):
                stats_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
                pokemon_row = df[df["Name"] == pokemon_name].iloc[0]

                # --- Base stats dictionary
                base_stats = {stat: int(pokemon_row[stat]) for stat in stats_cols}

                # --- Apply random shuffling of stats first via rotation
                if rotate != 0:
                    shuffled_stats = list(base_stats.values())
                    rng = random.Random(rotate)  # seed with rotate value
                    rng.shuffle(shuffled_stats)
                    s = {stat: shuffled_stats[i] for i, stat in enumerate(stats_cols)}
                else:
                    s = base_stats.copy()

                # --- Apply filter modifications AFTER rotation
                if filter_name == "Grayscale":
                    s["HP"] += 20
                    s["Defense"] += 20
                    s["Attack"] -= 20
                    s["Sp. Atk"] -= 20

                elif filter_name == "Sepia":
                    s["HP"] -= 20
                    s["Defense"] -= 20
                    s["Attack"] += 20
                    s["Sp. Atk"] += 20

                elif filter_name == "Pencil Sketch":
                    s["Attack"], s["Sp. Atk"] = s["Sp. Atk"], s["Attack"]
                    s["HP"] -= 10
                    s["Defense"] += 10

                elif filter_name == "Cartoonify":
                    s["HP"], s["Defense"] = s["Defense"], s["HP"]
                    s["Attack"] -= 10
                    s["Sp. Atk"] += 10

                elif filter_name == "Canny Edges":
                    s["Sp. Def"] += 20
                    s["Defense"] += 20
                    s["Attack"] -= 20
                    s["Sp. Atk"] -= 20

                elif filter_name == "Gaussian Blur":
                    s["Sp. Def"] -= 20
                    s["Defense"] -= 20
                    s["Attack"] += 20
                    s["Sp. Atk"] += 20

                elif filter_name == "Emboss":
                    s["Attack"], s["Defense"] = s["Defense"], s["Attack"]
                    s["Sp. Atk"] -= 10
                    s["Sp. Def"] -= 10

                elif filter_name == "Posterize":
                    s["Sp. Atk"], s["Sp. Def"] = s["Sp. Def"], s["Sp. Atk"]
                    s["Attack"] -= 10
                    s["Defense"] -= 10

                elif filter_name == "Negative":
                    s["HP"], s["Sp. Def"] = s["Sp. Def"], s["HP"]
                    s["Defense"] -= 10
                    s["Attack"] += 10

                # --- Apply scaling: multiply Speed
                s["Speed"] = int(s["Speed"] * 1/(scale))

                # --- Extract values for plotting
                values = [s[stat] for stat in stats_cols]
                total = sum(values)
                

                colors = ["#FFD34E"] * 6  # yellow
                colors[3] = "#8BC34A"     # Sp. Atk green
                colors[4] = "#8BC34A"     # Sp. Def green

                fig = go.Figure()

                for i, stat in enumerate(stats_cols):
                    fig.add_trace(go.Bar(
                        x=[values[i]],
                        y=[stat],
                        orientation="h",
                        marker=dict(color=colors[i]),
                        width=0.6,
                        name=stat,
                        text=[str(values[i])],
                        textposition="inside",
                        insidetextanchor="start"
                    ))

                # --- Layout with black background & white text
                fig.update_layout(
                    title=f"{pokemon_name} Base Stats (Total: {total})",
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 400]),
                    yaxis=dict(categoryorder="array", categoryarray=stats_cols[::-1]),
                    barmode="stack",
                    height=400,
                    margin=dict(l=80, r=80, t=50, b=30),
                    plot_bgcolor="black",
                    paper_bgcolor="black",
                    font=dict(color="white")
                )
                s["Name"] = pokemon_name
                s["Type"] = pokemon_row["Type"]

                return fig, s
            st.subheader("Base Stats")
            fig, modified_stats = plot_pokemon_stats(df, pokemon, filter_name=selected_filter, rotate=rotate_angle, scale=scale_val)
            st.plotly_chart(fig, use_container_width=True)
            
            # Battle
            col1, col2, col3 = st.columns([0.4, 1,0.4])
            with col2:
                st.image(fr"images/battle_logo.png", width=500)

            # Type effectiveness chart
            type_chart = {
                "bug":     {"strong": ["grass", "dark", "psychic"], "weak": ["fire", "flying", "rock"]},
                "dark":    {"strong": ["ghost", "psychic"], "weak": ["bug", "fairy", "fighting"]},
                "dragon":  {"strong": ["dragon"], "weak": ["dragon", "fairy", "ice"]},
                "electric":{"strong": ["flying", "water"], "weak": ["ground"]},
                "fairy":   {"strong": ["fighting", "dark", "dragon"], "weak": ["poison", "steel"]},
                "fighting":{"strong": ["dark", "ice", "normal", "rock", "steel"], "weak": ["fairy", "flying", "psychic"]},
                "fire":    {"strong": ["bug", "grass", "ice", "steel"], "weak": ["ground", "rock", "water"]},
                "flying":  {"strong": ["bug", "fighting", "grass"], "weak": ["electric", "ice", "rock"]},
                "ghost":   {"strong": ["ghost", "psychic"], "weak": ["dark", "ghost"]},
                "grass":   {"strong": ["ground", "rock", "water"], "weak": ["bug", "fire", "flying", "ice", "poison"]},
                "ground":  {"strong": ["electric", "fire", "poison", "rock", "steel"], "weak": ["grass", "ice", "water"]},
                "ice":     {"strong": ["dragon", "flying", "grass", "ground"], "weak": ["fighting", "fire", "rock", "steel"]},
                "normal":  {"strong": [], "weak": ["fighting"]},
                "poison":  {"strong": ["fairy", "grass"], "weak": ["ground", "psychic"]},
                "psychic": {"strong": ["fighting", "poison"], "weak": ["bug", "dark", "ghost"]},
                "rock":    {"strong": ["bug", "fire", "flying", "ice"], "weak": ["fighting", "grass", "ground", "steel", "water"]},
                "steel":   {"strong": ["fairy", "ice", "rock"], "weak": ["fighting", "fire", "ground"]},
                "water":   {"strong": ["fire", "ground", "rock"], "weak": ["electric", "grass"]}
            }

            def calculate_multiplier(move_type, target_type):
                move_type = move_type.lower()
                target_type = target_type.lower()

                if target_type in type_chart[move_type]["strong"]:
                    return 2.0
                elif target_type in type_chart[move_type]["weak"]:
                    return 0.5
                else:
                    return 1.0

            # Define moves (simplified strongest moves)
            pokemon_moves = {
                "Pikachu": {
                    "Thunderbolt": {"power": 90, "atk_type": "Special", "type": "electric"},
                    "Iron Tail": {"power": 100, "atk_type": "Attack", "type": "steel"},
                    "Thunder": {"power": 110, "atk_type": "Special", "type": "electric"},
                    "Body Slam": {"power": 85, "atk_type": "Attack", "type": "normal"}
                },
                "Charizard": {
                    "Flamethrower": {"power": 90, "atk_type": "Special", "type": "fire"},
                    "Slash": {"power": 75, "atk_type": "Attack", "type": "normal"},
                    "Dragon Claw": {"power": 80, "atk_type": "Attack", "type": "dragon"},
                    "Air Slash": {"power": 75, "atk_type": "Attack", "type": "flying"}
                },
                "Blastoise": {
                    "Hydro Pump": {"power": 110, "atk_type": "Special", "type": "water"},
                    "Bite": {"power": 60, "atk_type": "Attack", "type": "dark"},
                    "Brick Break": {"power": 75, "atk_type": "Attack", "type": "fighting"},
                    "Ice Punch": {"power": 75, "atk_type": "Attack", "type": "ice"}
                },
                "Venusaur": {
                    "Solar Beam": {"power": 120, "atk_type": "Special", "type": "grass"},
                    "Take Down": {"power": 90, "atk_type": "Attack", "type": "normal"},
                    "Poison Jab": {"power": 80, "atk_type": "Attack", "type": "poison"},
                    "Earth Power": {"power": 90, "atk_type": "Special", "type": "ground"}
                }
            }

            # Simple damage calculation
            def calculate_damage(attacker, defender, move_name, move):
                if move["atk_type"] == "Special":
                    atk = attacker["Sp. Atk"]
                    defense = defender["Sp. Def"]
                else:
                    atk = attacker["Attack"]
                    defense = defender["Defense"]

                # Basic Pokémon-style formula (simplified)
                base_damage = ((((2 * 50 / 5) + 2) * move["power"] * atk / defense) / 50) + 2
                # Add randomness (±15%)
                damage = base_damage * random.uniform(0.85, 1.0)

                # Ensure defender["Type"] is always a list
                defender_types = defender["Type"]
                if isinstance(defender_types, str):
                    defender_types = [defender_types]
                # Type effectiveness 
                multiplier = 1.0
                move_type = move["type"].lower()
                for target_type in defender["Type"]:  # works for dual-types
                    multiplier *= calculate_multiplier(move_type, target_type)

                # Final damage (at least 1)
                return int(max(1, damage * multiplier))

            # Convert stats row to dict
            def pokemon_from_df(df, name, stats_dict=None):
                stats_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
                row = df[df["Name"] == name].iloc[0]

                if stats_dict:  # use filtered/modified stats
                    stats = stats_dict.copy()
                else:
                    stats = {stat: int(row[stat]) for stat in stats_cols}

                stats["Name"] = name
                stats["Type"] = [t.lower() for t in row["Type"]]  # ensure lowercase list
                return stats

            # Battle system
            def battle(df, my_pokemon_stats, opponent_name):

                # Setup Pokémon
                my_pokemon = my_pokemon_stats
                opponent = pokemon_from_df(df, opponent_name)

                # Initialize battle state
                if "battle_state" not in st.session_state or st.button("Restart Battle"):
                    st.session_state.battle_state = {
                        "my_hp": my_pokemon["HP"] * 3,
                        "opp_hp": opponent["HP"] * 3,
                        "log": [],
                        "winner": None
                    }

                battle = st.session_state.battle_state

                col1, col2 = st.columns([1, 1])
                with col1:
                    # Show HP bars
                    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width=300)
                    st.write(f"**{my_pokemon['Name']} HP:** {battle['my_hp']} / {my_pokemon['HP'] * 3}")
                    st.progress(min(1.0, battle['my_hp'] / (my_pokemon["HP"] * 3)))
                with col2:
                    st.image(Image.open(fr'images/{opponent_name}.png'), width=300)
                    st.write(f"**{opponent['Name']} HP:** {battle['opp_hp']} / {opponent['HP'] * 3}")
                    st.progress(min(1.0, (battle['opp_hp'])/ (opponent["HP"] * 3)))

                # Check for winner
                if battle["winner"]:
                    st.success(f"🎉 {battle['winner']} wins!")
                    return

                # Player turn (full round happens here)
                st.write("Your turn! Choose a move:")
                moves = list(pokemon_moves[my_pokemon["Name"]].keys())
                for move in moves:
                    if st.button(move):
                        # Determine turn order using Speed
                        if my_pokemon["Speed"] > opponent["Speed"]:
                            first, second = "player", "opponent"
                        elif my_pokemon["Speed"] < opponent["Speed"]:
                            first, second = "opponent", "player"
                        else:
                            first = random.choice(["player", "opponent"])
                            second = "player" if first == "opponent" else "opponent"

                        # Execute first attacker
                        if first == "player":
                            # Player attacks
                            damage = calculate_damage(my_pokemon, opponent, move, pokemon_moves[my_pokemon["Name"]][move])
                            battle["opp_hp"] = max(0, battle["opp_hp"] - damage)
                            battle["log"].append(f"{my_pokemon['Name']} used {move}! It dealt {damage} damage.")

                            if battle["opp_hp"] <= 0:
                                battle["winner"] = my_pokemon["Name"]
                                st.rerun()

                        else:
                            # Opponent attacks first
                            opp_move = random.choice(list(pokemon_moves[opponent["Name"]].keys())) # random choice
                            damage = calculate_damage(opponent, my_pokemon, opp_move, pokemon_moves[opponent["Name"]][opp_move])
                            battle["my_hp"] = max(0, battle["my_hp"] - damage)
                            battle["log"].append(f"{opponent['Name']} used {opp_move}! It dealt {damage} damage.")

                            if battle["my_hp"] <= 0:
                                battle["winner"] = opponent["Name"]
                                st.rerun()

                        # Execute second attacker (if no winner yet)
                        if not battle["winner"]:
                            if second == "player":
                                damage = calculate_damage(my_pokemon, opponent, move, pokemon_moves[my_pokemon["Name"]][move])
                                battle["opp_hp"] = max(0, battle["opp_hp"] - damage)
                                battle["log"].append(f"{my_pokemon['Name']} used {move}! It dealt {damage} damage.")
                                if battle["opp_hp"] <= 0:
                                    battle["winner"] = my_pokemon["Name"]
                            else:
                                opp_move = random.choice(list(pokemon_moves[opponent["Name"]].keys()))
                                damage = calculate_damage(opponent, my_pokemon, opp_move, pokemon_moves[opponent["Name"]][opp_move])
                                battle["my_hp"] = max(0, battle["my_hp"] - damage)
                                battle["log"].append(f"{opponent['Name']} used {opp_move}! It dealt {damage} damage.")
                                if battle["my_hp"] <= 0:
                                    battle["winner"] = opponent["Name"]

                        # Force Streamlit to update immediately
                        st.rerun()

                # Show battle log
                st.write("### Battle Log")
                for entry in battle["log"]:
                    st.write(entry)
            
            opponent = st.selectbox("Choose your opponent:", ["Charizard", "Blastoise", "Venusaur", "Pikachu"])
            battle(df, modified_stats, opponent)

    st.markdown("---")
    st.markdown("**Project by:** Jeremiah Daniel Regalario, Isaiah John Mariano, Meluisa Montealto")

if __name__ == '__main__':
    main()
