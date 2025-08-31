# ImageFilter — Live Image Filters App (Streamlit)

A Streamlit app that applies real-time image filters to webcam input (via WebRTC) or uploaded images. Includes a variety of creative filters (grayscale, sepia, pencil sketch, cartoonify, canny edges, blur, posterize, emboss, negative) and image transforms (rotate, scale). Also supports a *Filter Roulette* mode for automatic/random filter changes.

---

## Features
- Live webcam processing using `streamlit-webrtc` for low-latency browser-based transforms.
- Upload and process static images (PNG / JPG).
- Filters: Grayscale, Sepia, Pencil Sketch, Cartoonify, Canny Edges, Gaussian Blur, Posterize, Negative, Emboss.
- Transforms: Rotation and scaling.
- Filter Roulette: manual and automatic modes to randomly cycle filters.
- Snapshot / Download processed frames or images.

---

## Requirements

- Python 3.8+
- Packages:
  - `streamlit`
  - `streamlit-webrtc`
  - `opencv-python`
  - `numpy`
  - `Pillow`

You can install them with pip:

```bash
pip install streamlit streamlit-webrtc opencv-python numpy Pillow
```

> Note: On some systems `opencv-python` might require additional OS-level dependencies. See OpenCV or your OS package manager docs if you hit installation issues.

---

## Run locally

1. Save the provided script (the one you pasted) to a file, for example `app.py`.

2. From the same folder, run:

```bash
streamlit run app.py
```

3. A browser tab should automatically open at `http://localhost:8501/`. If it does not, open that URL in your browser manually.

### WebRTC notes (live webcam)
- Your browser will ask permission to use the camera. Allow it so the live webcam will work.
- `streamlit-webrtc` uses WebRTC, so you may need a secure context (HTTPS) for some browsers when not running locally. On `localhost` this is usually not required.
- The app includes a small `rtc_config` placeholder. If you plan to run the app over a remote server or through networks that need STUN/TURN servers, set `rtc_config` to a valid ICE server configuration. Example:
```py
rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
```

---

## App usage / UI overview

- **Input Mode** (sidebar): switch between **Live Webcam** and **Upload Image** modes.
- **Left panel**: shows the live video preview (or uploaded image) and processed result. In Live mode you can take snapshots and download them as PNG.
- **Right panel**: filter and transform settings.
  - Choose a filter from the dropdown (or use Filter Roulette).
  - Adjust `Rotate` and `Scale` using sliders or type precise values in the small boxes to the right of sliders.
  - Filter-specific controls appear when relevant (e.g., Sepia intensity, Canny thresholds, blur kernel).
  - Performance options: choose a requested camera resolution and enable **Performance mode** to reduce heavy parameters (fewer bilateral passes, lower sigma, etc.) for smoother FPS.
- **Filter Roulette**:
  - Manual: enable checkbox and press *Next random filter* to pick a random filter.
  - Auto: enable auto roulette and set interval (seconds) to cycle filters periodically.
  - Upload mode supports a timed roulette that can rerun the app with a new random filter every `n` seconds.

---

## Tips & Troubleshooting

- If the webcam preview doesn't start: check that your browser has camera permission for the page. Close other apps that might use the camera (Zoom, other browser tabs).
- If the FPS is low, enable **Performance mode** or pick a lower requested resolution (e.g., 640×480 or 320×240).
- If `streamlit-webrtc` video fails in a hosted environment, you may need to configure STUN/TURN servers with `rtc_config` and ensure inbound ports or websockets are allowed by your hosting provider.
- For faster development iterations, use `streamlit run --server.port 8501 app.py` (or change the port if 8501 is used).

---

## Project authors
- Jeremiah Daniel Regalario
- Isaiah John Mariano
- Meluisa Montealto

