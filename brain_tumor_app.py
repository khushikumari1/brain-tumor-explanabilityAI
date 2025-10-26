# brain_tumor_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os
from datetime import datetime
import qrcode


# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

TARGET_SIZE = (150, 150)

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_brain_tumor_model(path="brain_tumor_model.h5"):
    model = load_model(path, compile=False)
    dummy = np.zeros((1, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    try:
        _ = model(dummy)
    except Exception:
        pass
    return model

model = load_brain_tumor_model()

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
FRIENDLY_LABELS = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}

# ------------------ UTILS ------------------
def preprocess_pil(img):
    arr = img.resize(TARGET_SIZE)
    arr = np.array(arr).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.expand_dims(arr, axis=0)

def predict(img_pil):
    x = preprocess_pil(img_pil)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx], preds, x

# ------------------ GRAD-CAM ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("No convolutional layer found.")
    if last_conv_layer_name not in [l.name for l in model.layers]:
        last_conv_layer_name = conv_layers[-1]

    new_input = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    x = new_input
    cloned_outputs = []

    for old_layer in model.layers:
        try:
            config = old_layer.get_config()
            LayerClass = old_layer.__class__
            new_layer = LayerClass.from_config(config)
            try:
                new_layer.build(x.shape)
                new_layer.set_weights(old_layer.get_weights())
            except Exception:
                pass
        except Exception:
            new_layer = old_layer
        try:
            if 'training' in new_layer.call.__code__.co_varnames:
                x = new_layer(x, training=False)
            else:
                x = new_layer(x)
        except Exception:
            x = new_layer(x)
        cloned_outputs.append(x)

    idx = [l.name for l in model.layers].index(last_conv_layer_name)
    cloned_conv_output = cloned_outputs[idx]
    grad_model = Model(inputs=new_input, outputs=[cloned_conv_output, x])

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        top_channel = predictions[:, pred_index]

    grads = tape.gradient(top_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.45):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

# ------------------ PDF REPORT ------------------


def generate_pdf(pred_label, conf, overlay_rgb, img_pil=None, preds=None):
    """
    Generates a professional PDF report with:
    - Cover page
    - Original MRI + Grad-CAM overlay side-by-side
    - Prediction and confidence table
    - Highlight predicted class
    - Notes section
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ------------------ Cover Page ------------------
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 200, "🧠 Brain Tumor Diagnosis Report")
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height - 230, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.showPage()  # new page

    # ------------------ Main Page ------------------
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Brain MRI Analysis")

    # Prediction Summary
    friendly = FRIENDLY_LABELS.get(pred_label, pred_label)
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Prediction: {friendly}")
    c.drawString(50, height - 100, f"Confidence: {conf*100:.2f}%")

    # Highlight predicted class if tumor detected
    if pred_label != "notumor":
        rect_y = height - 140  # place it below prediction & confidence
        rect_height = 20
        c.setFillColorRGB(1, 0, 0)  # red
        c.rect(45, rect_y, 500, rect_height, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)  # white text
        c.drawString(50, rect_y + 5, f"Tumor Detected: {friendly}")
        c.setFillColorRGB(0, 0, 0)  # reset to black
    # ------------------ Original + Grad-CAM ------------------
    if img_pil is not None:
        tmp_original = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img_pil.resize((400, 400)).save(tmp_original.name, "JPEG")
        tmp_overlay = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.fromarray(overlay_rgb).resize((400, 400)).save(tmp_overlay.name, "JPEG")

        # Draw side-by-side
        c.drawImage(ImageReader(tmp_original.name), 50, height - 500, width=200, height=200)
        c.drawImage(ImageReader(tmp_overlay.name), 270, height - 500, width=200, height=200)

        tmp_original.close()
        tmp_overlay.close()

    # ------------------ Confidence Table ------------------
    if preds is not None:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 520, "Prediction Confidence:")
        c.setFont("Helvetica", 12)
        y_pos = height - 540
        for cls, prob in zip(CLASS_NAMES, preds):
            c.drawString(50, y_pos, f"{FRIENDLY_LABELS[cls]}: {prob*100:.2f}%")
            y_pos -= 20

    # ------------------ Notes Section ------------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 100, "Notes / Recommendations:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 80, "This report is generated automatically by the AI model for analysis purposes.")
    c.drawString(50, 60, "For clinical decisions, please consult a certified radiologist or medical professional.")

    # ------------------ Optional QR / Report ID ------------------
    qr = qrcode.make(f"Brain Tumor Report: {datetime.now().strftime('%Y%m%d%H%M%S')}")
    tmp_qr = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    qr.save(tmp_qr.name)
    c.drawImage(ImageReader(tmp_qr.name), width - 120, 50, width=70, height=70)
    tmp_qr.close()

    # ------------------ Save PDF ------------------
    c.save()
    buffer.seek(0)
    return buffer


# ------------------ APP UI ------------------
st.markdown(
    "<h1 style='text-align:center; color:#1E88E5;'>🧠 Brain Tumor Classifier</h1>"
    "<p style='text-align:center; font-size:16px;'>Upload a brain MRI and get predictions with explainability.</p>",
    unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Grad-CAM", "Download Report"])

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])
analyze_btn = st.sidebar.button("🔍 Analyze MRI")

with st.sidebar.expander("ℹ️ How to use this app"):
    st.write("""
    1. Upload an MRI scan (jpg/jpeg/png).  
    2. Click 'Analyze MRI' to predict tumor type.  
    3. View Grad-CAM heatmap for model explainability.  
    4. Download PDF report for medical records.
    """)

# ------------------ Session State ------------------
for key in ["pred_label", "conf", "preds", "x", "heatmap", "overlay_rgb", "img_pil", "last_conv"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------ Run Prediction ------------------
if uploaded and analyze_btn:
    st.session_state.img_pil = Image.open(uploaded).convert("RGB")
    st.session_state.pred_label, st.session_state.conf, st.session_state.preds, st.session_state.x = predict(st.session_state.img_pil)
    convs = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    st.session_state.last_conv = convs[-1] if convs else None
    st.session_state.heatmap = make_gradcam_heatmap(st.session_state.x, model, st.session_state.last_conv)
    base = np.array(st.session_state.img_pil.resize(TARGET_SIZE))
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    st.session_state.overlay_rgb = cv2.cvtColor(overlay_heatmap(st.session_state.heatmap, base_bgr), cv2.COLOR_BGR2RGB)

# ------------------ Navigation Pages ------------------
if page == "Home":
    st.header("🩺 Prediction")
    if st.session_state.pred_label:
        friendly = FRIENDLY_LABELS[st.session_state.pred_label]
        st.image(st.session_state.img_pil, caption="Uploaded MRI", use_container_width=True)
        if st.session_state.pred_label == "notumor":
            st.success(f"No tumor detected — {friendly} ✅")
        else:
            st.error(f"Tumor detected — {friendly} ⚠️ Please consult a specialist.")

        # Prediction Confidence Plot
        fig = go.Figure(
            go.Bar(
                x=[FRIENDLY_LABELS[c] for c in CLASS_NAMES],
                y=st.session_state.preds*100,
                marker_color=["#43A047" if c==st.session_state.pred_label else "#B0BEC5" for c in CLASS_NAMES],
                text=[f"{p*100:.2f}%" for p in st.session_state.preds],
                textposition="outside"
            )
        )
        fig.update_layout(
            yaxis_title="Confidence (%)",
            xaxis_title="Tumor Type",
            template="plotly_white",
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload an MRI and click Analyze MRI to see predictions.")

elif page == "Grad-CAM":
    st.header("🔍 Grad-CAM")
    if st.session_state.overlay_rgb is not None:
        alpha = st.slider("Overlay Intensity", 0.1, 1.0, 0.45, 0.05)
        base = np.array(st.session_state.img_pil.resize(TARGET_SIZE))
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        overlay_bgr = overlay_heatmap(st.session_state.heatmap, base_bgr, alpha=alpha)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, caption="Grad-CAM Overlay", use_container_width=True)
        st.session_state.overlay_rgb = overlay_rgb
    else:
        st.info("Upload an MRI and click Analyze MRI to generate Grad-CAM.")

elif page == "Download Report":
    st.header("📄 Download Report")
    if st.session_state.overlay_rgb is not None:
        # <-- PLACE PDF GENERATION HERE
        pdf_buf = generate_pdf(
            st.session_state.pred_label,
            st.session_state.conf,
            st.session_state.overlay_rgb,
            img_pil=st.session_state.img_pil,
            preds=st.session_state.preds
        )

        st.download_button(
            label="Download PDF",
            data=pdf_buf,
            file_name="brain_tumor_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Grad-CAM not generated yet. Upload and analyze an MRI first.")

