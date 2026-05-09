import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.cm as cm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f0f8ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a56db 0%, #1e3a8a 100%);
}
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #bfdbfe !important;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1a56db 0%, #1e40af 100%);
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(26,86,219,0.3);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
}
.header-banner p {
    color: #bfdbfe;
    font-size: 1rem;
    margin: 8px 0 0 0;
}

/* Stat cards */
.stat-card {
    background: #ffffff;
    border: 2px solid #bfdbfe;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(26,86,219,0.1);
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1a56db;
}
.stat-card .stat-label {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 4px;
}

/* Upload section */
.upload-section {
    background: #ffffff;
    border: 2px dashed #1a56db;
    border-radius: 16px;
    padding: 20px;
    margin: 20px 0;
}

/* Tumor class cards */
.tumor-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

/* Section headers */
h2, h3 {
    color: #1a56db !important;
}

/* Logo text */
.logo-text {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
}
.logo-sub {
    font-size: 0.75rem;
    color: #bfdbfe;
    text-align: center;
}

/* Metric boxes */
.stMetric {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_INFO = {
    "Glioma": {"color": "🔴", "description": "Glioma is a tumor that occurs in the brain and spinal cord.", "risk": "High Risk Tumor"},
    "Meningioma": {"color": "🟠", "description": "Meningioma is a tumor that forms on membranes covering the brain.", "risk": "Moderate Risk"},
    "No Tumor": {"color": "🟢", "description": "No tumor detected. Brain appears healthy.", "risk": "Healthy Brain"},
    "Pituitary": {"color": "🟡", "description": "Pituitary tumor forms in the pituitary gland at brain base.", "risk": "Moderate Risk"}
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.keras", compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def get_gradcam(model, img_array):
    vgg16 = model.get_layer("vgg16")
    last_conv = vgg16.get_layer("block5_conv3")
    feature_extractor = tf.keras.Model(
        inputs=vgg16.input,
        outputs=[last_conv.output, vgg16.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, _ = feature_extractor(img_array)
        tape.watch(conv_outputs)
        x = model.get_layer("global_average_pooling2d")(conv_outputs)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x)
        x = model.get_layer("dense_1")(x)
        x = model.get_layer("dropout_1")(x)
        preds = model.get_layer("dense_2")(x)
        predicted_class = tf.argmax(preds[0])
        class_channel = preds[:, predicted_class]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()

def apply_gradcam(model, image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    heatmap, predictions = get_gradcam(model, img_array)
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_resized).resize((224, 224))
    heatmap_array = np.array(heatmap_img)
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_array / 255.0)
    heatmap_colored = np.uint8(heatmap_colored * 255)[:, :, :3]
    original = np.array(img_resized)
    superimposed = np.uint8(heatmap_colored * 0.4 + original * 0.6)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(np.max(predictions[0])) * 100
    return predicted_class, confidence, predictions[0], img_resized, heatmap_colored, superimposed

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">🧠 BrainAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Medical Imaging Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🖥️ Model Info")
    st.metric("Test Accuracy", "92.44%")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Best AUC", "0.994")
    with col_b:
        st.metric("MRI Scans", "7,200")

    st.markdown("---")
    st.markdown("### 🎯 Detectable Classes")
    st.markdown("""
    🔴 Glioma Tumor  
    🟠 Meningioma Tumor  
    🟢 No Tumor (Healthy)  
    🟡 Pituitary Tumor
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By")
    st.markdown("""
    **Muhammad Omar**  
    Roll No: 22-EE-126  
      
    **Ali Hassan**  
    Roll No: 22-EE-152
    """)

    st.markdown("---")
    st.markdown("### 👩‍🏫 Supervisor")
    st.markdown("""
    **Engr. Zainab Shahid**  
    Dept. of EE, UET Taxila
    """)

    st.markdown("---")
    st.warning("⚠️ For educational purposes only.\nNot a substitute for medical advice.")

# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🧠 Brain Tumor Detection System</h1>
    <p>AI-Powered MRI Classification with Grad-CAM Visualization | VGG16 Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-value">92.44%</div><div class="stat-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-value">0.994</div><div class="stat-label">Best AUC Score</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-value">7,200</div><div class="stat-label">Training Images</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-value">4</div><div class="stat-label">Tumor Classes</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────
with st.spinner("Loading AI Model..."):
    model = load_model()
st.success("✅ AI Model Loaded Successfully! Ready for Analysis.")

# ── UPLOAD ───────────────────────────────────────────────
st.markdown("### 📤 Upload MRI Brain Scan")
uploaded_file = st.file_uploader("Drag and drop or click to upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("🔍 Analyzing MRI with Grad-CAM..."):
        predicted_class, confidence, predictions, img_resized, heatmap, superimposed = apply_gradcam(model, image)

    info = CLASS_INFO[predicted_class]
    if predicted_class == "No Tumor":
        st.success(f"### {info['color']} {predicted_class} — Confidence: {confidence:.2f}%")
    else:
        st.error(f"### {info['color']} {predicted_class} Detected — Confidence: {confidence:.2f}%")
    st.markdown(f"_{info['description']}_")

    st.markdown("---")
    st.markdown("### 🖼️ Grad-CAM Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original MRI**")
        st.image(img_resized, use_container_width=True)
    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        st.image(heatmap, use_container_width=True)
    with col3:
        st.markdown("**Tumor Region Highlighted**")
        st.image(superimposed, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 All Class Probabilities")
    cols = st.columns(4)
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        with cols[i]:
            pct = float(prob) * 100
            info_cls = CLASS_INFO[cls]
            st.metric(label=f"{info_cls['color']} {cls}", value=f"{pct:.1f}%")
            st.progress(float(prob))

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:4rem;">🧠</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center; color:#1a56db;">Upload an MRI Scan to Begin</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#6b7280;">Our AI will analyze the scan and detect any brain tumors with Grad-CAM visualization</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("🔴", "Glioma", "High Risk Tumor", "#fee2e2", "#dc2626"),
        ("🟠", "Meningioma", "Moderate Risk", "#fff7ed", "#ea580c"),
        ("🟢", "No Tumor", "Healthy Brain", "#f0fdf4", "#16a34a"),
        ("🟡", "Pituitary", "Moderate Risk", "#fefce8", "#ca8a04"),
    ]
    for col, (emoji, name, risk, bg, color) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div style="background:{bg}; border-radius:12px; padding:20px; text-align:center; border: 1px solid {color}30;">
                <div style="font-size:2.5rem;">{emoji}</div>
                <div style="font-weight:700; color:{color}; font-size:1.1rem;">{name}</div>
                <div style="color:#6b7280; font-size:0.85rem;">{risk}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 How to use:")
    st.markdown("""
    1. Click **Browse files** above
    2. Select an MRI brain scan image
    3. Wait for AI analysis
    4. View prediction + Grad-CAM visualization!
    """)
