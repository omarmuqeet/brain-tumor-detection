import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.cm as cm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_INFO = {
    "Glioma": {"color": "🔴", "description": "Glioma is a tumor that occurs in the brain and spinal cord."},
    "Meningioma": {"color": "🟠", "description": "Meningioma is a tumor that forms on membranes covering the brain."},
    "No Tumor": {"color": "🟢", "description": "No tumor detected. Brain appears healthy."},
    "Pituitary": {"color": "🟡", "description": "Pituitary tumor forms in the pituitary gland at brain base."}
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "brain_tumor_model.keras",
        compile=False
    )
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

# Header
st.title("🧠 Brain Tumor Detection")
st.markdown("### AI-Powered MRI Classification with Grad-CAM Visualization")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** VGG16 Transfer Learning
    **Accuracy:** 82.63%
    **Feature:** Grad-CAM Visualization
    
    **Classes:**
    - 🔴 Glioma
    - 🟠 Meningioma
    - 🟢 No Tumor
    - 🟡 Pituitary
    """)
    st.warning("⚠️ For educational purposes only.")

# Load model
with st.spinner("Loading AI Model..."):
    model = load_model()
st.success("✅ AI Model loaded successfully!")

# Upload
st.markdown("### 📤 Upload an MRI Image")
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

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
    st.info("👆 Please upload an MRI brain scan image to get started!")
    col1, col2 = st.columns(2)
    with col1:
        st.error("🔴 Glioma Tumor")
        st.warning("🟠 Meningioma Tumor")
    with col2:
        st.success("🟢 No Tumor")
        st.warning("🟡 Pituitary Tumor")
    st.markdown("### 💡 How to use:")
    st.markdown("""
    1. Click **Browse files** above
    2. Select an MRI brain scan image
    3. Wait for AI analysis
    4. View prediction + Grad-CAM visualization!
    """)