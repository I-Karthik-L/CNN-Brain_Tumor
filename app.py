import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# =============================================================
# Config — must match how the model was trained
# =============================================================
HF_REPO_ID = "karthikkkkl/brain-tumor-resnet50"
HF_FILENAME = "brain_tumor_resnet50.keras"

IMG_SIZE = (128, 128)          # must match training image size
CLASS_NAMES = ["yes", "no"]    # index 0 -> yes (tumor), index 1 -> no (no tumor)
CONFIDENCE_WARNING_THRESHOLD = 60.0   # below this, show a low-confidence warning

# =============================================================
# Page setup
# =============================================================
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI scan and this model will predict whether it shows a tumor. "
    "Built with a ResNet50 transfer-learning model, fine-tuned on a brain MRI dataset."
)

st.warning(
    "⚠️ This is a student/learning project trained on a small dataset. "
    "It is **not** a medical diagnostic tool and should not be used for real "
    "medical decisions. Always consult a qualified medical professional.",
    icon="⚠️",
)

# =============================================================
# Load model (cached so it only downloads/loads once per session)
# =============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    # safe_mode=False is required because the model contains a Lambda layer
    # (wrapping ResNet50's preprocess_input). Keras 3 blocks deserializing
    # arbitrary functions inside Lambda layers by default as a security
    # measure. This is safe here because we trained and uploaded this model
    # ourselves -- never set safe_mode=False for a model file from an
    # untrusted source.
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    return model


model = load_model()

# =============================================================
# Preprocessing — must mirror training exactly
# =============================================================
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert an uploaded PIL image into the exact format the model expects.
    NOTE: pixel values are kept in the 0-255 range on purpose — the model's
    first layers already include ResNet50's preprocess_input (mean subtraction),
    so we must NOT divide by 255 here, or the input distribution would be wrong.
    """
    img = pil_image.convert("RGB")               # ensure 3 channels (handles grayscale/RGBA uploads)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)  # shape: (128, 128, 3), values 0-255
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)
    return img_array


# =============================================================
# Main UI
# =============================================================
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_image, caption="Uploaded scan", use_container_width=True)

    with st.spinner("Analyzing scan..."):
        input_tensor = preprocess_image(pil_image)
        predictions = model.predict(input_tensor, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx]) * 100

    with col2:
        st.subheader("Result")

        if predicted_label == "yes":
            st.error(f"**Tumor detected**\n\nConfidence: {confidence:.1f}%")
        else:
            st.success(f"**No tumor detected**\n\nConfidence: {confidence:.1f}%")

        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            st.warning(
                "The model isn't very confident about this prediction — "
                "the image may be unclear, unusual, or not a typical brain MRI scan.",
                icon="❓",
            )

    st.divider()
    st.subheader("Class probabilities")
    prob_dict = {label: float(prob) for label, prob in zip(CLASS_NAMES, predictions)}
    st.bar_chart(prob_dict)

else:
    st.info("👆 Upload a brain MRI image (JPG or PNG) to get a prediction.")

st.divider()
st.caption(
    "Model: ResNet50 transfer learning (fine-tuned) — trained with data augmentation "
    "and a 3-stage learning progression (basic CNN → augmentation → transfer learning). "
    "Test accuracy: 88%."
)
