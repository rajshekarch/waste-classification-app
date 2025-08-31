import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model once
@st.cache_resource
def load_my_model():
    return load_model("models/best_model_EfficientNetB0.keras")

model = load_my_model()

class_names = [
    '1-Cardboard', '2-Food Organics', '3-Glass', '4-Metal',
    '5-Miscellaneous Trash', '6-Paper', '7-Plastic',
    '8-Textile Trash', '9-Vegetation'
]

img_size = (224, 224)

def preprocess_image(img: Image.Image):
    img = img.convert('RGB').resize(img_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("♻️ Waste Classification Demo")
st.write("Upload an image and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)
    label = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
