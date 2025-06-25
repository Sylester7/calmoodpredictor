import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cat_mood_model.h5")

# Define class labels (based on your dataset folders)
labels = ['Angry', 'Happy', 'Relaxed', 'Sad']  # adjust if needed

# Image preprocessing
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.set_page_config(page_title="🐾 Cat Mood Predictor", layout="centered")
st.title("🐱 Cat Mood Predictor")
st.markdown("Upload a photo of your cat and see what mood it's in!")

uploaded_file = st.file_uploader("Choose a cat image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Cat", use_column_width=True)

    with st.spinner("Analyzing..."):
        pred = model.predict(preprocess(image))
        mood = labels[np.argmax(pred)]
        confidence = np.max(pred) * 100

    st.success(f"**Mood:** {mood} ({confidence:.2f}% confidence)")
