import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Download model from Google Drive if not present
MODEL_URL = "https://drive.google.com/uc?id=1Y6UfzGLlSByE4mkKIupXy9GYlpePztkg"
MODEL_PATH = "cat_mood_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)

model = tf.keras.models.load_model(MODEL_PATH)
labels = ['Angry', 'Happy', 'Relaxed', 'Sad']  # Update based on your dataset

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

st.set_page_config(page_title="🐱 Cat Mood Predictor", layout="centered")
st.title("🐾 Cat Mood Predictor")
st.markdown("Upload a picture of your cat and let AI guess its mood!")

uploaded_file = st.file_uploader("Upload a cat image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Cat", use_column_width=True)
    with st.spinner("Predicting mood..."):
        pred = model.predict(preprocess(image))
        mood = labels[np.argmax(pred)]
        confidence = np.max(pred) * 100
    st.success(f"**Mood:** {mood} ({confidence:.2f}% confident)")
