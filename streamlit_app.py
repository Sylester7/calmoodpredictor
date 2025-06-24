import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("cat_mood_model.h5")
labels = ['Angry', 'Happy', 'Relaxed', 'Sad']  # Adjust to your classes

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

st.set_page_config(page_title="Cat Mood Predictor 🐱", layout="centered")
st.title("😺 Cat Mood Predictor")
st.markdown("Upload a picture of a cat and let AI guess its mood!")

file = st.file_uploader("Upload a cat image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Your Cat 😼", use_column_width=True)

    pred = model.predict(preprocess(img))
    mood = labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"**Mood:** {mood} ({confidence:.2f}% confident)")
