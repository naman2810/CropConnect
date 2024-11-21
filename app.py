import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
from PIL import Image  # Add this import

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tomato_leaf_disease_model.h5')

model = load_model()

# Class labels mapping based on your provided class_labels
class_labels = {
    0: 'Tomato___Bacterial_spot',
    1: 'Tomato___Early_blight',
    2: 'Tomato___Late_blight',
    3: 'Tomato___Leaf_Mold',
    4: 'Tomato___Septoria_leaf_spot',
    5: 'Tomato___Spider_mites Two-spotted_spider_mite',
    6: 'Tomato___Target_Spot',
    7: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    8: 'Tomato___Tomato_mosaic_virus',
    9: 'Tomato___healthy'
}

# Helper function to prepare an image for prediction
def prepare_image(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255.0  # Normalize pixel values
    resized_array = cv2.resize(img_array, (150, 150))  # Resize to model input size
    return resized_array.reshape(-1, 150, 150, 3)

# Helper function to predict the class
def predict_image(image_path):
    image = prepare_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

# Streamlit App
st.title("Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to identify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)  # Use PIL to open the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Save the uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the class
    prediction = predict_image("temp.jpg")
    st.write(f"**Prediction:** {prediction}")
