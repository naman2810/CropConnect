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

# Class labels mapping with details
class_labels = {
    0: {
        "name": 'Tomato___Bacterial_spot',
        "cause": "Caused by the bacteria *Xanthomonas campestris*.",
        "symptoms": "Yellow or brown spots on leaves, stems, and fruit.",
        "effects": "Reduces fruit quality and plant growth."
    },
    1: {
        "name": 'Tomato___Early_blight',
        "cause": "Caused by the fungus *Alternaria solani*.",
        "symptoms": "Dark concentric rings on older leaves; yellowing and wilting.",
        "effects": "Reduces photosynthesis and yield."
    },
    2: {
        "name": 'Tomato___Late_blight',
        "cause": "Caused by the water mold *Phytophthora infestans*.",
        "symptoms": "Grayish, water-soaked spots on leaves; dark lesions on stems and fruit.",
        "effects": "Rapid plant death and highly destructive in humid conditions."
    },
    3: {
        "name": 'Tomato___Leaf_Mold',
        "cause": "Caused by the fungus *Passalora fulva*.",
        "symptoms": "Yellow spots on the upper side of leaves; velvety mold on the underside.",
        "effects": "Reduces photosynthesis and causes defoliation."
    },
    4: {
        "name": 'Tomato___Septoria_leaf_spot',
        "cause": "Caused by the fungus *Septoria lycopersici*.",
        "symptoms": "Small, circular spots with dark edges and light centers on leaves.",
        "effects": "Premature leaf drop, reducing fruit yield."
    },
    5: {
        "name": 'Tomato___Spider_mites Two-spotted_spider_mite',
        "cause": "Infestation by *Tetranychus urticae*.",
        "symptoms": "Yellow stippling on leaves; webbing on the plant.",
        "effects": "Weakened plants and reduced productivity."
    },
    6: {
        "name": 'Tomato___Target_Spot',
        "cause": "Caused by the fungus *Corynespora cassiicola*.",
        "symptoms": "Brown lesions with concentric rings on leaves, stems, and fruit.",
        "effects": "Leaf drop and poor fruit development."
    },
    7: {
        "name": 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        "cause": "Caused by a virus spread by whiteflies.",
        "symptoms": "Yellowing and curling of leaves; stunted plant growth.",
        "effects": "Severe yield loss and unmarketable fruit."
    },
    8: {
        "name": 'Tomato___Tomato_mosaic_virus',
        "cause": "Caused by a viral infection.",
        "symptoms": "Mottled patterns of light and dark green on leaves; deformed growth.",
        "effects": "Decreased fruit yield and plant vigor."
    },
    9: {
        "name": 'Tomato___healthy',
        "cause": "No disease detected.",
        "symptoms": "Healthy leaf with no visible spots or discoloration.",
        "effects": "Optimal growth and yield."
    }
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
st.write("Upload a tomato leaf image to identify the disease and learn more about its causes, symptoms, and effects.")

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
    st.write(f"**Prediction:** {prediction['name']}")
    st.write(f"**Cause:** {prediction['cause']}")
    st.write(f"**Symptoms:** {prediction['symptoms']}")
    st.write(f"**Effects:** {prediction['effects']}")
