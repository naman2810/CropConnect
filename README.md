### 🍅 Tomato Leaf Disease Detection using VGG19
This repository contains a deep learning project to classify tomato leaf images into various disease categories or healthy leaves using a VGG19 model for transfer learning. The goal is to assist farmers and agricultural experts in identifying tomato plant diseases quickly and accurately.

🌟 Try the live app here: [CropConnect - Tomato Disease Detection](https://cropconnect.streamlit.app/)
### 📋 Project Overview
Tomato plants are susceptible to numerous diseases that can significantly affect yield and quality. This project leverages deep learning techniques to automate the detection of common tomato plant diseases using images of leaves. The model was built using the pre-trained VGG19 architecture and fine-tuned for this specific task.

The application is hosted online to provide easy accessibility for users to upload leaf images and get instant predictions along with details about the diseases.

### 🗂️ Dataset
The dataset includes tomato leaf images categorized into 10 classes (9 diseases and healthy leaves). It consists of:

Training data: Used to train the model.
Validation data: Used to fine-tune the model during training.
Testing data: Used to evaluate model performance.
Path Structure:
train/: Contains folders for each class of tomato leaf disease.
valid/: Contains folders for each class for validation purposes.
