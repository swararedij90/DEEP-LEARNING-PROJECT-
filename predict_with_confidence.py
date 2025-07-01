from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('output/cat_dog_cnn_model.h5')

# 👇 Change this path to any test image you want to predict
img_path = 'dataset/test_set/cats/cat.4001.jpg'  # Example: use a cat image

# Load and preprocess the image
img = image.load_img(img_path, target_size=(64, 64))  # same size as training
img_array = image.img_to_array(img) / 255.0  # normalize
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Make prediction
prediction = model.predict(img_array)[0][0]

# Interpret prediction and calculate confidence
if prediction >= 0.5:
    label = "Dog 🐶"
    confidence = prediction * 100
else:
    label = "Cat 🐱"
    confidence = (1 - prediction) * 100

# Output result
print(f"✅ Prediction: {label}")
print(f"🎯 Confidence: {confidence:.2f}%")
