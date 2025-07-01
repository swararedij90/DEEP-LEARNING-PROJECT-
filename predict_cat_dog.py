from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Load the trained model
model = load_model('output/cat_dog_cnn_model.h5')

# Path to the image you want to test
img_path = 'dataset/test_set/cats/cat.4001.jpg'  # 📝 Change this to your image

# Load and preprocess the image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)

# Output result
if prediction[0][0] >= 0.5:
    print("🐶 It's a Dog!")
else:
    print("🐱 It's a Cat!")
