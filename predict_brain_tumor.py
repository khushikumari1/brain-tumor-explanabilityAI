import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('brain_tumor_model.h5')

# Class names in the same order as training
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img).astype('float32')  # Fixed line
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Predicted Class: {predicted_class} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    img_path = input("Enter the full path of the image you want to predict: ").strip()
    img = Image.open(img_path).convert('RGB')  # Added convert('RGB') just in case
    predict_image(img)
