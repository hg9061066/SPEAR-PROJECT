import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep # Import the sleep function from the time module

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW) # Set pin 8 to be an output pin and set initial value to low (off)
GPIO.setup(10, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(16, GPIO.OUT, initial=GPIO.LOW)

# Load the saved model
model_path = "resnet50_fine_tuned_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Optional recompilation
print("Model loaded successfully!")

# Preprocess the input image
def preprocess_image(image_path, target_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
        
    img = load_img(image_path, target_size=target_size)  # Load and resize image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Perform inference
def predict(image_path):
    input_size = (224, 224)  # ResNet50 expects 224x224 input
    input_data = preprocess_image(image_path, input_size)

    predictions = model.predict(input_data, verbose=0)  # Suppress progress bar
    return predictions

# Test the model with a sample image
image_path = "/home/pi/Desktop/Healthy/images (1).jpeg"
try:
    predictions = predict(image_path)

    # Check the model's output shape
    print(f"Raw predictions: {predictions}")
    print(f"Predictions shape: {predictions.shape}")

    # Update class_labels with the correct number of classes
    class_labels = ['Grass', 'Soyabean']  # Replace with actual class names

    if len(predictions[0]) != len(class_labels):
        raise ValueError("Mismatch between model output and class labels.")

    predicted_class = class_labels[np.argmax(predictions)]
    print(f"Predicted class: {predicted_class}")
except Exception as e:
    print(f"Error: {e}")
