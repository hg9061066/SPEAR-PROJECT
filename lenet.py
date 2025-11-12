import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# --- 1. Configuration ---
DATA_DIR = r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\Final_Data"
# LeNet-5 is designed for very small images, typically 32x32 or 28x28.
IMAGE_SIZE = (32, 32) 
BATCH_SIZE = 32
CLASS_NAMES = ['Grass', 'Soyabean']
N_CLASSES = len(CLASS_NAMES)
VALIDATION_SPLIT = 0.3 # Using 30% for validation/testing

# --- 2. Load Data ---
print(f"Loading data and resizing to {IMAGE_SIZE}...")

# Training dataset
train_ds = image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=101 
)

# Validation/Test dataset
val_ds = image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=101
)

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Data loaded successfully.")

# --- 3. Build the LeNet-5 Model ---
print("Building LeNet-5 model...")
model = Sequential([
    Input(shape=IMAGE_SIZE + (3,)),
    
    # LeNet traditionally used 'tanh' activation, which
    # expects inputs scaled from -1 to 1.
    Rescaling(1./127.5, offset=-1),

    # Layer C1: Convolution
    Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh'),
    
    # Layer S2: Subsampling (Average Pooling)
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Layer C3: Convolution
    Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh'),

    # Layer S4: Subsampling (Average Pooling)
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Flatten before the dense layers
    Flatten(),

    # Layer C5: Dense (Fully Connected)
    Dense(120, activation='tanh'),

    # Layer F6: Dense (Fully Connected)
    Dense(84, activation='tanh'),

    # Output Layer
    Dense(N_CLASSES, activation='softmax') # 2 classes: Grass, Soyabean
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model summary
model.summary()

# --- 4. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=10, 
    verbose=1
)
print("Training complete.")

# --- 5. Save the New Model ---
MODEL_SAVE_PATH = 'lenet_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# --- 6. Plot Training History ---
print("Generating plots...")
plt.figure(figsize=(12, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('LeNet-5 Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LeNet-5 Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('lenet_training_history.png')
print("Saved training history plot to 'lenet_training_history.png'")

# --- 7. Evaluate the Model ---
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Get predictions
y_true = []
for images, labels in val_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\n--- Classification Report (LeNet-5) ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("LeNet-5 Confusion Matrix")
plt.ylabel('Actual (True) Class')
plt.xlabel('Predicted Class')
plt.savefig('lenet_confusion_matrix.png')
print("Saved confusion matrix to 'lenet_confusion_matrix.png'")