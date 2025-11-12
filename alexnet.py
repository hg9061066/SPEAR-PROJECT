import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# --- 1. Configuration ---
DATA_DIR = r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\Final_Data"
# AlexNet traditionally uses 227x227 input
IMAGE_SIZE = (227, 227)
BATCH_SIZE = 32
CLASS_NAMES = ['Grass', 'Soyabean']
N_CLASSES = len(CLASS_NAMES)
VALIDATION_SPLIT = 0.3 # Using 30% for validation/testing

# --- 2. Load Data ---
print("Loading data...")

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
    seed=101 # Use a seed for reproducibility
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

# --- 3. Build the AlexNet Model ---
# This is the classic AlexNet architecture, built from scratch
print("Building AlexNet model...")
model = Sequential([
    Input(shape=IMAGE_SIZE + (3,)),
    
    # Add normalization (rescaling) layer
    Rescaling(1./255),

    # Layer 1
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

    # Layer 2
    Conv2D(256, (5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

    # Layer 3
    Conv2D(384, (3, 3), padding='same', activation='relu'),

    # Layer 4
    Conv2D(384, (3, 3), padding='same', activation='relu'),

    # Layer 5
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

    # Flatten before the dense layers
    Flatten(),

    # Layer 6 (Dense)
    Dense(4096, activation='relu'),
    Dropout(0.5),

    # Layer 7 (Dense)
    Dense(4096, activation='relu'),
    Dropout(0.5),

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
# Note: 10 epochs may be too few for AlexNet. 
# Increase this to 20 or 30 if your accuracy is low.
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=1, # You might need to increase this
    verbose=1
)
print("Training complete.")

# --- 5. Save the New Model ---
MODEL_SAVE_PATH = 'alexnet_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# --- 6. Plot Training History ---
print("Generating plots...")

plt.figure(figsize=(12, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('AlexNet Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('AlexNet Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('alexnet_training_history.png')
print("Saved training history plot to 'alexnet_training_history.png'")

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
print("\n--- Classification Report (AlexNet) ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("AlexNet Confusion Matrix")
plt.ylabel('Actual (True) Class')
plt.xlabel('Predicted Class')
plt.savefig('alexnet_confusion_matrix.png')
print("Saved confusion matrix to 'alexnet_confusion_matrix.png'")