import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import image_dataset_from_directory
import pandas as pd

# --- 1. Configuration: YOU MUST UPDATE THESE ---

# Path to the model you want to evaluate
MODEL_PATH = r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\resnet50_fine_tuned_model.h5"

# Path to your TEST dataset folder
TEST_DATA_DIR = r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\Final_Data" 

# Model and class details (must match your training)
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Grass', 'Soyabean']
BATCH_SIZE = 32
# -------------------------------------------------

# --- 2. Load the Model ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. Load the Test Data ---
if not os.path.exists(TEST_DATA_DIR):
    print(f"Error: Test data directory not found at {TEST_DATA_DIR}")
    exit()

print(f"Loading test data from: {TEST_DATA_DIR}")
test_dataset = image_dataset_from_directory(
    TEST_DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

# --- 4. Normalize the Data ---
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

print("Normalizing test data...")
test_dataset = test_dataset.map(normalize_image)

# --- 5. Run Evaluation ---
print("\n--- Keras 'model.evaluate()' ---")
loss, accuracy = model.evaluate(test_dataset)
print(f"Overall Test Loss:     {loss:.4f}")
print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

# --- 6. Get Predictions for Detailed Report ---
print("\n--- Generating Detailed Report and Plots ---")
y_true = []
y_pred_probs = []

for images, labels in test_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1)) 
    y_pred_probs.extend(model.predict(images, verbose=0))

y_pred = np.argmax(y_pred_probs, axis=1)

# --- 7. Print Classification Report ---
print("\n--- Detailed Classification Report (Scikit-learn) ---")
try:
    # Get report as text
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report_text)
    
    # Get report as a dictionary for plotting
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

except ValueError as e:
    print(f"\nError generating report: {e}")
    exit()


# --- 8. NEW: PLOT CONFUSION MATRIX ---
print("\nGenerating Confusion Matrix plot...")
try:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual (True) Class')
    plt.xlabel('Predicted Class')
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    print(f"Saved confusion matrix to confusion_matrix.png")
    plt.close() # Close the plot to free memory

except Exception as e:
    print(f"Error plotting confusion matrix: {e}")


# --- 9. NEW: PLOT METRICS BAR CHART ---
print("\nGenerating Metrics Bar Chart...")
try:
    # Convert the report dictionary to a pandas DataFrame and plot
    # We drop 'accuracy', 'macro avg', and 'weighted avg' to only plot the classes
    report_df = pd.DataFrame(report_dict).transpose()
    report_df_classes = report_df.loc[CLASS_NAMES]

    report_df_classes.plot(kind='bar', y=['precision', 'recall', 'f1-score'], figsize=(12, 8))
    plt.title('Metrics per Class (Precision, Recall, F1-Score)')
    plt.ylabel('Score')
    plt.xticks(rotation=0) # Keep class names horizontal
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('metrics_barchart.png')
    print(f"Saved metrics bar chart to metrics_barchart.png")
    plt.close() # Close the plot

except Exception as e:
    print(f"Error plotting metrics bar chart: {e}")


print("\n--- Evaluation and Visualization Complete ---")