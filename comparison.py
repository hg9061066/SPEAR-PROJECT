import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model

# --- 1. Configuration ---
TEST_DATA_DIR = r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\Final_Data"

MODEL_PATHS = {
    'ResNet50': r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\resnet50_fine_tuned_model.h5",
    'AlexNet': r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\alexnet_model.h5",
    'LeNet-5': r"C:\Users\white\OneDrive\Desktop\Coding\My Work\SPEAR Project\lenet_model.h5"
}

IMAGE_SIZES = {
    'ResNet50': (224, 224),
    'AlexNet': (227, 227),
    'LeNet-5': (32, 32)
}

CLASS_NAMES = ['Grass', 'Soyabean']
BATCH_SIZE = 32
# We remove AUTOTUNE here to process batches manually in the loop below

# --- 2. Helper Function for Data Loading ---
def get_test_data(image_size, needs_manual_norm=False):
    print(f"Loading test data at size {image_size}...")
    # FIX: shuffle=True ensures we get both Grass and Soyabean images
    test_ds = image_dataset_from_directory(
        TEST_DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        validation_split=0.3, 
        subset='validation',
        seed=101,             
        shuffle=True          # <--- CHANGED TO TRUE TO FIX EMPTY GRASS
    )
    
    if needs_manual_norm:
        print("Applying manual normalization (x/255.0)")
        def normalize_image(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        test_ds = test_ds.map(normalize_image)

    return test_ds

# --- 3. Run Comparisons ---
results = {}

for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        continue
        
    print(f"\nProcessing {model_name}...")
    model = load_model(model_path)
    
    # Load data specifically for this model
    is_resnet = (model_name == 'ResNet50')
    ds = get_test_data(IMAGE_SIZES[model_name], needs_manual_norm=is_resnet)
    
    # --- SAFELY EXTRACT LABELS AND PREDICTIONS ---
    # We iterate manually to ensure images and labels stay perfectly aligned
    y_true_all = []
    y_pred_all = []
    probs_all = []
    
    print("Generating predictions...")
    for images, labels in ds:
        # 1. Get True Labels for this batch
        batch_true = np.argmax(labels.numpy(), axis=1)
        y_true_all.extend(batch_true)
        
        # 2. Get Predictions for this batch
        batch_probs = model.predict(images, verbose=0)
        batch_pred = np.argmax(batch_probs, axis=1)
        
        y_pred_all.extend(batch_pred)
        probs_all.extend(batch_probs)
        
    # Convert lists to numpy arrays
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    probs = np.array(probs_all)
    
    # Store everything needed for plotting
    results[model_name] = {
        'y_true': y_true,
        'y_pred': y_pred,
        'probs': probs,
        'report': classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    }
    
    # Print Report to Console
    print(f"--- {model_name} Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# --- 4. Generate Comparative Plots ---
print("\nGenerating comparison plots...")
plt.style.use('ggplot')

# PLOT 1: Side-by-Side Confusion Matrices
try:
    plt.figure(figsize=(20, 6))
    colors = ['Blues', 'Greens', 'cividis']
    
    for i, (model_name, data) in enumerate(results.items()):
        plt.subplot(1, 3, i + 1)
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i],
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'{model_name}', fontsize=14)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('comparison_resnet_alex_lenet_cms.png')
    print("Saved: comparison_resnet_alex_lenet_cms.png")
    plt.close()
except Exception as e: print(f"Error Plot 1: {e}")

# PLOT 2: Combined ROC Curve
try:
    plt.figure(figsize=(10, 8))
    colors = {'ResNet50': 'blue', 'AlexNet': 'green', 'LeNet-5': 'darkorange'}
    
    for model_name, data in results.items():
        # Binary targets for ROC (Soyabean=1)
        y_true_bin = (data['y_true'] == 1).astype(int)
        y_prob = data['probs'][:, 1]
        
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors.get(model_name, 'black'), lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.savefig('comparison_resnet_alex_lenet_roc.png')
    print("Saved: comparison_resnet_alex_lenet_roc.png")
    plt.close()
except Exception as e: print(f"Error Plot 2: {e}")

# PLOT 3: Comparative Metrics Bar Chart
try:
    rows = []
    for model_name, data in results.items():
        report = data['report']
        for cls in CLASS_NAMES:
            rows.append({'Model': model_name, 'Class': cls, 'Metric': 'Precision', 'Score': report[cls]['precision']})
            rows.append({'Model': model_name, 'Class': cls, 'Metric': 'Recall', 'Score': report[cls]['recall']})
            rows.append({'Model': model_name, 'Class': cls, 'Metric': 'F1-Score', 'Score': report[cls]['f1-score']})

    df = pd.DataFrame(rows)
    
    g = sns.catplot(data=df, kind='bar', x='Metric', y='Score', hue='Model', col='Class',
                    palette=colors, height=5, aspect=0.9)
    
    # This sets the visual limit to 1.05 so you can see the top of the bars clearly
    g.set(ylim=(0, 1.05))
    
    plt.savefig('comparison_resnet_alex_lenet_metrics.png')
    print("Saved: comparison_resnet_alex_lenet_metrics.png")
    plt.close()
except Exception as e: print(f"Error Plot 3: {e}")

# --- 5. Print Summary Table ---
print("\n=== FINAL SUMMARY TABLE ===")
summary_rows = []
for model_name, data in results.items():
    rep = data['report']
    summary_rows.append({
        'Model': model_name,
        'Accuracy': rep['accuracy'] * 100,
        'F1 (Grass)': rep['Grass']['f1-score'],
        'F1 (Soyabean)': rep['Soyabean']['f1-score']
    })
print(pd.DataFrame(summary_rows).set_index('Model'))