# Grass vs. Soyabean Image Classification

This project implements and compares several Convolutional Neural Network (CNN) architectures for the task of classifying images of **Grass** and **Soyabean**.

## Models

This repository contains scripts to train, evaluate, and compare multiple CNN models, including:
* Modern, pre-trained architectures (e.g., ResNet)
* Classic, foundational architectures (e.g., AlexNet, LeNet)

## Dataset Structure

The models expect the data to be in the following structure:

```
Final_Data/
   |
   |-- Grass/
   |   |-- ... (image files)
   |
   |-- Soyabean/
   |   |-- ... (image files)
```

## Installation

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow scikit-learn pandas matplotlib seaborn
    ```

## How to Run

This project contains several scripts for training, evaluation, and comparison.

* **Training:** Use a `train_*.py` script to train a new model (e.g., `train_alexnet.py`).
* **Evaluation:** Use an `evaluate_*.py` script to test a single model.
* **Comparison:** Use a `compare_*.py` script to generate comparison reports for multiple models.
