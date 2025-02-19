import os
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from sklearn.metrics import f1_score

EPOCH = 30
DATA_DIR = 'data'  # Directory containing the dataset files
MODEL_DIR = f'cnn_model_old_{EPOCH}'  # Directory containing the trained models
SUBMISSION_FILE = f'submission_old_{EPOCH}.csv'

# Function to generate predictions and calculate F1 scores for all tasks
def evaluate_and_predict(models, datasets, dataset_names):
    predictions = []
    f1_scores = {}

    global_id = 0  # Sequential ID to track the overall image index
    for model, dataset, dataset_name in zip(models, datasets, dataset_names):
        test_images = dataset['test_images']  # Load test images
        test_labels = dataset['test_labels']  # Load test labels (ground truth)

        # Ensure images have the correct shape (add channel dimension for grayscale images)
        if len(test_images.shape) == 3:  # Grayscale images without channel dimension
            test_images = np.expand_dims(test_images, axis=-1)

        # Predict probabilities for each class
        y_pred_probs = model.predict(test_images)

        # Convert probabilities to class labels
        y_pred_labels = np.argmax(y_pred_probs, axis=1)

        # Compute the macro F1 score for the current task
        f1 = f1_score(test_labels, y_pred_labels, average='macro')
        f1_scores[dataset_name] = f1  # Store the F1 score

        # Add predictions to the list
        for idx, label in enumerate(y_pred_labels):
            predictions.append({
                'id_image_in_task': idx,  # Image index within the current task
                'task_name': dataset_name,  # Dataset name (task name)
                'label': label  # Predicted class label
            })
            global_id += 1  # Increment global ID

    return predictions, f1_scores

# Function to compute the harmonic mean of F1 scores
def compute_harmonic_mean(f1_scores):
    n = len(f1_scores)  # Number of tasks
    harmonic_mean = n / sum(1 / score for score in f1_scores.values())  # Harmonic mean formula
    return harmonic_mean

# Function to generate the submission file
def generate_submission_file(predictions, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['id', 'id_image_in_task', 'task_name', 'label'])

        # Write the prediction rows
        for idx, prediction in enumerate(predictions):
            writer.writerow([
                idx,  # Global sequential ID
                prediction['id_image_in_task'],  # Image index within the task
                prediction['task_name'],  # Task name
                prediction['label']  # Predicted label
            ])

    print(f"Submission file saved to {output_file}")

# Load pre-trained models and corresponding datasets
models = []  # List to store loaded models
datasets = []  # List to store datasets
dataset_names = []  # List to store dataset names

# Iterate through all dataset files
for file_name in os.listdir(DATA_DIR):
    dataset_name = file_name.split('.')[0]  # Extract dataset name (without file extension)
    file_path = os.path.join(DATA_DIR, file_name)  # Full path to the dataset file
    print("File path:", file_path)

    dataset = np.load(file_path)  # Load the dataset

    # # Display dataset details
    # print(f"Dataset: {dataset_name}, Test Images: {dataset['test_images'].shape[0]}")
    # print("Test Images:", dataset['test_images'].shape)
    # print("Test Labels:", dataset['test_labels'].shape)

    # Load the corresponding trained model
    model_path = os.path.join(MODEL_DIR, f'{dataset_name}_model.keras')
    model = tf.keras.models.load_model(model_path)

    # Append model, dataset, and dataset name to the respective lists
    models.append(model)
    datasets.append(dataset)
    dataset_names.append(dataset_name)

# Generate predictions and calculate F1 scores for all datasets
predictions, f1_scores = evaluate_and_predict(models, datasets, dataset_names)

# Calculate total number of images across all datasets
total_images = sum(dataset['test_images'].shape[0] for dataset in datasets)
# print(f"Total images across all tasks: {total_images}")

# Validate that the total number of rows matches the predictions generated
total_rows = len(predictions)
# print(f"Total predictions generated: {total_rows}")

# Save the predictions to a CSV file
generate_submission_file(predictions, SUBMISSION_FILE)

# Calculate the harmonic mean of F1 scores across all tasks
harmonic_mean = compute_harmonic_mean(f1_scores)

# Print F1 scores and harmonic mean
print("\nF1 Scores:")
for dataset_name, f1 in f1_scores.items():
    print(f"{dataset_name}: {f1:.4f}")

print(f"\nHarmonic Mean of F1 Scores: {harmonic_mean:.4f}")


"""

Submission file saved to submission_old_30.csv

F1 Scores:
bloodmnist: 0.8695
breastmnist: 0.6969
dermamnist: 0.3844
octmnist: 0.6527
organamnist: 0.7435
organcmnist: 0.6794
organsmnist: 0.4997
pathmnist: 0.7005
pneumoniamnist: 0.8999
retinamnist: 0.3051
tissuemnist: 0.3662

Harmonic Mean of F1 Scores: 0.5477

--------------------------------------------------
Submission file saved to submission_old_50.csv

F1 Scores:
bloodmnist: 0.8807
breastmnist: 0.7520
dermamnist: 0.3995
octmnist: 0.6076
organamnist: 0.7648
organcmnist: 0.7452
organsmnist: 0.5417
pathmnist: 0.6742
pneumoniamnist: 0.8846
retinamnist: 0.3281
tissuemnist: 0.3572

Harmonic Mean of F1 Scores: 0.5621

"""

"""
Submission file saved to submission_cnn_30.csv

F1 Scores:
bloodmnist: 0.9381
breastmnist: 0.7116
dermamnist: 0.4717
octmnist: 0.7477
organamnist: 0.8354
organcmnist: 0.8023
organsmnist: 0.6371
pathmnist: 0.6643
pneumoniamnist: 0.8496
retinamnist: 0.3297
tissuemnist: 0.4990

Harmonic Mean of F1 Scores: 0.6228
"""