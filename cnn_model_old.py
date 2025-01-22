import os
import numpy as np
from medmnist import INFO, dataset as medmnist
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Directories for data and saved models
EPOCH = 30
IMAGE_SIZE = 28  # Image size
DATA_DIR = 'data'  # Directory where datasets are stored
MODEL_DIR = f'cnn_model_old_{EPOCH}'  # Directory to save trained models

# Function to compute class weights for handling class imbalance
def compute_class_weights(labels):
    labels_flat = np.argmax(labels, axis=1)  # Convert one-hot encoded labels to class indices
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_flat), y=labels_flat)
    class_weight_dict = dict(enumerate(class_weights))  # Convert to dictionary for easy use
    return class_weight_dict

# Data augmentation settings to improve model generalization
datagen = ImageDataGenerator(
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of width
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20% of height
    shear_range=0.2,  # Apply random shear transformations
    zoom_range=0.2,  # Randomly zoom in/out on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill missing pixels after transformations with nearest values
)

# Function to build a simple Convolutional Neural Network (CNN) model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # First convolutional layer
        MaxPooling2D((2, 2)),  # Max pooling to reduce spatial dimensions
        Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
        MaxPooling2D((2, 2)),  # Max pooling
        Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
        MaxPooling2D((2, 2)),  # Max pooling
        Flatten(),  # Flatten feature maps into a single vector
        Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
        Dropout(0.5),  # Dropout for regularization
        Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
    return model

# Dictionary to store evaluation metrics for all datasets
metrics_summary = {}

# Loop through each dataset file in the data directory
for file_name in os.listdir(DATA_DIR):
    dataset_name = file_name.split('.')[0]  # Extract dataset name from file name
    file_name = f'{dataset_name}.npz'  # Ensure the correct file extension
    file_path = os.path.join(DATA_DIR, file_name)  # Full path to the dataset file
    print(file_path)

    dataset = np.load(file_path)  # Load the dataset

    # # Print the shapes of the training data and labels
    # print(f"Shape of train images: ", dataset['train_images'].shape[0])
    # print(f"Shape of train labels: ", dataset['train_labels'].shape[0])

    # Extract training data and compute class weights
    train_images, train_labels = dataset['train_images'], dataset['train_labels']
    class_weights = compute_class_weights(train_labels)
    # print(f"Class weights: {class_weights}")

    # Ensure images have a channel dimension (e.g., grayscale to single channel)
    if len(train_images.shape) == 3:  # If grayscale images lack a channel dimension
        train_images = np.expand_dims(train_images, axis=-1)  # Add a single channel

    # Fit the data augmentation generator on the training data
    datagen.fit(train_images)

    # Define the input shape based on the dataset
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, train_images.shape[-1])  # Include channels

    num_classes = len(np.unique(train_labels))  # Determine the number of unique classes
    model = build_cnn_model(input_shape, num_classes)  # Build the CNN model

    # Prepare one-hot encoded labels for training
    train_labels_one_hot = to_categorical(train_labels)

    # Lists to store evaluation metrics for cross-validation
    accuracy_list = []
    auc_list = []

    # Use StratifiedKFold for cross-validation
    kf = StratifiedKFold(n_splits=5)  # 5-fold stratified splitting

    # Perform cross-validation
    for train_idx, val_idx in kf.split(train_images, train_labels):
        X_train, X_val = train_images[train_idx], train_images[val_idx]  # Split data
        y_train, y_val = train_labels_one_hot[train_idx], train_labels_one_hot[val_idx]  # Split labels
        
        # Train the model with data augmentation
        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=EPOCH,
                  validation_data=(X_val, y_val),
                  class_weight=class_weights)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)  # Get predicted probabilities
        
    # Convert validation labels (one-hot encoded) to integer format
    val_labels_int = np.argmax(y_val, axis=1)

    # Convert predictions to integer labels
    y_pred_int = np.argmax(y_pred, axis=1)

    # Compute F1 score
    f1 = f1_score(val_labels_int, y_pred_int, average='macro')

    metrics_summary[dataset_name] = {'f1_score': f1}

    print(f"{dataset_name}: F1 Score = {f1:.4f}")

    # Save the trained model
    os.makedirs(MODEL_DIR, exist_ok=True)  # Create model directory if not exists
    model_save_path = os.path.join(MODEL_DIR, f'{dataset_name}_model.keras')  # Path for saving the model
    model.save(model_save_path)  # Save the model
    print(f"Model saved to {model_save_path}")

# Print summary
print("Summary of Metrics:")
for dataset, metrics in metrics_summary.items():
    print(f"{dataset}: {metrics}")

"""
Summary of Metrics:
bloodmnist: {'f1_score': 0.9007754782360156}
breastmnist: {'f1_score': 0.8628930817610063}
dermamnist: {'f1_score': 0.33569046542102177}
octmnist: {'f1_score': 0.7037385184667349}
organamnist: {'f1_score': 0.8599807667038317}
organcmnist: {'f1_score': 0.8354755058801683}
organsmnist: {'f1_score': 0.56725210747802}
pathmnist: {'f1_score': 0.6658908111781361}
pneumoniamnist: {'f1_score': 0.9406812685532213}
retinamnist: {'f1_score': 0.3193035934601356}
tissuemnist: {'f1_score': 0.3846015253913705}

Summary of Metrics:
bloodmnist: {'f1_score': 0.8817072517036533}
breastmnist: {'f1_score': 0.782435129740519}
dermamnist: {'f1_score': 0.36840708679197215}
octmnist: {'f1_score': 0.6911853097411287}
organamnist: {'f1_score': 0.8523208582400329}
organcmnist: {'f1_score': 0.7573112577948737}
organsmnist: {'f1_score': 0.5548836449244691}
pathmnist: {'f1_score': 0.742697573695512}
pneumoniamnist: {'f1_score': 0.9322545902734583}
retinamnist: {'f1_score': 0.3791267696267696}
tissuemnist: {'f1_score': 0.3588439447786499}
"""