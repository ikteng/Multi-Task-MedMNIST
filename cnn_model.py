import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

EPOCH = 30
IMAGE_SIZE = 28
DATA_DIR = "data"  # Directory containing datasets
MODEL_DIR = f"cnn_model_{EPOCH}"  # Directory to save models
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure directory exists

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

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),  # Replaces deeper pooling layers
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
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

    train_images, train_labels = dataset['train_images'], dataset['train_labels']
    class_weights = compute_class_weights(train_labels)

    if len(train_images.shape) == 3:
        train_images = np.expand_dims(train_images, axis=-1)

    datagen.fit(train_images)

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, train_images.shape[-1])
    num_classes = len(np.unique(train_labels))
    model = build_cnn_model(input_shape, num_classes)

    train_labels_one_hot = to_categorical(train_labels)

    accuracy_list = []
    auc_list = []

    kf = StratifiedKFold(n_splits=5)

    fold = 1
    best_f1_score = -1
    best_model_weights = None  # Store the best model's weights

    for train_idx, val_idx in kf.split(train_images, train_labels):
        X_train, X_val = train_images[train_idx], train_images[val_idx]
        y_train, y_val = train_labels_one_hot[train_idx], train_labels_one_hot[val_idx]

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
        checkpoint_path = os.path.join(MODEL_DIR, f'folds/{dataset_name}_fold{fold}_best_model.keras')
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=EPOCH,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[early_stopping, model_checkpoint]
        )

        model.load_weights(checkpoint_path)

        y_pred = model.predict(X_val)
        val_labels_int = np.argmax(y_val, axis=1)
        y_pred_int = np.argmax(y_pred, axis=1)

        fold_f1_score = f1_score(val_labels_int, y_pred_int, average='macro')

        if fold_f1_score > best_f1_score:
            best_f1_score = fold_f1_score
            best_model_weights = model.get_weights()  # Save the best model weights

        fold += 1

    # Save the best model for this dataset
    final_model_path = os.path.join(MODEL_DIR, f'{dataset_name}_model.keras')
    model.set_weights(best_model_weights)  # Load the best model weights
    save_model(model, final_model_path)  # Save the entire model

    metrics_summary[dataset_name] = {'f1_score': best_f1_score}
    print(f"{dataset_name}: Best F1 Score = {best_f1_score:.4f}")

# Print summary of metrics
print("Summary of Metrics:")
for dataset, metrics in metrics_summary.items():
    print(f"{dataset}: {metrics}")

"""
Summary of Metrics:
bloodmnist: {'f1_score': 0.8985248500759161}
breastmnist: {'f1_score': 0.8552107233425914}
dermamnist: {'f1_score': 0.3875112592294125}
octmnist: {'f1_score': 0.6617194373553434}
organamnist: {'f1_score': 0.8329979044586387}
organcmnist: {'f1_score': 0.7717250964941486}
organsmnist: {'f1_score': 0.5513458603439065}
pathmnist: {'f1_score': 0.8160376146421119}
pneumoniamnist: {'f1_score': 0.9550229182582124}
retinamnist: {'f1_score': 0.3988642827345178}
tissuemnist: {'f1_score': 0.3662021755969528}

Summary of Metrics:
bloodmnist: {'f1_score': 0.9421913296388192}
breastmnist: {'f1_score': 0.8288854003139718}
dermamnist: {'f1_score': 0.5369639851830669}
octmnist: {'f1_score': 0.7822873345837125}
organamnist: {'f1_score': 0.9361176419390129}
organcmnist: {'f1_score': 0.8943919217665262}
organsmnist: {'f1_score': 0.7496667080082801}
pathmnist: {'f1_score': 0.8775386055864117}
pneumoniamnist: {'f1_score': 0.9562453270362885}
retinamnist: {'f1_score': 0.41987245579867877}
tissuemnist: {'f1_score': 0.5119450641966621}
"""

