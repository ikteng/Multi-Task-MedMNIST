import numpy as np
import os
import matplotlib.pyplot as plt

DATA_DIR = 'data'

total_test = 0

# Function to print dataset statistics
def print_statistics(dataset):
    # Display the keys available in the dataset
    print(f"Available keys in dataset: {dataset.keys()}")

    # Print the size of the training set if it exists in the dataset
    if 'train_images' in dataset and 'train_labels' in dataset:
        print(f"Train size: {dataset['train_images'].shape[0]}")
        print(f"Train image dimensions: {dataset['train_images'].shape[1:]}")  # Shape of a single image
    
    # Print the size of the validation set if it exists in the dataset
    if 'val_images' in dataset and 'val_labels' in dataset:
        print(f"Val size: {dataset['val_images'].shape[0]}")
        print(f"Val image dimensions: {dataset['val_images'].shape[1:]}")  # Shape of a single image
    
    # Print the size of the test set if it exists in the dataset
    if 'test_images' in dataset and 'test_labels' in dataset:
        print(f"Test size: {dataset['test_images'].shape[0]}")
        print(f"Test image dimensions: {dataset['test_images'].shape[1:]}")  # Shape of a single image
        
        # Update the global total_test counter with the size of this dataset's test set
        global total_test
        total_test += dataset['test_images'].shape[0]

    # Calculate and display class distribution for the training set
    if 'train_labels' in dataset:
        train_labels = dataset['train_labels']
        
        # Get unique class labels and their respective counts
        unique, counts = np.unique(train_labels, return_counts=True)
        
        # Convert counts to percentages
        counts = counts / train_labels.shape[0] * 100
        
        # Display the number of unique classes and their respective distributions
        print(f"Classes: {len(unique)}")
        print('Labels:', *unique)
        print('Class distribution', *[f'\n\t{i} -> {count:.2f}%' for i, count in zip(unique, counts)])

# Function to plot label distribution
def plot_label_distribution(dataset_name):
    # Load the dataset
    dataset = np.load(os.path.join(DATA_DIR, dataset_name))
    
    # Set the plot title
    plt.title(dataset_name)
    
    # Check if the dataset contains training labels
    if 'train_labels' in dataset:
        train_labels = dataset['train_labels']
        
        # Plot a histogram of the training labels
        plt.hist(train_labels, bins=np.max(train_labels) + 1)
        plt.xlabel("Class Label")
        plt.ylabel("Frequency")
        plt.show()

# Loop through all files in the dataset directory and process each one
for file_name in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file_name)  # Full path to the dataset file
    print(file_path)  # Print the path of the current dataset being processed
    
    # Load the .npz file containing the dataset
    dataset = np.load(file_path)
    
    # Print statistics about the dataset
    print_statistics(dataset)
    
    # Plot the label distribution for the dataset's training set
    plot_label_distribution(file_name)

# Print the total number of test samples across all datasets
print(f"\nTotal test samples: {total_test}")

"""
Output:

data\bloodmnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\bloodmnist.npz' with keys: train_images, train_labels, val_images, val_labels, test_images...)
Train size: 11959
Train image dimensions: (28, 28, 3)
Val size: 1712
Val image dimensions: (28, 28, 3)
Test size: 3421
Test image dimensions: (28, 28, 3)
Classes: 8
Labels: 0 1 2 3 4 5 6 7
Class distribution 
        0 -> 7.12% 
        1 -> 18.24% 
        2 -> 9.07% 
        3 -> 16.94% 
        4 -> 7.10% 
        5 -> 8.30% 
        6 -> 19.48% 
        7 -> 13.74%
data\breastmnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\breastmnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)
Train size: 546
Train image dimensions: (28, 28)
Val size: 78
Val image dimensions: (28, 28)
Test size: 156
Test image dimensions: (28, 28)
Classes: 2
Labels: 0 1
Class distribution 
        0 -> 26.92%
        1 -> 73.08%
data\dermamnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\dermamnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)     
Train size: 7007
Train image dimensions: (28, 28, 3)
Val size: 1003
Val image dimensions: (28, 28, 3)
Test size: 2005
Test image dimensions: (28, 28, 3)
Classes: 7
Labels: 0 1 2 3 4 5 6
Class distribution
        0 -> 3.25%
        1 -> 5.12%
        2 -> 10.97%
        3 -> 1.14%
        4 -> 11.12%
        5 -> 66.98%
        6 -> 1.41%
data\octmnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\octmnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)       
Train size: 97477
Train image dimensions: (28, 28)
Val size: 10832
Val image dimensions: (28, 28)
Test size: 1000
Test image dimensions: (28, 28)
Classes: 4
Labels: 0 1 2 3
Class distribution
        0 -> 34.35%
        1 -> 10.48%
        2 -> 7.95%
        3 -> 47.22%
data\organamnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\organamnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)    
Train size: 34581
Train image dimensions: (28, 28)
Val size: 6491
Val image dimensions: (28, 28)
Test size: 17778
Test image dimensions: (28, 28)
Classes: 11
Labels: 0 1 2 3 4 5 6 7 8 9 10
Class distribution
        0 -> 5.66%
        1 -> 4.07%
        2 -> 3.93%
        3 -> 4.26%
        4 -> 11.46%
        5 -> 11.04%
        6 -> 17.82%
        7 -> 11.33%
        8 -> 11.36%
        9 -> 8.76%
        10 -> 10.30%
data\organcmnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\organcmnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)    
Train size: 13000
Train image dimensions: (28, 28)
Val size: 2392
Val image dimensions: (28, 28)
Test size: 8268
Test image dimensions: (28, 28)
Classes: 11
Labels: 0 1 2 3 4 5 6 7 8 9 10
Class distribution
        0 -> 8.87%
        1 -> 4.82%
        2 -> 4.68%
        3 -> 4.62%
        4 -> 8.37%
        5 -> 9.00%
        6 -> 22.97%
        7 -> 7.71%
        8 -> 7.86%
        9 -> 9.02%
        10 -> 12.09%
data\organsmnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\organsmnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)    
Train size: 13940
Train image dimensions: (28, 28)
Val size: 2452
Val image dimensions: (28, 28)
Test size: 8829
Test image dimensions: (28, 28)
Classes: 11
Labels: 0 1 2 3 4 5 6 7 8 9 10
Class distribution
        0 -> 8.24%
        1 -> 4.57%
        2 -> 4.41%
        3 -> 5.17%
        4 -> 8.12%
        5 -> 8.03%
        6 -> 24.85%
        7 -> 5.32%
        8 -> 5.76%
        9 -> 14.38%
        10 -> 11.16%
data\pathmnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\pathmnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)      
Train size: 89996
Train image dimensions: (28, 28, 3)
Val size: 10004
Val image dimensions: (28, 28, 3)
Test size: 7180
Test image dimensions: (28, 28, 3)
Classes: 9
Labels: 0 1 2 3 4 5 6 7 8
Class distribution
        0 -> 10.41%
        1 -> 10.57%
        2 -> 11.51%
        3 -> 11.56%
        4 -> 8.90%
        5 -> 13.54%
        6 -> 8.76%
        7 -> 10.45%
        8 -> 14.32%
data\pneumoniamnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\pneumoniamnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...) 
Train size: 4708
Train image dimensions: (28, 28)
Val size: 524
Val image dimensions: (28, 28)
Test size: 624
Test image dimensions: (28, 28)
Classes: 2
Labels: 0 1
Class distribution
        0 -> 25.79%
        1 -> 74.21%
data\retinamnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\retinamnist.npz' with keys: train_images, val_images, test_images, train_labels, val_labels...)    
Train size: 1080
Train image dimensions: (28, 28, 3)
Val size: 120
Val image dimensions: (28, 28, 3)
Test size: 400
Test image dimensions: (28, 28, 3)
Classes: 5
Labels: 0 1 2 3 4
Class distribution
        0 -> 45.00%
        1 -> 11.85%
        2 -> 19.07%
        3 -> 17.96%
        4 -> 6.11%
data\tissuemnist.npz
Available keys in dataset: KeysView(NpzFile 'data\\tissuemnist.npz' with keys: train_images, train_labels, val_images, val_labels, test_images...)    
Train size: 165466
Train image dimensions: (28, 28)
Val size: 23640
Val image dimensions: (28, 28)
Test size: 47280
Test image dimensions: (28, 28)
Classes: 8
Labels: 0 1 2 3 4 5 6 7
Class distribution
        0 -> 32.08%
        1 -> 4.72%
        2 -> 3.55%
        3 -> 9.31%
        4 -> 7.12%
        5 -> 4.66%
        6 -> 23.69%
        7 -> 14.87%

Total test samples: 96941
"""