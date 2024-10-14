import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from PIL import Image
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import torch
import torchvision

from src.utils import calculate_mean_std


def display_random_images(image_paths: List[str], n: int = 25) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grid of random images from the dataset using matplotlib.

    Args:
        image_paths (List[str]): List of paths to all images in the dataset.
        n (int): Number of images to display. Default is 25.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the Figure and Axes objects.

    Example:
        fig, axes = display_random_images(filepaths, n=25)
        plt.show()  # To display the plot
        # or
        fig.savefig('random_images.png')  # To save the plot
    """
    random_images = random.sample(image_paths, n)
    grid_size = int(n**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    for i, img_path in enumerate(random_images):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1} from {img_path.split('/')[2]}\nLabel: {img_path.split('/')[3]}")
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig, axes


def visualize_class_samples(image_paths: List[str], num_samples: int = 3, max_classes: int = 10) -> plt.Figure:
    """
    Visualize sample images for each class in the dataset using matplotlib.

    Args:
        image_paths (List[str]): List of paths to all images in the dataset.
        num_samples (int): Number of sample images to display for each class. Default is 3.
        max_classes (int): Maximum number of classes to display. Default is 10.

    Returns:
        plt.Figure: A matplotlib Figure object containing the sample images for each class.

    Example:
        fig = visualize_class_samples(train_paths, num_samples=3, max_classes=10)
        plt.show()
    """
    class_images: Dict[str, List[str]] = {}
    for path in image_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_images:
            class_images[class_name] = []
        class_images[class_name].append(path)
    sorted_classes = sorted(class_images.items(), key=lambda x: len(x[1]), reverse=True)[:max_classes]

    fig, axes = plt.subplots(max_classes, num_samples, figsize=(num_samples * 3, max_classes * 3))
    fig.suptitle("Images Sample for Classes", fontsize=16, verticalalignment = 'top', y = 1.0)
    for i, (class_name, paths) in enumerate(sorted_classes):
        sample_paths = random.sample(paths, min(num_samples, len(paths)))
        for j, img_path in enumerate(sample_paths):
            img = Image.open(img_path)
            ax = axes[i, j] if max_classes > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(class_name, fontsize=10)
    for i in range(len(sorted_classes), max_classes):
        for j in range(num_samples):
            fig.delaxes(axes[i, j] if max_classes > 1 else axes[j])
    plt.tight_layout()
    return fig


def plot_class_distribution(image_paths: List[str]) -> go.Figure:
    """
    Plot the distribution of classes in the dataset using Plotly.

    Args:
        image_paths (List[str]): List of paths to all images in the dataset.

    Returns:
        go.Figure: A Plotly Figure object containing the class distribution plot.

    Example:
        fig = plot_class_distribution(train_paths)
        fig.show()
    """
    classes = [os.path.basename(os.path.dirname(path)) for path in image_paths]    
    class_counts = Counter(classes)    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)    
    class_names, counts = zip(*sorted_classes)
    total_images = sum(counts)
    
    fig = go.Figure(data=[go.Bar(x=class_names, y=counts)])    
    fig.update_layout(
        title=f'Class distribution in {image_paths[0].split("/")[2].capitalize()} dataset (Total images: {total_images})',
        xaxis_title='Classes',
        yaxis_title='Number of Images',
        xaxis_tickangle=-45
    )
    return fig


def compare_class_distribution(dataset_paths: Dict[str, Tuple[List[str], List[str]]]) -> go.Figure:
    """
    Compare the distribution of classes across train, test, and validation sets using Plotly.

    Args:
        dataset_paths (Dict[str, Tuple[List[str], List[str]]]): A dictionary containing file paths for each dataset split.

    Returns:
        go.Figure: A Plotly Figure object containing the comparative class distribution plot.

    Example:
        fig = compare_class_distribution(dataset_paths)
        fig.show()
    """
    split_counts = {}
    all_classes = set()
    for split, (paths, _) in dataset_paths.items():
        classes = [os.path.basename(os.path.dirname(path)) for path in paths]
        split_counts[split] = Counter(classes)
        all_classes.update(classes)
    all_classes = sorted(all_classes)

    fig = go.Figure()
    for split, counts in split_counts.items():
        fig.add_trace(go.Bar(
            name=split.capitalize(),
            x=all_classes,
            y=[counts.get(cls, 0) for cls in all_classes],
            text=[counts.get(cls, 0) for cls in all_classes],
            textposition='auto'
        ))
    fig.update_layout(
        title='Comparison of the Class Distribution between Train, Test e Validation',
        xaxis_title='Classes',
        yaxis_title='Number of Images',
        barmode='group',
        xaxis_tickangle=-45,
        legend_title='Dataset Split'
    )

    return fig


def analyze_image_dimensions(image_paths: List[str]) -> go.Figure:
    """
    Analyze and plot the distribution of image dimensions in the dataset using Plotly.

    Args:
        image_paths (List[str]): List of paths to all images in the dataset.

    Returns:
        go.Figure: A Plotly Figure object containing the image dimension analysis plots.

    Example:
        fig = analyze_image_dimensions(train_paths)
        fig.show()
    """
    widths = []
    heights = []
    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Images Dimensions', 'Widths Distribution', 
                                        'Heights Distributions'))
    fig.add_trace(go.Scatter(x=widths, y=heights, mode='markers', marker=dict(opacity=0.5), name = "Width vs Height"),
                   row=1, col=1)
    fig.add_trace(go.Histogram(x=widths, name = "Width"), row=1, col=2)
    fig.add_trace(go.Histogram(x=heights, name = "Height"), row=2, col=1)    
    fig.update_layout(height=800, width=1000, title_text="Images Dimensions Analysis")
    fig.update_xaxes(title_text="Width (pixel)", row=1, col=1)
    fig.update_yaxes(title_text="Height (pixel)", row=1, col=1)
    fig.update_xaxes(title_text="Width (pixel)", row=1, col=2)
    fig.update_xaxes(title_text="Height (pixel)", row=2, col=1)
    
    return fig


def analyze_color_distribution(image_paths: List[str], n_samples: int = 1000) -> go.Figure:
    """
    Analyze and plot the distribution of colors in the dataset using Plotly.

    Args:
        image_paths (List[str]): List of paths to all images in the dataset.
        n_samples (int): Number of images to sample for analysis. Default is 1000.

    Returns:
        go.Figure: A Plotly Figure object containing the color distribution plots.

    Example:
        fig = analyze_color_distribution(train_paths, n_samples=1000)
        fig.show()
    """
    sampled_paths = np.random.choice(image_paths, min(n_samples, len(image_paths)), replace=False)
    r_values = []
    g_values = []
    b_values = []
    for path in sampled_paths:
        with Image.open(path) as img:
            img_array = np.array(img)
            r_values.extend(img_array[:,:,0].flatten())
            g_values.extend(img_array[:,:,1].flatten())
            b_values.extend(img_array[:,:,2].flatten())
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Red Distribution', 'Green Distribution', 'Blue Distribution'))    
    fig.add_trace(go.Histogram(x=r_values, marker_color='red', name = 'Red'), row=1, col=1)
    fig.add_trace(go.Histogram(x=g_values, marker_color='green', name = 'Green'), row=1, col=2)
    fig.add_trace(go.Histogram(x=b_values, marker_color='blue', name = 'Blue'), row=1, col=3)
    fig.update_layout(height=500, width=1200, title_text="Colour Distribution Analysis")
    fig.update_xaxes(title_text="Pixel Value")
    fig.update_yaxes(title_text="Frequencies")
    
    return fig


def visualize_random_images(dataset: torchvision.datasets.ImageFolder, num_images: int = 25, axis: bool = False) -> None:
    """
    Visualize a specified number of random images from an ImageFolder dataset.

    Args:
        dataset (torchvision.datasets.ImageFolder): The dataset to visualize images from.
        num_images (int): Number of random images to display. Default is 25.
        axis (bool): if True axis would be shown. Dafault is False

    Returns:
        None: This function displays the plot directly.

    Example:
        visualize_random_images(trainset, num_images=25)
    """
    grid_size = int(num_images ** 0.5)
    if grid_size ** 2 < num_images:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f"Random Images from {dataset.root.split('//')[1].capitalize()} Dataset", fontsize=16, y = 1.01)
    axes = axes.flatten()
    indices = random.sample(range(len(dataset)), num_images)
    for i, idx in enumerate(indices):
        img, label = dataset[idx]        
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose((1, 2, 0))
        mean = dataset.transform.transforms.transforms[-2].mean
        std = dataset.transform.transforms.transforms[-2].std
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        if axis == False:
            axes[i].axis('off')
        axes[i].set_title(f"Class: {dataset.classes[label]}")

    for i in range(num_images, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()


def visualize_augmented_images(dataset: torchvision.datasets.ImageFolder, num_images: int = 5) -> None:
    """
    Visualize a specified number of random images from an ImageFolder dataset,
    showing both the original and augmented versions side by side.

    Args:
        dataset (torchvision.datasets.ImageFolder): The dataset to visualize images from.
        num_images (int): Number of random images to display. Default is 5.

    Returns:
        None: This function displays the plot directly.

    Example:
        visualize_augmented_images(trainset_aug, num_images=5)
    """
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 4*num_images))
    fig.suptitle("Original vs Augmented Images", fontsize=16, y = 1.0)

    indices = random.sample(range(len(dataset)), num_images)
    normalize_transform = None
    for transform in dataset.transform.transforms.transforms:
        if isinstance(transform, torchvision.transforms.Normalize):
            normalize_transform = transform
            break
    for i, idx in enumerate(indices):
        img_path, label = dataset.samples[idx]
        original_img = Image.open(img_path).convert('RGB')
        augmented_img, _ = dataset[idx]
        if isinstance(augmented_img, torch.Tensor):
            augmented_img = augmented_img.numpy().transpose((1, 2, 0))
        if normalize_transform:
            mean = np.array(normalize_transform.mean)
            std = np.array(normalize_transform.std)
            augmented_img = std * augmented_img + mean
        augmented_img = np.clip(augmented_img, 0, 1)
        axes[i, 0].imshow(original_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Original - Class: {dataset.classes[label]}")
        axes[i, 1].imshow(augmented_img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Augmented")

    plt.tight_layout()
    plt.show()


def scatter_plot_metrics(train_csv_path: str, val_csv_path: str) -> None:
    """
    Plot chosen metrics from training and validation CSV files using Plotly.
    
    Args:
        train_csv_path (str): Path to the CSV file containing training metrics.
        val_csv_path (str): Path to the CSV file containing validation metrics.
    
    Returns:
        None: Displays an interactive scatter plot of the training and validation metrics.
    """
    
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    epochs = train_df['epoch'].tolist()
    
    fig = go.Figure()
    dropdown_buttons = []
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    for i, metric in enumerate(metrics):
        train_mean, train_std = calculate_mean_std(train_df[metric])
        val_mean, val_std = calculate_mean_std(val_df[metric])
        
        fig.add_trace(go.Scatter(
            x=epochs, y=train_df[metric], mode='lines+markers', visible=(i == 0),
            name=f'Train {metric.capitalize()} (Mean: {train_mean:.2f}, Std: {train_std:.2f})'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_df[metric], mode='lines+markers', visible=(i == 0),
            name=f'Val {metric.capitalize()} (Mean: {val_mean:.2f}, Std: {val_std:.2f})'
        ))
        
        dropdown_buttons.append({
            'label': metric.capitalize(),
            'method': 'update',
            'args': [
                {'visible': [False] * len(metrics) * 2},
                {'title': f'{metric.capitalize()}'}
            ]
        })
        dropdown_buttons[-1]['args'][0]['visible'][i * 2] = True
        dropdown_buttons[-1]['args'][0]['visible'][i * 2 + 1] = True
    
    fig.update_layout(
        title='Model Performance per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Metric Value',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
        }],
        legend_title="Legend"
    )
    
    fig.show()



def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> None:
    """
    Plot and save a confusion matrix.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        classes (List[str]): List of class names.

    Returns:
        None: This function saves the plot as a file and doesn't return anything.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.close()