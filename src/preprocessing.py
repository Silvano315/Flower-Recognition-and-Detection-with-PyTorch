import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Tuple, Dict

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']

def create_preprocessing_pipeline(
    img_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    augment: bool = False
) -> Transforms:
    """
    Create a preprocessing pipeline for image classification tasks.

    Args:
        img_size (Tuple[int, int]): Target image size (height, width). Default is (224, 224).
        mean (Tuple[float, float, float]): Mean values for normalization. Default is (0.5, 0.5, 0.5).
        std (Tuple[float, float, float]): Standard deviation values for normalization. Default is (0.5, 0.5, 0.5).
        augment (bool): Whether to apply data augmentation. Default is False.

    Returns:
        Transforms: A Transforms object wrapping the Albumentations composition.

    Example:
        transform = create_preprocessing_pipeline(augment=True)
        trainset = torchvision.datasets.ImageFolder(root='dataset/train', transform=transform)
    """
    transforms = [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
    ]

    if augment:
        transforms.extend([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.3), lightness=(0.5, 0.7)),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            ], p=0.2),
        ])

    transforms.append(ToTensorV2())

    return Transforms(A.Compose(transforms))


def create_datasets(data_dir: str, augment_train: bool = False):
    """
    Create train, validation, and test datasets.

    Args:
        data_dir (str): Root directory containing 'train', 'val', and 'test' subdirectories.
        augment_train (bool): Whether to apply data augmentation to the training set. Default is False.

    Returns:
        tuple: (trainset, valset, testset)

    Example:
        trainset, valset, testset = create_datasets('path/to/dataset', augment_train=True)
    """
    train_transform = create_preprocessing_pipeline(augment=augment_train)
    val_transform = create_preprocessing_pipeline(augment=False)

    trainset = torchvision.datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    valset = torchvision.datasets.ImageFolder(root=f'{data_dir}/valid', transform=val_transform)
    testset = torchvision.datasets.ImageFolder(root=f'{data_dir}/test', transform=val_transform)

    return trainset, valset, testset


def create_data_loaders(
    datasets: Dict[str, ImageFolder],
    batch_size: int,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test datasets.

    Args:
        datasets (Dict[str, ImageFolder]): A dictionary containing the datasets.
            Expected keys are 'train', 'val', and 'test'.
        batch_size (int): The batch size to use for the DataLoaders.
        num_workers (int, optional): Number of subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Default is 4.

    Returns:
        Dict[str, DataLoader]: A dictionary containing the DataLoader objects.
            Keys are 'train', 'val', and 'test'.

    Example:
        datasets = {
            'train': trainset,
            'val': valset,
            'test': testset
        }
        dataloaders = create_data_loaders(datasets, batch_size=32)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
    """
    dataloaders = {}

    for split, dataset in datasets.items():
        shuffle = split == 'train' 
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return dataloaders