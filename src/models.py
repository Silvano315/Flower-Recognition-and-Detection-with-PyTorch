import numpy as np
import os
import torch
import logging
import shutil
import sys
import json
import timm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Callable, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score


class Experiment:
    """
    A class to manage machine learning experiments, including logging, 
    saving/loading weights, and visualizing training history.
    """
    def __init__(self, name: str, root: str, logger=None):
        self.name = name
        self.root = os.path.join(root, name)
        self.logger = logger
        self.epoch = 1
        self.best_val_loss = sys.float_info.max
        self.best_val_loss_epoch = 1
        self.weights_dir = os.path.join(self.root, 'weights')
        self.history_dir = os.path.join(self.root, 'history')
        self.results_dir = os.path.join(self.root, 'results')
        self.latest_weights = os.path.join(self.weights_dir, 'latest_weights.pth')
        self.latest_optimizer = os.path.join(self.weights_dir, 'latest_optim.pth')
        self.best_weights_path = self.latest_weights
        self.best_optimizer_path = self.latest_optimizer
        self.train_history_fpath = os.path.join(self.history_dir, 'train.csv')
        self.val_history_fpath = os.path.join(self.history_dir, 'val.csv')
        self.test_history_fpath = os.path.join(self.history_dir, 'test.csv')
        self.metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
        self.history = {split: {metric: [] for metric in self.metrics} for split in ['train', 'val', 'test']}
        self.metrics.append('lr')
        self.history['lr'] = []

    def log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def init(self):
        self.log("Creating new experiment")
        self.init_dirs()
        self.init_history_files()

    def resume(self, model: torch.nn.Module, optim: torch.optim.Optimizer, weights_fpath: str = None, optim_path: str = None):
        self.log("Resuming existing experiment")
        if weights_fpath is None:
            weights_fpath = self.latest_weights
        if optim_path is None:
            optim_path = self.latest_optimizer

        model, state = self.load_weights(model, weights_fpath)
        optim = self.load_optimizer(optim, optim_path)

        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.epoch = state['last_epoch'] + 1
        self.load_history_from_file('train')
        self.load_history_from_file('val')

        return model, optim

    def init_dirs(self):
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def init_history_files(self):
        header = ','.join(['epoch'] + self.metrics) + '\n'
        for split in ['train', 'val', 'test']:
            fpath = getattr(self, f'{split}_history_fpath')
            with open(fpath, 'w') as f:
                f.write(header)

    def increment_epoch(self):
        self.epoch += 1

    def load_history_from_file(self, split: str):
        fpath = getattr(self, f'{split}_history_fpath')
        data = np.loadtxt(fpath, delimiter=',', skiprows=1)
        for i, metric in enumerate(self.metrics):
            self.history[split][metric] = data[:, i+1].tolist()

    def save_history(self, split: str, **kwargs):
        for metric, value in kwargs.items():
            metric_name = metric[4:] if metric.startswith('val_') else metric
            if metric_name not in self.history[split]:
                self.history[split][metric_name] = []
            self.history[split][metric_name].append(value)
        fpath = getattr(self, f'{split}_history_fpath')
        with open(fpath, 'a') as f:
            values = [str(kwargs.get(metric, kwargs.get(f'val_{metric}', ''))) for metric in self.metrics]
            f.write(f"{self.epoch},{','.join(values)}\n")
        if split == 'lr':
            self.history['lr'].append(kwargs['lr'])
        if split == 'val' and 'loss' in kwargs:
            if self.is_best_loss(kwargs['loss']):
                self.best_val_loss = kwargs['loss']
                self.best_val_loss_epoch = self.epoch

    def is_best_loss(self, loss: float) -> bool:
        return loss < self.best_val_loss

    def save_weights(self, model: torch.nn.Module, **kwargs):
        weights_fname = f"{self.name}-weights-{self.epoch}-" + "-".join([f"{v:.3f}" for v in kwargs.values()]) + ".pth"
        weights_fpath = os.path.join(self.weights_dir, weights_fname)
        try:
            torch.save({
                'last_epoch': self.epoch,
                'best_val_loss': self.best_val_loss,
                'best_val_loss_epoch': self.best_val_loss_epoch,
                'experiment': self.name,
                'state_dict': model.state_dict(),
                **kwargs
            }, weights_fpath)
            shutil.copyfile(weights_fpath, self.latest_weights)
            if self.is_best_loss(kwargs['val_loss']):
                self.best_weights_path = weights_fpath
            self.log(f"Successfully saved weights to {weights_fpath}")
        except Exception as e:
            self.log(f"Error saving weights: {str(e)}")
            raise

    def load_weights(self, model: torch.nn.Module, fpath: str):
        self.log(f"Loading weights from '{fpath}'")
        try:
            state = torch.load(fpath)
            model.load_state_dict(state['state_dict'])
            self.log(f"Loaded weights from experiment {self.name} (last_epoch {state['last_epoch']})")
            return model, state
        except FileNotFoundError:
            self.log(f"Error: Weights file not found at {fpath}")
            raise
        except RuntimeError as e:
            self.log(f"Error loading state dict: {str(e)}")
            raise

    def save_optimizer(self, optimizer: torch.optim.Optimizer, val_loss: float):
        optim_fname = f"{self.name}-optim-{self.epoch}.pth"
        optim_fpath = os.path.join(self.weights_dir, optim_fname)
        try:
            torch.save({
                'last_epoch': self.epoch,
                'experiment': self.name,
                'state_dict': optimizer.state_dict()
            }, optim_fpath)
            shutil.copyfile(optim_fpath, self.latest_optimizer)
            if self.is_best_loss(val_loss):
                self.best_optimizer_path = optim_fpath
            self.log(f"Successfully saved optimizer to {optim_fpath}")
        except Exception as e:
            self.log(f"Error saving optimizer: {str(e)}")
            raise

    def load_optimizer(self, optimizer: torch.optim.Optimizer, fpath: str):
        self.log(f"Loading optimizer from '{fpath}'")
        try:
            optim = torch.load(fpath)
            optimizer.load_state_dict(optim['state_dict'])
            self.log(f"Successfully loaded optimizer from session {optim['experiment']}, last_epoch {optim['last_epoch']}")
            return optimizer
        except FileNotFoundError:
            self.log(f"Error: Optimizer file not found at {fpath}")
            raise
        except Exception as e:
            self.log(f"Error loading optimizer: {str(e)}")
            raise

    def save_checkpoint(self, model, optimizer, epoch, logs):
        self.save_weights(model)
        self.save_optimizer(optimizer)

    def load_checkpoint(self, model, optimizer):
        model = self.load_weights(model)
        optimizer = self.load_optimizer(optimizer)
        return model, optimizer

    def get_state(self):
        return {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_loss_epoch': self.best_val_loss_epoch,
            'history': self.history
        }

    def set_state(self, state):
        self.epoch = state['epoch']
        self.best_val_loss = state['best_val_loss']
        self.best_val_loss_epoch = state['best_val_loss_epoch']
        self.history = state['history']

    def plot_history(self):
        for metric in self.metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            for split in ['train', 'val']:
                ax.plot(self.history[split][metric], label=split.capitalize())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.set_title(f'{self.name} - {metric.capitalize()}')
            plt.savefig(os.path.join(self.history_dir, f'{metric}.png'))
            plt.close()

        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 6*len(self.metrics)))
        for i, metric in enumerate(self.metrics):
            for split in ['train', 'val']:
                axes[i].plot(self.history[split][metric], label=split.capitalize())
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].set_title(f'{metric.capitalize()}')
        fig.suptitle(f'{self.name} - Training History')
        plt.tight_layout()
        plt.savefig(os.path.join(self.history_dir, 'combined_history.png'))
        plt.close()

        if 'lr' in self.history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.history['lr'], label='Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')  
            ax.legend()
            ax.set_title(f'{self.name} - Learning Rate')
            plt.savefig(os.path.join(self.history_dir, 'learning_rate.png'))
            plt.close()

    def update_plots(self):
        self.plot_history()

    def calculate_average_metrics(self, split: str, last_n_epochs: int = 5) -> Dict[str, float]:
        """
        Calculate average metrics for the last n epochs.

        Args:
            split (str): The data split to calculate metrics for ('train', 'val', or 'test').
            last_n_epochs (int): Number of last epochs to consider for averaging.

        Returns:
            Dict[str, float]: A dictionary of averaged metrics.
        """
        avg_metrics = {}
        for metric in self.metrics:
            values = self.history[split][metric][-last_n_epochs:]
            avg_metrics[metric] = sum(values) / len(values)
        return avg_metrics

    def export_results_to_json(self, filepath: str):
        """
        Export experiment results to a JSON file.

        Args:
            filepath (str): Path to save the JSON file.
        """
        results = {
            "name": self.name,
            "best_val_loss": self.best_val_loss,
            "best_val_loss_epoch": self.best_val_loss_epoch,
            "final_metrics": {
                split: self.calculate_average_metrics(split) 
                for split in ['train', 'val', 'test']
            },
            "history": self.history
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
            self.log(f"Successfully exported results to {filepath}")
        except Exception as e:
            self.log(f"Error exporting results to JSON: {str(e)}")
            raise

    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """
        Get the epoch with the best performance for a given metric.

        Args:
            metric (str): The metric to consider.
            mode (str): 'min' if lower is better, 'max' if higher is better.

        Returns:
            int: The epoch with the best performance.
        """
        values = self.history['val'][metric]
        if mode == 'min':
            best_value = min(values)
        elif mode == 'max':
            best_value = max(values)
        else:
            raise ValueError("Mode must be 'min' or 'max'")
        return values.index(best_value) + 1  

    def plot_learning_rate(self, lr_history: List[float]):
        """
        Plot the learning rate over epochs.

        Args:
            lr_history (List[float]): List of learning rates for each epoch.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(lr_history) + 1), lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'{self.name} - Learning Rate Schedule')
        plt.savefig(os.path.join(self.history_dir, 'learning_rate.png'))
        plt.close()


class Callback:
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        pass

class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0, patience: int = 0, 
                 verbose: bool = False, mode: str = 'auto'):
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.mode = mode
        self.monitor_op = None
        self._init_monitor_op()

    def _init_monitor_op(self):
        if self.mode not in ['auto', 'min', 'max']:
            print(f'EarlyStopping mode {self.mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'
        
        if self.mode == 'min' or (self.mode == 'auto' and 'loss' in self.monitor):
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> bool:
        current = logs.get(self.monitor)
        if current is None:
            print(f"Early stopping conditioned on metric `{self.monitor}` which is not available. "
                  f"Available metrics are: {','.join(list(logs.keys()))}")
            return False

        if self.best is None:
            self.best = current
            self.wait = 0
        elif self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f'Epoch {epoch}: early stopping')
                return True
        return False
    

class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = 'val_loss', verbose: int = 0,
                 save_best_only: bool = False, mode: str = 'auto'):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = None
        self.monitor_op = None
        self._init_monitor_op()

    def _init_monitor_op(self):
        if self.mode not in ['auto', 'min', 'max']:
            print(f'ModelCheckpoint mode {self.mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'

        if self.mode == 'min' or (self.mode == 'auto' and 'loss' in self.monitor):
            self.monitor_op = np.less
            self.best = float('inf')
        else:
            self.monitor_op = np.greater
            self.best = -float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: torch.nn.Module, 
                     optimizer: torch.optim.Optimizer, experiment: Any):
        current = logs.get(self.monitor)
        if current is None:
            print(f"Can't save best model, metric `{self.monitor}` is not available. "
                  f"Available metrics are: {','.join(list(logs.keys()))}")
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f'\nEpoch {epoch:05d}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, '
                          f'saving model to {self.filepath}')
                self.best = current
                self._save_checkpoint(model, optimizer, epoch, logs, experiment)
        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch:05d}: saving model to {self.filepath}')
            self._save_checkpoint(model, optimizer, epoch, logs, experiment)

    def _save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, logs: Dict[str, float], experiment: Any):
        experiment.save_checkpoint(model, optimizer, epoch, logs)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logs': logs,
            'best': self.best,
            'experiment_state': experiment.get_state() 
        }
        
        torch.save(checkpoint, self.filepath)

class ReduceLROnPlateau(Callback):
    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'min', factor: float = 0.1, 
                 patience: int = 10, verbose: bool = False, min_lr: float = 0, eps: float = 1e-8,
                 monitor: str = 'val_loss'):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.eps = eps
        self.monitor = monitor
        self.cooldown_counter = 0
        self.wait = 0
        self.best = None
        self.mode_worse = None
        self.is_better = None
        self._init_is_better(mode)

    def _init_is_better(self, mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
            self.is_better = lambda a, best: a < best - self.eps
        if mode == 'max':
            self.mode_worse = -float('inf')
            self.is_better = lambda a, best: a > best + self.eps

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            print(f"ReduceLROnPlateau conditioned on metric `{self.monitor}` which is not available. "
                  f"Available metrics are: {','.join(list(logs.keys()))}")
            return

        if self.best is None or self.is_better(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self._reduce_lr(epoch)
            self.wait = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.')


class TransferLearningModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, use_custom_classifier: bool = False):
        super(TransferLearningModel, self).__init__()
        
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        if hasattr(self.base_model, 'num_features'):
            num_ftrs = self.base_model.num_features
        elif hasattr(self.base_model, 'fc'):
            num_ftrs = self.base_model.fc.in_features
        else:
            raise ValueError(f"Unable to determine the number of features for the model {model_name}")
        
        self.use_custom_classifier = use_custom_classifier
        
        if use_custom_classifier:
            self.base_model.reset_classifier(0)
            
            self.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                F.relu(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                F.relu(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = self.base_model.get_classifier()

    def forward(self, x):
        features = self.base_model.forward_features(x)
        if self.use_custom_classifier:
            return self.classifier(features)
        else:
            return self.base_model.forward_head(features)
    
    
def freeze_layers(model: nn.Module, num_layers: int = -1):
    """
    Freeze layers of the model for transfer learning.
    
    Args:
    model (nn.Module): The model to freeze layers in.
    num_layers (int): Number of layers to freeze from the start. -1 means freeze all except the last layer.
    """
    if isinstance(model, TransferLearningModel):
        if num_layers == -1:
            for name, param in model.model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = False
        else:
            for i, (name, param) in enumerate(model.model.named_parameters()):
                if i < num_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    else:
        raise NotImplementedError("Freezing layers is only implemented for TransferLearningModel")

def create_model(num_classes: int, model_type: str = 'efficientnet_b0', pretrained: bool = True, use_custom_classifier: bool = False) -> nn.Module:
    """
    Create a model for transfer learning.
    
    Args:
    num_classes (int): Number of classes in the dataset.
    model_type (str): Type of model to create ('efficientnetv2_m', 'convnext_base' or 'resnet50').
    pretrained (bool): Whether to use pretrained weights.
    use_custom_classifier (bool): Whether to use a custom classifier or the model's original classifier.

    Returns:
    nn.Module: The created model.
    """
    if model_type == 'efficientnetv2_m':
        # Top performer
        return TransferLearningModel('efficientnetv2_m', num_classes, pretrained, use_custom_classifier)
    elif model_type == 'convnext_base':
        # Recent, medium performance
        return TransferLearningModel('convnext_base', num_classes, pretrained, use_custom_classifier)
    elif model_type == 'resnet50':
        # Classic
        return TransferLearningModel('resnet50', num_classes, pretrained, use_custom_classifier)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): The DataLoader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for updating model parameters.
        device (torch.device): The device to run the training on (CPU or GPU).

    Returns:
        Dict[str, float]: A dictionary containing the average loss and various metrics for the epoch.
    """
    model.train()
    running_loss = 0.0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = np.mean(np.array(predictions) == np.array(targets))
    epoch_precision = precision_score(targets, predictions, average='weighted', zero_division=1)
    epoch_recall = recall_score(targets, predictions, average='weighted')
    epoch_f1 = f1_score(targets, predictions, average='weighted')

    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1
    }

def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Dict[str, float]:
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): The neural network model to validate.
        dataloader (DataLoader): The DataLoader for the validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the validation on (CPU or GPU).

    Returns:
        Dict[str, float]: A dictionary containing the average loss and various metrics for the validation set.
    """
    model.eval()
    running_loss = 0.0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = np.mean(np.array(predictions) == np.array(targets))
    epoch_precision = precision_score(targets, predictions, average='weighted', zero_division=1)
    epoch_recall = recall_score(targets, predictions, average='weighted')
    epoch_f1 = f1_score(targets, predictions, average='weighted')

    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                experiment: Any, callbacks: List[Any], num_epochs: int,
                device: torch.device, logger: logging.Logger, 
                resume_from: str = None) -> nn.Module:
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        experiment (Any): An object to track the experiment (e.g., for logging).
        callbacks (List[Any]): A list of callback objects for various training events.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device to run the training on (CPU or GPU).
        logger (logging.Logger): Logger object for detailed logging.
        resume_from (str): If set, the checkpoint will load and resume training from where it left off.

    Returns:
        nn.Module: The trained model.
    """

    if resume_from:
        #model, optimizer = experiment.load_checkpoint(model, optimizer)
        #model, optimizer = experiment.resume(model, optimizer)
        #start_epoch = experiment.epoch
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        experiment.set_state(checkpoint['experiment_state'])
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_epoch = 1

    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Criterion: {criterion.__class__.__name__}")
    logger.info(f"Device: {device}")

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")

        current_lr = optimizer.param_groups[0]['lr']
        experiment.save_history('lr', lr=current_lr)

        train_logs = train_epoch(model, train_loader, criterion, optimizer, device)
        val_logs = validate(model, val_loader, criterion, device)

        val_logs_prefixed = {'val_' + k: v for k, v in val_logs.items()}
        logs = {**train_logs, **val_logs_prefixed}
        
        log_message = f"Epoch {epoch} - "
        log_message += " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        logger.info(log_message)

        experiment.save_history('train', **train_logs)
        experiment.save_history('val', **val_logs_prefixed)
        experiment.update_plots()

        experiment.increment_epoch()

        stop_training = False
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.on_epoch_end(epoch, logs, model, optimizer, experiment)
                logger.info(f"ModelCheckpoint: Saved model at epoch {epoch}")
            elif isinstance(callback, ReduceLROnPlateau):
                old_lr = optimizer.param_groups[0]['lr']
                callback.on_epoch_end(epoch, logs)
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"ReduceLROnPlateau: Learning rate changed from {old_lr} to {new_lr}")
            else:
                stop_training = callback.on_epoch_end(epoch, logs)
                if stop_training:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        if stop_training:
            break

    logger.info("Training completed")
    return model


def get_predictions(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from the model for the entire dataset.

    Args:
        model (torch.nn.Module): The trained model to use for predictions.
        dataloader (DataLoader): DataLoader containing the dataset to predict on.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - The first array contains the true labels.
            - The second array contains the predicted labels.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    return np.array(all_labels), np.array(all_preds)