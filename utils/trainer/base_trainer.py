"""
Base trainer class with common functionality
"""
import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from collections import Counter
import numpy as np

from ..metrics_logger import MetricsLogger
from ..timer import Timer
from ..training_logger import TrainingLogger
from ..adv_metrics import AdversarialMetrics


def _init_history():
    return {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'duration': [],
        'true_labels': [],
        'predictions': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_predictions': [],
        'val_targets': [],
        # Adversarial metrics
        'adv_loss': [],
        'adv_accuracy': [],
        'adv_predictions': [],
        'adv_targets': [],
        # Detailed metrics
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_metrics': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_per_class_metrics': []
    }


class BaseTrainer:
    """Base trainer class with common functionality for all training types"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, model_name, task_name, dataset_name,
                 device, config, scheduler=None, model_depth=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.model_name = model_name
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.device = device
        self.config = config
        self.model_depth = model_depth

        # Training parameters
        self.has_trained = False
        self.epochs = getattr(config, 'epochs', 100)
        self.lambda_l2 = getattr(config, 'lambda_l2', 1e-4)
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self.error_if_nonfinite = False

        # Best metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_balanced_acc = 0.0
        self.no_improvement_count = 0

        # Initialize components
        self.scaler = GradScaler()
        self.timer = Timer()
        self.training_logger = TrainingLogger()
        self.history = _init_history()
        self.adv_metrics = AdversarialMetrics()

        # Move model to device
        self.model.to(self.device)

        # Initialize MetricsLogger
        self.tb_logger = MetricsLogger(
            task_name, dataset_name, model_name, use_tensorboard=False)

        # Log hyperparameters
        self._log_hyperparameters()

        logging.info(f"BaseTrainer initialized for {model_name}")

    def _log_hyperparameters(self):
        """Log hyperparameters to MetricsLogger"""
        hparams = {
            'learning_rate': getattr(self.config, 'lr', 0.001),
            'batch_size': getattr(self.config, 'train_batch', 32),
            'optimizer': getattr(self.config, 'optimizer', 'adam'),
            'model': self.model_name,
            'loss_type': getattr(self.config, 'loss_type', 'standard'),
            'weight_decay': getattr(self.config, 'weight_decay', 1e-4),
            'dropout': getattr(self.config, 'drop', 0.0),
            'scheduler': getattr(self.config, 'scheduler', 'none')
        }
        self.tb_logger.log_hparams(hparams)

    def _get_save_directory(self):
        """Get the proper save directory structure"""
        if self.model_depth is not None:
            model_name_with_depth = f"{self.model_name}_{self.model_depth}"
        else:
            model_name_with_depth = self.model_name

        save_dir = f"out/{self.task_name}/{self.dataset_name}/{model_name_with_depth}/save_model"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir, model_name_with_depth

    def save_model(self, filename=None):
        """Save the model state dict"""
        try:
            save_dir, model_name_with_depth = self._get_save_directory()

            if filename is None:
                filename = f"best_{model_name_with_depth}_{self.dataset_name}.pth"

            save_path = os.path.join(save_dir, filename)
            torch.save(self.model.state_dict(), save_path)
            logging.info(f"Model saved to {save_path}")
            return save_path
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return None

    def train(self, patience):
        """Main training loop - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement the train method")

    def validate(self):
        """Validation method - to be implemented by subclasses"""
        raise NotImplementedError(
            "Subclasses must implement the validate method")

    def _update_scheduler(self, val_loss, val_acc, early_stopping_metric='loss'):
        """Update learning rate scheduler"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_loss if early_stopping_metric == 'loss' else val_acc
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            # Update current learning rate
            if hasattr(self, 'optimizer'):
                self.current_lr = self.optimizer.param_groups[0]['lr']

    def _check_improvement(self, val_loss, val_acc, early_stopping_metric='loss'):
        """Check if validation metrics improved"""
        improved = False

        if early_stopping_metric == 'loss':
            if val_loss < self.best_val_loss:
                improved = True
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
        else:
            if val_acc > self.best_val_acc:
                improved = True
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc

        if improved:
            self.no_improvement_count = 0
            save_path = self.save_model()
            if save_path:
                logging.info(
                    f"Improved {early_stopping_metric}! Model saved to {save_path}")
            return True
        else:
            self.no_improvement_count += 1
            logging.info(
                f"No improvement in {early_stopping_metric} for {self.no_improvement_count} epochs.")
            return False

    def _should_stop_early(self, epoch, patience, min_epochs=0):
        """Check if early stopping should be triggered"""
        return (self.no_improvement_count >= patience and
                epoch >= min_epochs)

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'tb_logger'):
            self.tb_logger.close()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
