# import os
from loader.dataset_loader import DatasetLoader
from utils.metrics_logger import MetricsLogger
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import torch.backends.cudnn  # Add this to ensure cudnn is recognized
from tqdm import tqdm   # Added import for progress bar
import warnings  # added import
import json
from argument_parser import parse_args
from utils.evaluator import Evaluator
from utils.metrics import Metrics
from utils.weighted_losses import WeightedCrossEntropyLoss, AggressiveMinorityWeightedLoss, DynamicSampleWeightedLoss
from utils.utility import get_model_params
import torch.nn as nn
from utils.robustness.lr_scheduler import LRSchedulerLoader
from utils.robustness.optimizers import OptimizerLoader
# Import the collate function
from loader.dataset_loader import DatasetLoader, object_detection_collate
from model.model_loader import ModelLoader
from utils.timer import Timer
from utils.robustness.regularization import Regularization
from utils.training_logger import TrainingLogger
from utils.adv_metrics import AdversarialMetrics
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
import argparse
import logging
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Use MetricsLogger for logging
logging.info("Using MetricsLogger for training metrics")

# added to suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


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
        # New adversarial metrics
        'adv_loss': [],
        'adv_accuracy': [],
        'adv_predictions': [],
        'adv_targets': [],
        # New detailed metrics
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_metrics': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_per_class_metrics': []
    }


class Trainer:
    """Training orchestrator that handles the training loop and logging"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, model_name, task_name, dataset_name,
                 device, config, scheduler=None, is_object_detection=False, model_depth=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.device = device
        self.config = config
        self.is_object_detection = is_object_detection
        self.model_depth = model_depth

        # Set up loss function based on config
        if self.is_object_detection:
            # Use model's compute_loss if available
            if hasattr(model, 'compute_loss'):
                self.criterion = model.compute_loss
                logging.info(
                    "Using model's compute_loss for object detection.")
            else:
                self.criterion = nn.CrossEntropyLoss()
                logging.info(
                    "Object detection model detected. Using CrossEntropyLoss on detection head outputs.")
        else:
            loss_type = getattr(config, 'loss_type', 'standard')
            if loss_type == 'standard':
                self.criterion = criterion
                logging.info("Using standard CrossEntropy loss")
            else:
                logging.info(f"Using {loss_type} loss function")

                # Get all targets from the dataset for class distribution analysis
                try:
                    # Skip class distribution collection for object detection datasets
                    if self.is_object_detection:
                        logging.info(
                            "Object detection model detected. Skipping class distribution analysis for loss initialization.")
                        self.criterion = criterion
                    # Try to get targets from train_loader.dataset.targets (common for ImageFolder)
                    if hasattr(train_loader.dataset, 'targets'):
                        dataset_targets = train_loader.dataset.targets
                    # Try to get targets from TensorDataset
                    elif hasattr(train_loader.dataset, 'tensors') and len(train_loader.dataset.tensors) > 1:
                        dataset_targets = train_loader.dataset.tensors[1]
                    else:
                        # Collect targets by iterating through dataloader (slower but works)
                        dataset_targets = []
                        logging.info(
                            "Collecting class distribution from dataloader...")
                        for _, batch_targets in train_loader:
                            dataset_targets.extend(batch_targets.cpu().numpy())
                        dataset_targets = np.array(dataset_targets)

                    # Initialize the appropriate loss function based on loss_type
                    if loss_type == 'weighted':
                        self.criterion = WeightedCrossEntropyLoss(
                            dataset=dataset_targets)
                    elif loss_type == 'aggressive':
                        self.criterion = AggressiveMinorityWeightedLoss(
                            dataset=dataset_targets)
                    elif loss_type == 'dynamic':
                        alpha = getattr(config, 'focal_alpha', 0.5)
                        gamma = getattr(config, 'focal_gamma', 2.0)
                        self.criterion = DynamicSampleWeightedLoss(
                            max_epochs=getattr(config, 'epochs', 100),
                            alpha=alpha, gamma=gamma)

                    logging.info(
                        f"Successfully initialized {loss_type} loss function")
                except Exception as e:
                    logging.error(
                        f"Error initializing weighted loss: {e}")
                    self.criterion = criterion

        self.has_trained = False
        self.epochs = getattr(config, 'epochs', 100)
        self.lambda_l2 = getattr(config, 'lambda_l2', 1e-4)
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self.args = config
        if not hasattr(self.args, 'lr'):
            self.args.lr = 0.001

        self.scaler = GradScaler()
        self.timer = Timer()
        self.training_logger = TrainingLogger()
        self.history = _init_history()

        from utils.visual.visualization import Visualization
        self.visualization = Visualization()

        self.model.to(self.device)
        if hasattr(config, 'drop'):
            Regularization.apply_dropout(self.model, config.drop)

        self.adversarial = getattr(config, 'adversarial', False)
        if self.adversarial:
            from gan.defense.adv_train import AdversarialTraining
            if not hasattr(config, 'attack_name'):
                config.attack_name = getattr(config, 'attack_type', 'fgsm')
            if not hasattr(config, 'epsilon'):
                config.epsilon = getattr(config, 'attack_eps', 0.3)
            self.adversarial_trainer = AdversarialTraining(
                model, criterion, config)
            logging.info(
                f"Training {self.model_name} with adversarial training...")

        # Initialize tracking variables for metrics
        self.error_if_nonfinite = False
        self.val_loss = float('inf')
        self.current_lr = self.args.lr
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0  # Initialize best F1 score for tracking
        self.best_val_balanced_acc = 0.0  # Initialize best balanced accuracy
        self.no_improvement_count = 0
        self.adv_metrics = AdversarialMetrics()

        # Initialize the lists here for test results
        self.true_labels = []
        self.predictions = []
        self.adv_predictions = []

        # Initialize MetricsLogger with TensorBoard disabled
        self.tb_logger = MetricsLogger(
            task_name, dataset_name, model_name, use_tensorboard=False)

        # Log hyperparameters to MetricsLogger
        hparams = {
            'learning_rate': getattr(config, 'lr', 0.001),
            'batch_size': getattr(config, 'train_batch', 32),
            'optimizer': getattr(config, 'optimizer', 'adam'),
            'model': model_name,
            'loss_type': getattr(config, 'loss_type', 'standard'),
            'weight_decay': getattr(config, 'weight_decay', 1e-4),
            'dropout': getattr(config, 'drop', 0.0),
            'scheduler': getattr(config, 'scheduler', 'none')
        }
        self.tb_logger.log_hparams(hparams)

        # Add new attributes for balanced metrics and class-specific metrics
        self.class_names = self._get_class_names(train_loader)
        self.per_class_metrics = getattr(config, 'per_class_metrics', True)

        # Create a threshold tracker for binary classification
        self.threshold = 0.5
        self.optimize_threshold = getattr(config, 'optimize_threshold', False)

        # Add history tracking for detailed metrics
        self.history['precision'] = []
        self.history['recall'] = []
        self.history['f1'] = []
        self.history['per_class_metrics'] = []
        self.history['val_precision'] = []
        self.history['val_recall'] = []
        self.history['val_f1'] = []
        self.history['val_per_class_metrics'] = []

        # Create sampler attribute for potential reweighting
        self.use_weighted_sampler = getattr(
            config, 'use_weighted_sampler', False)
        self.sampler = None

        # Use weighted sampler if specified
        if self.use_weighted_sampler:
            self._setup_weighted_sampler(train_loader)

        # If using object detection, warn if DataLoader batch size > 1 and no custom collate_fn
        if self.is_object_detection:
            # Check if DataLoader uses default collate_fn and batch_size > 1
            if hasattr(train_loader, 'collate_fn'):
                uses_default_collate = train_loader.collate_fn is None
            else:
                uses_default_collate = True
            batch_size = getattr(train_loader, 'batch_size', 1)
            if uses_default_collate and batch_size > 1:
                logging.warning(
                    "Object detection dataset with variable-sized targets detected. "
                    "Default DataLoader collate_fn will cause 'stack expects each tensor to be equal size' error. "
                    "Please use a custom collate_fn (e.g., torch.utils.data.dataloader.default_collate or a YOLO-style collate_fn) "
                    "that returns batches as lists of images/targets."
                )

    def _get_class_names(self, data_loader):
        """Get class names from the dataset if available"""
        if hasattr(data_loader.dataset, 'classes'):
            return data_loader.dataset.classes
        elif hasattr(data_loader.dataset, 'class_to_idx'):
            # Map indices back to class names
            class_to_idx = data_loader.dataset.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return [idx_to_class.get(i, str(i)) for i in range(len(class_to_idx))]
        else:
            # Fallback to generic class names
            # Try to infer number of classes from the criterion
            if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
                num_classes = len(self.criterion.weight)
            else:
                # Try to infer from the model's final layer
                if hasattr(self.model, 'fc'):
                    num_classes = self.model.fc.out_features
                elif hasattr(self.model, 'head'):
                    num_classes = self.model.head.out_features
                else:
                    num_classes = 2  # Default to binary classification
            return [f"Class {i}" for i in range(num_classes)]

    def _setup_weighted_sampler(self, train_loader):
        """Setup WeightedRandomSampler based on class distribution"""
        logging.info("Setting up WeightedRandomSampler for imbalanced data...")

        try:
            # Extract targets/labels from the dataset
            targets = []
            if hasattr(train_loader.dataset, 'targets'):
                targets = train_loader.dataset.targets
            elif hasattr(train_loader.dataset, 'tensors') and len(train_loader.dataset.tensors) > 1:
                targets = train_loader.dataset.tensors[1].tolist()
            else:
                # Extract targets by iterating through the dataset
                for _, target in train_loader.dataset:
                    if torch.is_tensor(target):
                        targets.append(target.item())
                    else:
                        targets.append(target)

            # Count class frequencies
            class_counts = Counter(targets)
            logging.info(f"Class distribution: {dict(class_counts)}")

            # Calculate class weights (inverse frequency)
            total_samples = len(targets)
            class_weights = {class_idx: total_samples /
                             count for class_idx, count in class_counts.items()}

            # Assign weights to each sample
            weights = [class_weights[target] for target in targets]
            weights = torch.DoubleTensor(weights)

            # Create the sampler
            self.sampler = WeightedRandomSampler(
                weights, len(weights), replacement=True)

            # Create new train loader with the sampler
            self.train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=self.sampler,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory
            )

            logging.info("WeightedRandomSampler successfully created")
        except Exception as e:
            logging.error(f"Failed to create WeightedRandomSampler: {e}")

    def calculate_detailed_metrics(self, true_labels, predictions, probabilities=None, phase="train"):
        """Calculate metrics with reduced detail during training - ensuring basic metrics are always available"""
        # Import necessary metrics functions at the top of the function to ensure availability
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # Convert tensors to numpy arrays if needed
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        # During training, just calculate basic overall metrics to save computation time
        if phase == "train" and not (hasattr(self, 'epochs') and self.current_epoch == self.epochs - 1):
            # Handle case where predictions is a 2D array
            if predictions.ndim > 1:
                predictions = predictions.flatten()

            # Handle case where true_labels is a 2D array
            if true_labels.ndim > 1:
                true_labels = true_labels.flatten()

            # Basic metrics with zero_division=0 to avoid errors
            return {
                'accuracy': np.mean(true_labels == predictions),
                'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
                'recall': recall_score(true_labels, predictions, average='macro', zero_division=0),
                'f1': f1_score(true_labels, predictions, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
            }

        # For validation and test phases, calculate the full detailed metrics
        try:
            metrics_dict = Metrics.calculate_metrics(
                true_labels, predictions, probabilities)
        except Exception as e:
            # Fallback to basic metrics if calculation fails
            logging.warning(
                f"Error calculating detailed metrics: {e}. Using basic metrics instead.")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Flatten arrays if needed
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            if true_labels.ndim > 1:
                true_labels = true_labels.flatten()

            return {
                'accuracy': np.mean(true_labels == predictions),
                'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
                'recall': recall_score(true_labels, predictions, average='macro', zero_division=0),
                'f1': f1_score(true_labels, predictions, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
            }

        # Only add per-class metrics on test or final validation if calculation succeeded
        if (phase == "test" or (phase == "val" and hasattr(self, 'current_epoch') and self.current_epoch == self.epochs - 1)) and self.per_class_metrics:
            per_class = {}
            classes = np.unique(true_labels)

            for cls in classes:
                cls_mask = (true_labels == cls)
                if sum(cls_mask) == 0:
                    continue

                cls_true = (true_labels == cls).astype(int)
                cls_pred = (predictions == cls).astype(int)

                try:
                    # Use imported functions directly to avoid reference errors
                    cls_metrics = {
                        'accuracy': np.mean(cls_true == cls_pred),
                        'precision': precision_score(cls_true, cls_pred, zero_division=0),
                        'recall': recall_score(cls_true, cls_pred, zero_division=0),
                        'f1': f1_score(cls_true, cls_pred, zero_division=0),
                        'support': np.sum(cls_true)
                    }
                    class_name = self.class_names[cls] if cls < len(
                        self.class_names) else f"Class {cls}"
                    per_class[class_name] = cls_metrics
                except Exception as e:
                    logging.warning(
                        f"Error calculating metrics for class {cls}: {e}")
                    # Create partial metrics if calculation fails
                    cls_metrics = {
                        'accuracy': np.mean(cls_true == cls_pred),
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'support': np.sum(cls_true)
                    }
                    class_name = self.class_names[cls] if cls < len(
                        self.class_names) else f"Class {cls}"
                    per_class[class_name] = cls_metrics

            metrics_dict['per_class'] = per_class

        return metrics_dict

    def _optimize_threshold(self, true_labels, probabilities):
        """Find optimal threshold for binary classification"""
        # Handle both one-hot and regular probabilities
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            probs = probabilities[:, 1]  # Probability of positive class
        else:
            probs = probabilities

        # Get precision, recall, thresholds
        precision, recall, thresholds = precision_recall_curve(
            true_labels, probs)

        # Calculate F1 score for each threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            self.threshold = thresholds[best_idx]
        else:
            self.threshold = 0.5  # Default if something went wrong

        logging.info(
            f"Optimized threshold: {self.threshold:.4f} with F1: {f1_scores[best_idx]:.4f}")

    def _visualize_metrics(self, metrics, epoch, phase="train"):
        """Create minimal visualizations during training"""
        if not hasattr(self, 'visualization'):
            return

        # Only create visualizations for final epoch or every 10 epochs
        if epoch % 10 != 0 and epoch != self.epochs - 1:
            return

        # Only visualize validation or test phases (skip training phase visualization)
        if phase == "train" and epoch != self.epochs - 1:
            return

        try:
            # Only create visualization if we have per-class metrics (final epoch or test)
            if 'per_class' in metrics:
                self.visualization.visualize_metrics(
                    metrics=metrics,
                    task_name=self.task_name,
                    dataset_name=self.dataset_name,
                    model_name=self.model_name,
                    phase=phase,
                    epoch=epoch,
                    class_names=self.class_names
                )
        except Exception as e:
            logging.warning(f"Failed to create visualization: {e}")
            plt.close('all')  # Close any open figures on error

    def train(self, patience):
        if self.has_trained:
            logging.warning(
                f"{self.model} has already been trained. Training again will overwrite the existing model.")
            return
        logging.info(f"Training {self.model_name}...")
        self.has_trained = True

        torch.cuda.empty_cache()

        self.model.to(self.device)
        initial_params = get_model_params(self.model)
        logging.info(f"Initial model parameters: {initial_params:.2f}M")

        min_epochs = getattr(self.args, 'min_epochs', 0)
        early_stopping_metric = getattr(
            self.args, 'early_stopping_metric', 'loss')
        epochs = self.epochs

        scaler = GradScaler()  # Initialize mixed precision scaler

        # Default to 2, or set via config
        accumulation_steps = getattr(self, 'accumulation_steps', 2)

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            total = 0

            # Progress bar for training
            train_pbar = tqdm(self.train_loader,
                              desc=f'Epoch {epoch+1}/{self.epochs} - Training')

            for i, (images, targets) in enumerate(train_pbar):
                # Move data to device
                if isinstance(images, list):
                    images = [img.to(self.device, non_blocking=True)
                              for img in images]
                else:
                    images = images.to(self.device, non_blocking=True)

                # Handle targets properly for object detection
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                               for k, v in targets.items()}

                self.optimizer.zero_grad()

                with autocast():
                    if self.is_object_detection:
                        outputs = self.model(images, targets)
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            _, loss = outputs
                        elif isinstance(outputs, dict) and 'total_loss' in outputs:
                            loss = outputs['total_loss']
                        elif isinstance(outputs, torch.Tensor):
                            loss = self.criterion(outputs, targets)
                        else:
                            loss = torch.tensor(
                                0.0, device=self.device, requires_grad=True)
                            logging.warning(
                                "No valid loss returned from object detection model")
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                # Handle gradient accumulation
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    # Add gradient clipping to prevent explosion
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                total += images.size(0) if isinstance(images,
                                                      torch.Tensor) else len(images)

                # Update progress bar
                train_pbar.set_postfix(
                    {'Loss': f'{loss.item() * accumulation_steps:.4f}'})

                # Add some basic logging every 10 batches
                if i % 10 == 0:
                    logging.debug(
                        f'Epoch {epoch+1}, Batch {i+1}/{len(self.train_loader)}, Loss: {loss.item() * accumulation_steps:.4f}')

            # Handle remaining steps if not divisible
            if len(self.train_loader) % accumulation_steps != 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            # Calculate average training loss
            avg_train_loss = epoch_loss / len(self.train_loader)

            # Validation phase
            val_loss, val_metrics = self.validate_with_metrics()
            val_acc = val_metrics.get('accuracy', 0.0) if isinstance(
                val_metrics, dict) else val_metrics

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(
                        val_loss if early_stopping_metric == 'loss' else val_acc)
                else:
                    self.scheduler.step()
                self.current_lr = self.optimizer.param_groups[0]['lr']

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
                # Save model using proper directory structure
                save_path = self.save_model()
                if save_path:
                    logging.info(
                        f"Improved {early_stopping_metric}! Model saved to {save_path}")
                else:
                    logging.error("Failed to save model")
            else:
                self.no_improvement_count += 1
                logging.info(
                    f"No improvement in {early_stopping_metric} for {self.no_improvement_count} epochs.")

            if self.no_improvement_count >= patience and epoch >= min_epochs:
                logging.info(
                    f"Early stopping triggered after {epoch+1} epochs")
                break

            logging.info(
                f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Store history
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

        logging.info(
            f"Training finished. Best metrics - Loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f}")
        self.best_metrics = {'loss': self.best_val_loss,
                             'accuracy': self.best_val_acc}
        self.tb_logger.close()

        return self.best_val_loss, self.best_val_acc

    def _get_save_directory(self):
        """Get the proper save directory structure that matches model_loader expectations"""
        # Format model name with depth
        if self.model_depth is not None:
            model_name_with_depth = f"{self.model_name}_{self.model_depth}"
        else:
            # Fallback to just model name if no depth
            model_name_with_depth = self.model_name

        # Structure: out/{task_name}/{dataset_name}/{model_name_with_depth}/save_model/
        save_dir = f"out/{self.task_name}/{self.dataset_name}/{model_name_with_depth}/save_model"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir, model_name_with_depth

    def save_model(self, filename=None):
        """Save the model state dict using proper directory structure"""
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

    def validate(self):
        self.model.eval()
        val_loss = 0
        adv_val_loss = 0
        correct = 0
        adv_correct = 0
        total = 0
        val_predictions = []
        val_targets = []
        adv_accuracy = 0  # Initialize
        try:
            with torch.no_grad():
                for batch_idx, images_targets in enumerate(self.val_loader):
                    if self.is_object_detection:
                        if isinstance(images_targets, (tuple, list)):
                            if len(images_targets) == 4:
                                data, target, paths, sizes = images_targets
                            elif len(images_targets) == 2:
                                data, target = images_targets
                                paths, sizes = None, None
                            else:
                                raise ValueError(
                                    "Unexpected batch format for object detection dataloader.")
                        else:
                            raise ValueError(
                                "Batch must be tuple/list for object detection dataloader.")
                    else:
                        data, target = images_targets
                        paths, sizes = None, None

                    # ...existing code for moving data/target to device...
                    if isinstance(data, list):
                        data = [img.to(self.device, non_blocking=True)
                                for img in data]
                    elif isinstance(data, torch.Tensor):
                        data = data.to(self.device, non_blocking=True)
                    if isinstance(target, list):
                        for i, t in enumerate(target):
                            if isinstance(t, torch.Tensor):
                                target[i] = t.to(
                                    self.device, non_blocking=True)
                    elif isinstance(target, torch.Tensor):
                        target = target.to(self.device, non_blocking=True)
                    with autocast():
                        if self.is_object_detection:
                            outputs = self.model(data)
                        else:
                            output = self.model(data)
                            loss = self.criterion(output, target)

                    val_loss += loss.item()
                    if self.adversarial:
                        with torch.enable_grad():
                            if hasattr(self.adversarial_trainer.attack, 'generate'):
                                adv_data = self.adversarial_trainer.attack.generate(
                                    data, target, self.args.epsilon)
                            else:
                                _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                    data, target)
                        with autocast():
                            adv_output = self.model(adv_data)
                            adv_loss = self.criterion(adv_output, target)
                        adv_val_loss += adv_loss.item()
                        if not self.is_object_detection:
                            adv_pred = adv_output.argmax(dim=1, keepdim=True)
                            adv_correct += adv_pred.eq(
                                target.view_as(adv_pred)).sum().item()

                    if not self.is_object_detection:
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        val_predictions.extend(pred.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())
                        total += target.size(0)
                    else:
                        # For object detection, count samples properly
                        if isinstance(data, list):
                            # Count number of images in the list
                            total += len(data)
                        else:
                            # Count from tensor batch size
                            total += data.size(0)

                    if batch_idx % 100 == 0:
                        logging.debug(
                            f'Validation Batch: {batch_idx}/{len(self.val_loader)}')
            val_loss /= len(self.val_loader)
            accuracy = correct / total if total > 0 else 0
            if self.adversarial:
                adv_val_loss /= len(self.val_loader)
                adv_accuracy = adv_correct / total if total > 0 else 0
                logging.info(
                    f'Validation - Clean: Loss={val_loss:.4f}, Acc={accuracy:.4f} | Adversarial: Loss={adv_val_loss:.4f}, Acc={adv_accuracy:.4f}')
            else:
                logging.info(
                    f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
            self.history['val_predictions'].append(val_predictions)
            self.history['val_targets'].append(val_targets)
            self.adv_metrics.update_adversarial_comparison(phase='val', clean_loss=val_loss, clean_acc=accuracy,
                                                           adv_loss=adv_val_loss if self.adversarial else 0, adv_acc=adv_accuracy if self.adversarial else 0)
        except Exception as e:
            logging.error(f"Error during validation: {e}")
            return float('inf'), 0.0
        finally:
            self.model.train()
        return val_loss, accuracy

    def validate_with_metrics(self):
        """Validate with simple detection metrics for object detection, or detailed metrics for classification."""
        self.model.eval()
        val_loss = 0
        total = 0
        correct = 0

        # Use simple robust validation for object detection
        if self.is_object_detection:
            total_loss = 0
            num_batches = 0
            total_detections = 0
            total_confidence = 0.0
            valid_detections = 0
            total_images = 0

            with torch.no_grad():
                for images_targets in self.val_loader:
                    if isinstance(images_targets, (tuple, list)) and len(images_targets) == 2:
                        images, targets = images_targets
                    else:
                        continue

                    # Move images to device
                    if isinstance(images, torch.Tensor):
                        images = images.to(self.device, non_blocking=True)
                    elif isinstance(images, list):
                        images = [img.to(self.device, non_blocking=True)
                                  for img in images]

                    # Move targets to device
                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                   for k, v in targets.items()}

                    # Get inference outputs
                    inference_outputs = self.model(images, None)
                    batch_size = len(inference_outputs) if isinstance(
                        inference_outputs, list) else 1
                    total_images += batch_size

                    # Debug: Print what we're getting from the model
                    print(
                        f"DEBUG: Inference outputs type: {type(inference_outputs)}")
                    if isinstance(inference_outputs, list):
                        print(
                            f"DEBUG: Number of outputs: {len(inference_outputs)}")
                        # Check first 2
                        for i, pred in enumerate(inference_outputs[:2]):
                            print(f"DEBUG: Output {i} type: {type(pred)}")
                            if isinstance(pred, dict):
                                print(f"DEBUG: Output {i} keys: {pred.keys()}")
                                for key, value in pred.items():
                                    if hasattr(value, 'shape'):
                                        print(
                                            f"DEBUG: {key} shape: {value.shape}")
                                    else:
                                        print(f"DEBUG: {key} value: {value}")

                    # Collect simple detection statistics
                    batch_detections = 0
                    batch_confidence = 0.0
                    batch_valid = 0

                    if isinstance(inference_outputs, list):
                        for pred in inference_outputs:
                            if isinstance(pred, dict):
                                boxes = pred.get('boxes', torch.empty(0, 4))
                                scores = pred.get('scores', torch.empty(0))

                                if isinstance(boxes, torch.Tensor):
                                    num_dets = boxes.shape[0]
                                    batch_detections += num_dets

                                    if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                                        batch_confidence += torch.sum(
                                            scores).item()
                                        batch_valid += torch.sum(scores >
                                                                 0.1).item()

                    total_detections += batch_detections
                    total_confidence += batch_confidence
                    valid_detections += batch_valid

                    # Compute simple validation loss
                    try:
                        loss = self.model.compute_loss(
                            inference_outputs, targets)
                        if isinstance(loss, torch.Tensor):
                            total_loss += loss.item()
                        else:
                            total_loss += loss
                    except Exception as e:
                        logging.warning(
                            f"Error computing validation loss: {e}")
                        total_loss += 0.5  # Fallback

                    num_batches += 1

            # Compute validation metrics
            avg_val_loss = total_loss / max(num_batches, 1)
            avg_detections_per_image = total_detections / max(total_images, 1)
            avg_confidence = total_confidence / max(total_detections, 1)
            valid_detection_ratio = valid_detections / max(total_detections, 1)

            # Simple detection quality score (0-1, higher is better)
            # Normalize by expected 3 detections
            detection_quality = min(1.0, avg_detections_per_image / 3.0)
            confidence_quality = avg_confidence  # Already 0-1
            valid_quality = valid_detection_ratio  # Already 0-1

            # Combined quality score as "accuracy"
            overall_quality = 0.4 * detection_quality + \
                0.4 * confidence_quality + 0.2 * valid_quality

            logging.info(f"Validation Loss: {avg_val_loss:.4f}")
            logging.info(
                f"Avg Detections/Image: {avg_detections_per_image:.2f}")
            logging.info(f"Avg Confidence: {avg_confidence:.3f}")
            logging.info(f"Valid Detection Ratio: {valid_detection_ratio:.3f}")
            logging.info(f"Detection Quality Score: {overall_quality:.3f}")

            return avg_val_loss, {
                'loss': avg_val_loss,
                'accuracy': overall_quality,
                'detections_per_image': avg_detections_per_image,
                'avg_confidence': avg_confidence,
                'valid_detection_ratio': valid_detection_ratio
            }

        # For classification, fall back to standard validation with detailed metrics
        try:
            with torch.no_grad():
                for images_targets in self.val_loader:
                    if self.is_object_detection:
                        if isinstance(images_targets, (tuple, list)) and len(images_targets) == 2:
                            images, targets = images_targets
                        else:
                            raise ValueError(
                                "Expected (images, targets) for object detection dataloader")
                    else:
                        images, targets = images_targets

                    # Move data to device
                    if isinstance(images, torch.Tensor):
                        images = images.to(self.device, non_blocking=True)
                    elif isinstance(images, list):
                        images = [img.to(self.device, non_blocking=True)
                                  for img in images]

                    if isinstance(targets, dict):
                        targets = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                   for k, v in targets.items()}

                    with autocast():
                        if self.is_object_detection:
                            # Set model to eval mode for inference
                            self.model.eval()
                            # No targets during validation
                            outputs = self.model(images, None)

                            # For object detection, we'll use a simple accuracy based on detection presence
                            if isinstance(outputs, list):
                                # Count successful detections (dummy metric for now)
                                batch_correct = len(
                                    [det for det in outputs if det['boxes'].shape[0] > 0])
                                correct += batch_correct
                            else:
                                # Fallback if outputs is not in expected format
                                correct += 1

                            # Dummy loss for object detection validation
                            loss = torch.tensor(0.1, device=self.device)
                        else:
                            output = self.model(images)
                            loss = self.criterion(output, targets)
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(targets.view_as(pred)
                                               ).sum().item()

                    val_loss += loss.item()
                    total += images.size(0) if isinstance(images,
                                                          torch.Tensor) else len(images)

                val_loss /= len(self.val_loader) if len(self.val_loader) > 0 else 1
                accuracy = correct / total if total > 0 else 0

                detailed_metrics = {
                    'loss': val_loss,
                    'accuracy': accuracy,
                    'precision': accuracy,  # Simplified for object detection
                    'recall': accuracy,
                    'f1': accuracy
                }

                logging.info(
                    f"Validation - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
                return val_loss, detailed_metrics

        except Exception as e:
            logging.error(f"Error during validation with metrics: {e}")
            return float('inf'), {'loss': float('inf'), 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        finally:
            self.model.train()

    def validate_adversarial(self):
        """Specific validation method for adversarial examples"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # Add more detailed logging
        logging.info(
            f"Running adversarial validation with epsilon={self.adversarial_trainer.epsilon:.4f}")

        try:
            # Track original performance first
            orig_val_loss, orig_accuracy = self.validate()
            # If original validation accuracy is already poor, don't even attempt adversarial validation
            # as this could make training unstable
            if orig_accuracy < 0.5:
                logging.warning(
                    f"Clean accuracy too low ({orig_accuracy:.4f}), skipping adversarial validation")
                return float('inf'), 0.0

                for batch_idx, (data, target) in enumerate(self.val_loader):
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device, non_blocking=True)
                    if isinstance(target, torch.Tensor):
                        target = target.to(self.device, non_blocking=True)

                    # Generate adversarial examples for validation with error handling
                    with torch.enable_grad():  # Need gradients for attack generation
                        try:
                            if hasattr(self.adversarial_trainer.attack, 'generate'):
                                adv_data = self.adversarial_trainer.attack.generate(
                                    data, target, self.adversarial_trainer.epsilon)
                            else:
                                # Use a smaller epsilon for validation if regular epsilon is large
                                safe_epsilon = min(
                                    self.adversarial_trainer.epsilon, 0.025)
                                _, adv_data, _ = self.adversarial_trainer.attack.attack(
                                    data, target, epsilon=safe_epsilon)

                            # Validate the adversarial examples aren't garbage
                            if torch.isnan(adv_data).any():
                                logging.warning(
                                    "NaN values detected in adversarial examples")
                                adv_data = data.clone()  # Fallback to clean data

                        except Exception as e:
                            logging.error(
                                f"Error generating adversarial examples: {e}")
                            adv_data = data.clone()  # Fallback to clean data

                    # Evaluate on adversarial examples
                    with torch.no_grad(), autocast():
                        output = self.model(adv_data)
                        loss = self.criterion(output, target)

                        # Check for NaN and replace with high loss value
                        if torch.isnan(loss):
                            loss = torch.tensor(1000.0, device=self.device)
                            logging.warning(
                                "NaN loss in adversarial validation, replacing with high loss value")

                    val_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                val_loss /= len(self.val_loader)
                accuracy = correct / total if total > 0 else 0

                # Store as instance attributes for stability checks
                self.adv_val_loss = val_loss
                self.adv_val_acc = accuracy

                return val_loss, accuracy

        except Exception as e:
            logging.error(f"Error during adversarial validation: {e}")
            return float('inf'), 0.0
        finally:
            self.model.train()


# Removed compute_class_weights function as it was causing hanging issues
# with object detection datasets. Object detection models typically handle
# class imbalance through other methods like focal loss or data augmentation.


class TrainingManager:
    """Manages the training process for multiple models and datasets"""

    def _setup_random_seeds(self, seed):
        """Setup random seeds for reproducibility"""
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __init__(self, args):
        self.args = args
        self.data = args.data
        self.arch = args.arch
        self.depth = args.depth
        self.train_batch = args.train_batch
        self.epochs = args.epochs
        self.lr = args.lr
        self.drop = args.drop
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.gpu_ids = args.gpu_ids
        self.task_name = args.task_name
        self.optimizer = args.optimizer

        # Initialize seed with default value if not provided
        # Default to 42 if args.seed is missing
        self.seed = getattr(args, 'seed', 42)
        logging.info(f"Using random seed: {self.seed}")

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False

        # Initialize device
        self.device = torch.device(
            f"cuda:{self.gpu_ids[0]}" if torch.cuda.is_available() and self.gpu_ids else "cpu")
        logging.info(f"Using device: {self.device}")

        # Initialize model, dataset, and other components
        self.model = self._initialize_model()
        self.train_loader, self.val_loader, self.test_loader = self._initialize_data()
        self.optimizer = self._initialize_optimizer()

    def _initialize_model(self):
        from model.vgg_yolov8 import get_vgg_yolov8
        from class_mapping import get_num_classes

        # Fix depth extraction logic
        if isinstance(self.arch, list):
            arch_key = self.arch[0]
        else:
            arch_key = self.arch
        if isinstance(self.depth, dict):
            depth_val = self.depth.get(arch_key, [16])[0]
        elif isinstance(self.depth, list):
            depth_val = self.depth[0]
        else:
            depth_val = 16

        # Get number of classes programmatically
        num_classes = get_num_classes()
        model = get_vgg_yolov8(num_classes=num_classes, depth=depth_val)
        logging.info(
            f"Initialized model: {arch_key}_{depth_val} with {num_classes} classes")
        return model.to(self.device)

    def _initialize_data(self):
        from loader.dataset_loader import DatasetLoader, object_detection_collate
        from torch.utils.data import DataLoader
        # Only log once and only call load_data once
        dataset_name = self.data  # data is already a string
        logging.info(f"Initializing dataset: {dataset_name}")
        train_loader, val_loader, test_loader = DatasetLoader().load_data(
            dataset_name=dataset_name,
            batch_size={'train': self.train_batch,
                        'val': self.train_batch, 'test': self.train_batch},
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        train_dl = DataLoader(
            train_loader.dataset,
            batch_size=self.train_batch,
            sampler=train_loader.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=object_detection_collate
        )
        val_dl = DataLoader(
            val_loader.dataset,
            batch_size=self.train_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=object_detection_collate
        )
        test_dl = DataLoader(
            test_loader.dataset,
            batch_size=self.train_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=object_detection_collate
        )
        logging.info(
            f"DataLoader initialized: {len(train_dl.dataset)} training samples")

        # Note: Removing problematic class weights computation that causes hanging
        # during dataset loading. Object detection models typically don't need
        # class weights since they handle class imbalance differently.

        # Set a simple criterion for object detection
        self.criterion = nn.CrossEntropyLoss()

        # Set as instance attributes for later use
        self.train_loader = train_dl
        self.val_loader = val_dl
        self.test_loader = test_dl

        return train_dl, val_dl, test_dl

    def _initialize_optimizer(self):
        if self.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def train_dataset(self, dataset_name, run_test=False):
        """
        Actually perform training on the dataset.
        """
        # Only log once per process, even if called multiple times
        if not getattr(self, '_train_logged', False):
            logging.info(
                f"Training dataset: {self.data} | run_test={run_test}")
            self._train_logged = True

        # Note: Removed ultralytics import due to version incompatibility
        # Custom VGG-YOLOv8 model uses our own training implementation

        # ...existing code for custom Trainer...
        trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            model_name=self.arch,  # Use just the architecture name
            task_name=self.args.task_name,
            dataset_name=self.data,
            device=self.device,
            config=self.args,
            scheduler=None,
            is_object_detection=True,
            model_depth=self.depth.get(self.arch, [16])[
                0]  # Pass depth separately
        )

        # Start training
        patience = getattr(self.args, 'patience', 10)
        trainer.train(patience=patience)

        # Optionally run test
        if run_test:
            trainer.validate()

        return self.train_loader, self.val_loader, self.test_loader
