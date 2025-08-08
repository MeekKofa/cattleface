"""
Classification Trainer - for standard image classification tasks
"""
import torch
import torch.nn as nn
import logging
from torch.cuda.amp import autocast
from tqdm import tqdm

from .base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """Specialized trainer for classification models"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, model_name, task_name, dataset_name,
                 device, config, scheduler=None, model_depth=None):

        super().__init__(model, train_loader, val_loader, test_loader,
                         optimizer, criterion, model_name, task_name, dataset_name,
                         device, config, scheduler, model_depth)

        self.is_object_detection = False
        logging.info(f"ClassificationTrainer initialized for {model_name}")

    def train(self, patience):
        """Training loop for classification"""
        if self.has_trained:
            logging.warning(f"{self.model} has already been trained.")
            return

        logging.info(f"Training classification model: {self.model_name}...")
        self.has_trained = True

        min_epochs = getattr(self.config, 'min_epochs', 0)
        early_stopping_metric = getattr(
            self.config, 'early_stopping_metric', 'loss')

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss = self._train_epoch(epoch)

            # Validation phase
            val_loss, val_acc = self.validate()

            # Update scheduler
            self._update_scheduler(val_loss, val_acc, early_stopping_metric)

            # Check for improvement
            improved = self._check_improvement(
                val_loss, val_acc, early_stopping_metric)

            # Early stopping
            if self._should_stop_early(epoch, patience, min_epochs):
                logging.info(
                    f"Early stopping triggered after {epoch+1} epochs")
                break

            # Log results
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

            # Store history
            self._update_history(epoch, train_loss, val_loss, val_acc)

        logging.info(
            f"Training finished. Best metrics - Loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f}")
        return self.best_val_loss, self.best_val_acc

    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        train_pbar = tqdm(self.train_loader,
                          desc=f'Epoch {epoch+1}/{self.epochs} - Training')

        for i, (images, targets) in enumerate(train_pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Gradient accumulation
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps
            train_pbar.set_postfix(
                {'Loss': f'{loss.item() * self.accumulation_steps:.4f}'})

        return epoch_loss / len(self.train_loader)

    def validate(self):
        """Validation for classification"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                total += targets.size(0)

        val_loss /= len(self.val_loader)
        accuracy = correct / total if total > 0 else 0

        return val_loss, accuracy

    def _update_history(self, epoch, train_loss, val_loss, val_acc):
        """Update training history"""
        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
