"""
Object Detection Trainer - specialized for YOLO-style object detection models
"""
import torch
import torch.nn as nn
import logging
from torch.cuda.amp import autocast
from tqdm import tqdm

from .base_trainer import BaseTrainer


class ObjectDetectionTrainer(BaseTrainer):
    """Specialized trainer for object detection models"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, model_name, task_name, dataset_name,
                 device, config, scheduler=None, model_depth=None):

        super().__init__(model, train_loader, val_loader, test_loader,
                         optimizer, criterion, model_name, task_name, dataset_name,
                         device, config, scheduler, model_depth)

        # Object detection specific settings
        self.is_object_detection = True

        # Use model's compute_loss if available
        if hasattr(model, 'compute_loss'):
            self.criterion = model.compute_loss
            logging.info("Using model's compute_loss for object detection.")
        else:
            logging.warning(
                "Model doesn't have compute_loss method, using provided criterion")

        # Detection thresholds - use very low thresholds during early training
        self.confidence_threshold = getattr(
            config, 'confidence_threshold', 0.01)  # Lowered from 0.1 to 0.01
        self.nms_threshold = getattr(config, 'nms_threshold', 0.5)

        logging.info(
            f"ObjectDetectionTrainer initialized with confidence_threshold={self.confidence_threshold}")

    def train(self, patience):
        """Training loop for object detection"""
        if self.has_trained:
            logging.warning(
                f"{self.model} has already been trained. Training again will overwrite the existing model.")
            return

        logging.info(f"Training object detection model: {self.model_name}...")
        self.has_trained = True

        torch.cuda.empty_cache()

        min_epochs = getattr(self.config, 'min_epochs', 0)
        early_stopping_metric = getattr(
            self.config, 'early_stopping_metric', 'loss')

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss = self._train_epoch(epoch)

            # Validation phase
            val_loss, val_metrics = self.validate_with_metrics()
            val_acc = val_metrics.get('accuracy', 0.0) if isinstance(
                val_metrics, dict) else val_metrics

            # Update scheduler
            self._update_scheduler(val_loss, val_acc, early_stopping_metric)

            # Check for improvement and save model
            improved = self._check_improvement(
                val_loss, val_acc, early_stopping_metric)

            # Early stopping check
            if self._should_stop_early(epoch, patience, min_epochs):
                logging.info(
                    f"Early stopping triggered after {epoch+1} epochs")
                break

            # Log epoch results
            logging.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Quality: {val_acc:.4f}"
            )

            # Store history
            self._update_history(epoch, train_loss, val_loss, val_acc)

        logging.info(
            f"Training finished. Best metrics - Loss: {self.best_val_loss:.4f}, Quality: {self.best_val_acc:.4f}")
        self.best_metrics = {'loss': self.best_val_loss,
                             'accuracy': self.best_val_acc}

        return self.best_val_loss, self.best_val_acc

    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        train_pbar = tqdm(self.train_loader,
                          desc=f'Epoch {epoch+1}/{self.epochs} - Training')

        for i, (images, targets) in enumerate(train_pbar):
            # Move data to device
            images, targets = self._move_to_device(images, targets)

            # Zero gradients
            self.optimizer.zero_grad()

            with autocast():
                # Forward pass
                loss = self._forward_pass(images, targets)

                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(
                        f"Invalid loss detected: {loss}, skipping batch")
                    continue

            # Backward pass with gradient accumulation
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps

            # Update progress bar
            train_pbar.set_postfix(
                {'Loss': f'{loss.item() * self.accumulation_steps:.4f}'})

            # Debug logging
            if i % 10 == 0:
                logging.debug(
                    f'Epoch {epoch+1}, Batch {i+1}/{len(self.train_loader)}, Loss: {loss.item() * self.accumulation_steps:.4f}')

        # Handle remaining steps
        if len(self.train_loader) % self.accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(self.train_loader)
        return avg_train_loss

    def _forward_pass(self, images, targets):
        """Forward pass for object detection model"""
        try:
            # Training mode - model expects targets
            outputs = self.model(images, targets)

            if isinstance(outputs, tuple) and len(outputs) == 2:
                _, loss = outputs
            elif isinstance(outputs, dict) and 'total_loss' in outputs:
                loss = outputs['total_loss']
            elif hasattr(self.model, 'compute_loss'):
                # If model has compute_loss method, use it
                loss = self.model.compute_loss(outputs, targets)
            elif isinstance(outputs, torch.Tensor):
                loss = self.criterion(outputs, targets)
            else:
                # Fallback loss
                loss = torch.tensor(
                    1.0, device=self.device, requires_grad=True)
                logging.warning(
                    "No valid loss computation found, using fallback loss")

            return loss

        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            # Return a fallback loss
            return torch.tensor(1.0, device=self.device, requires_grad=True)

    def _move_to_device(self, images, targets):
        """Move images and targets to device"""
        # Handle images
        if isinstance(images, list):
            images = [img.to(self.device, non_blocking=True) for img in images]
        else:
            images = images.to(self.device, non_blocking=True)

        # Handle targets
        if isinstance(targets, dict):
            targets = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}
        elif isinstance(targets, list):
            targets = [t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                       for t in targets]
        elif isinstance(targets, torch.Tensor):
            targets = targets.to(self.device, non_blocking=True)

        return images, targets

    def validate_with_metrics(self):
        """Validation with object detection metrics"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        total_detections = 0
        total_confidence = 0.0
        valid_detections = 0
        total_images = 0

        logging.debug(
            "Starting validation with detailed object detection metrics")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                try:
                    if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                        images, targets = batch_data
                    else:
                        logging.warning(
                            f"Unexpected batch format: {type(batch_data)}")
                        continue

                    # Move to device
                    images, targets = self._move_to_device(images, targets)

                    # Set model to eval mode explicitly
                    self.model.eval()

                    # Get inference outputs (no targets for inference)
                    inference_outputs = self.model(images, None)

                    # Count images
                    if isinstance(images, torch.Tensor):
                        batch_size = images.shape[0]
                    elif isinstance(images, list):
                        batch_size = len(images)
                    else:
                        batch_size = 1

                    total_images += batch_size

                    # Analyze detection outputs
                    batch_stats = self._analyze_detections(inference_outputs)
                    total_detections += batch_stats['detections']
                    total_confidence += batch_stats['confidence']
                    valid_detections += batch_stats['valid_detections']

                    # Compute validation loss
                    try:
                        if hasattr(self.model, 'compute_loss'):
                            loss = self.model.compute_loss(
                                inference_outputs, targets)
                        else:
                            loss = 0.5  # Fallback constant loss

                        if isinstance(loss, torch.Tensor):
                            total_loss += loss.item()
                        else:
                            total_loss += loss
                    except Exception as e:
                        logging.warning(
                            f"Error computing validation loss: {e}")
                        total_loss += 0.5

                    num_batches += 1

                    # Debug logging for first few batches
                    if batch_idx < 3:
                        logging.debug(f"Batch {batch_idx}: {batch_stats}")

                except Exception as e:
                    logging.error(
                        f"Error processing validation batch {batch_idx}: {e}")
                    continue

        # Compute metrics
        avg_val_loss = total_loss / max(num_batches, 1)
        avg_detections_per_image = total_detections / max(total_images, 1)
        avg_confidence = total_confidence / max(total_detections, 1)
        valid_detection_ratio = valid_detections / max(total_detections, 1)

        # Quality score (0-1, higher is better)
        # Normalize by expected 3 detections
        detection_quality = min(1.0, avg_detections_per_image / 3.0)
        confidence_quality = avg_confidence
        valid_quality = valid_detection_ratio

        overall_quality = 0.4 * detection_quality + \
            0.4 * confidence_quality + 0.2 * valid_quality

        # Log validation results
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Avg Detections/Image: {avg_detections_per_image:.2f}")
        logging.info(f"Avg Confidence: {avg_confidence:.3f}")
        logging.info(f"Valid Detection Ratio: {valid_detection_ratio:.3f}")
        logging.info(f"Detection Quality Score: {overall_quality:.3f}")

        metrics = {
            'loss': avg_val_loss,
            'accuracy': overall_quality,
            'detections_per_image': avg_detections_per_image,
            'avg_confidence': avg_confidence,
            'valid_detection_ratio': valid_detection_ratio
        }

        return avg_val_loss, metrics

    def _analyze_detections(self, outputs):
        """Analyze detection outputs and return statistics"""
        stats = {
            'detections': 0,
            'confidence': 0.0,
            'valid_detections': 0
        }

        if not isinstance(outputs, list):
            return stats

        for pred in outputs:
            if isinstance(pred, dict):
                boxes = pred.get('boxes', torch.empty(0, 4))
                scores = pred.get('scores', torch.empty(0))

                if isinstance(boxes, torch.Tensor):
                    num_dets = boxes.shape[0]
                    stats['detections'] += num_dets

                    if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                        stats['confidence'] += torch.sum(scores).item()
                        stats['valid_detections'] += torch.sum(
                            scores > self.confidence_threshold).item()

        return stats

    def validate(self):
        """Simple validation method for compatibility"""
        val_loss, metrics = self.validate_with_metrics()
        accuracy = metrics.get('accuracy', 0.0) if isinstance(
            metrics, dict) else metrics
        return val_loss, accuracy

    def _update_history(self, epoch, train_loss, val_loss, val_acc):
        """Update training history"""
        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)
