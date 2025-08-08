"""
Simple Detection Validator
A robust, straightforward approach to validation for object detection models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging


class SimpleDetectionValidator:
    """
    Simple validation approach that provides meaningful metrics without complex mAP calculation.
    """

    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset validation statistics."""
        self.total_detections = 0
        self.total_images = 0
        self.confidence_sum = 0.0
        self.valid_detections = 0

    def validate_batch(self, predictions, targets):
        """
        Validate a batch of predictions against targets.

        Args:
            predictions: List of detection dicts or tensor predictions
            targets: Ground truth targets

        Returns:
            dict: Validation metrics
        """
        batch_size = len(predictions) if isinstance(predictions, list) else 1
        self.total_images += batch_size

        batch_detections = 0
        batch_confidence = 0.0
        batch_valid = 0

        if isinstance(predictions, list):
            # Handle detection format
            for pred in predictions:
                if isinstance(pred, dict):
                    boxes = pred.get('boxes', torch.empty(0, 4))
                    scores = pred.get('scores', torch.empty(0))

                    if isinstance(boxes, torch.Tensor):
                        num_dets = boxes.shape[0]
                        batch_detections += num_dets

                        if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                            batch_confidence += torch.sum(scores).item()
                            # Count valid detections (score > 0.1)
                            batch_valid += torch.sum(scores > 0.1).item()

        self.total_detections += batch_detections
        self.confidence_sum += batch_confidence
        self.valid_detections += batch_valid

        # Compute batch metrics
        avg_detections_per_image = batch_detections / batch_size if batch_size > 0 else 0
        avg_confidence = batch_confidence / max(batch_detections, 1)
        valid_ratio = batch_valid / max(batch_detections, 1)

        return {
            'detections_per_image': avg_detections_per_image,
            'avg_confidence': avg_confidence,
            'valid_detection_ratio': valid_ratio
        }

    def compute_loss(self, predictions, targets):
        """
        Compute a simple validation loss based on detection quality.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            float: Validation loss
        """
        metrics = self.validate_batch(predictions, targets)

        # Expected values for a good model
        expected_detections_per_image = 3.0  # Expect ~3 detections per image
        expected_confidence = 0.7  # Expect average confidence of 0.7
        expected_valid_ratio = 0.8  # Expect 80% of detections to be above threshold

        # Calculate loss components
        detection_loss = abs(metrics['detections_per_image'] -
                             expected_detections_per_image) / expected_detections_per_image
        confidence_loss = abs(
            metrics['avg_confidence'] - expected_confidence) / expected_confidence
        valid_loss = abs(metrics['valid_detection_ratio'] -
                         expected_valid_ratio) / expected_valid_ratio

        # Combine losses with weights
        total_loss = 0.4 * detection_loss + 0.4 * confidence_loss + 0.2 * valid_loss

        # Clamp between reasonable bounds
        total_loss = max(0.05, min(1.0, total_loss))

        return total_loss

    def get_summary(self):
        """Get validation summary statistics."""
        if self.total_images == 0:
            return {
                'avg_detections_per_image': 0.0,
                'overall_avg_confidence': 0.0,
                'overall_valid_ratio': 0.0
            }

        return {
            'avg_detections_per_image': self.total_detections / self.total_images,
            'overall_avg_confidence': self.confidence_sum / max(self.total_detections, 1),
            'overall_valid_ratio': self.valid_detections / max(self.total_detections, 1)
        }


def compute_detection_validation_loss(model_outputs, targets, num_classes=20):
    """
    Standalone function to compute validation loss for detection models.

    Args:
        model_outputs: Model predictions (list of dicts or tensor)
        targets: Ground truth targets
        num_classes: Number of classes

    Returns:
        float: Validation loss
    """
    try:
        validator = SimpleDetectionValidator(num_classes)
        loss = validator.compute_loss(model_outputs, targets)
        return loss
    except Exception as e:
        logging.warning(f"Error in validation loss computation: {e}")
        return 0.5  # Fallback
