"""
Object Detection Metrics and Evaluation Tools
"""
import torch
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional, Tuple


class DetectionMetrics:
    """Class to compute and track object detection metrics"""

    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_predictions = 0
        self.total_targets = 0
        self.class_predictions = {i: 0 for i in range(self.num_classes)}
        self.class_targets = {i: 0 for i in range(self.num_classes)}
        self.losses = []

    def update(self, predictions: Dict, targets: Dict, loss: float = None):
        """Update metrics with batch predictions and targets"""
        if loss is not None:
            self.losses.append(loss)

        # Count predictions and targets
        if 'boxes' in predictions:
            self.total_predictions += len(predictions['boxes'])

        if 'boxes' in targets:
            self.total_targets += len(targets['boxes'])

        # Count per-class targets
        if 'labels' in targets:
            labels = targets['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            for label in labels:
                if 0 <= label < self.num_classes:
                    self.class_targets[label] += 1

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute final metrics"""
        metrics = {
            'total_predictions': self.total_predictions,
            'total_targets': self.total_targets,
            'avg_predictions_per_image': self.total_predictions / max(len(self.losses), 1),
            'avg_targets_per_image': self.total_targets / max(len(self.losses), 1),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'class_distribution': self.class_targets,
            'num_classes_with_targets': sum(1 for count in self.class_targets.values() if count > 0)
        }
        return metrics

    def save_metrics_to_file(self, filepath: str, model_name: str, dataset_name: str,
                             epoch: int = None, additional_info: Dict = None):
        """Save metrics to a text file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        metrics = self.compute_metrics()

        with open(filepath, 'w') as f:
            f.write(f"Object Detection Evaluation Results\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            if epoch is not None:
                f.write(f"Epoch: {epoch}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n\n")

            f.write(f"Overall Statistics:\n")
            f.write(f"- Total Predictions: {metrics['total_predictions']}\n")
            f.write(
                f"- Total Ground Truth Targets: {metrics['total_targets']}\n")
            f.write(
                f"- Average Predictions per Image: {metrics['avg_predictions_per_image']:.2f}\n")
            f.write(
                f"- Average Targets per Image: {metrics['avg_targets_per_image']:.2f}\n")
            f.write(f"- Average Loss: {metrics['avg_loss']:.4f}\n")
            f.write(
                f"- Number of Classes with Targets: {metrics['num_classes_with_targets']}/{self.num_classes}\n\n")

            f.write(f"Class Distribution (Ground Truth):\n")
            for class_id, count in metrics['class_distribution'].items():
                if count > 0:
                    f.write(f"- Class {class_id}: {count} instances\n")

            if additional_info:
                f.write(f"\nAdditional Information:\n")
                for key, value in additional_info.items():
                    f.write(f"- {key}: {value}\n")

            f.write(
                f"\nNote: This is a simplified evaluation. For comprehensive object detection\n")
            f.write(
                f"evaluation, implement proper mAP calculation with IoU matching.\n")

        logging.info(f"Detection metrics saved to: {filepath}")


def create_detection_evaluation_report(model_name: str, predictions: List, targets: List,
                                       losses: List, output_dir: str, dataset_name: str,
                                       num_classes: int = 20) -> str:
    """Create a comprehensive detection evaluation report"""

    metrics_calculator = DetectionMetrics(num_classes=num_classes)

    # Process all predictions and targets
    for pred, target, loss in zip(predictions, targets, losses):
        metrics_calculator.update(pred, target, loss)

    # Create output file path
    report_filename = f"{model_name}_{dataset_name}_detection_report.txt"
    report_path = os.path.join(output_dir, report_filename)

    # Additional information
    additional_info = {
        'total_batches_processed': len(predictions),
        'evaluation_type': 'Object Detection',
        'metrics_computed': 'Basic statistics (counts, losses, class distribution)'
    }

    # Save detailed report
    metrics_calculator.save_metrics_to_file(
        report_path, model_name, dataset_name,
        additional_info=additional_info
    )

    return report_path


def log_detection_summary(model_name: str, total_predictions: int, total_targets: int,
                          avg_loss: float, num_classes_with_targets: int, num_classes: int):
    """Log a summary of detection metrics"""
    logging.info(f"=== Detection Summary for {model_name} ===")
    logging.info(f"Total Predictions: {total_predictions}")
    logging.info(f"Total Ground Truth Targets: {total_targets}")
    logging.info(f"Average Loss: {avg_loss:.4f}")
    logging.info(
        f"Classes with Targets: {num_classes_with_targets}/{num_classes}")
    logging.info(
        f"Detection Rate: {total_predictions/max(total_targets, 1):.2f} predictions per target")
