"""
Training diagnostics and visualization utilities for better training monitoring.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os
import json
from datetime import datetime


class TrainingDiagnostics:
    """Comprehensive training diagnostics and monitoring."""

    def __init__(self, save_dir='training_diagnostics', window_size=100):
        self.save_dir = save_dir
        self.window_size = window_size
        os.makedirs(save_dir, exist_ok=True)

        # Metrics tracking
        self.metrics_history = defaultdict(list)
        self.recent_metrics = defaultdict(lambda: deque(maxlen=window_size))

        # Loss component tracking
        self.loss_components = defaultdict(list)

        # Learning rate tracking
        self.lr_history = []

        # Gradient tracking
        self.gradient_stats = defaultdict(list)

    def log_metrics(self, epoch, metrics_dict):
        """Log training metrics for an epoch."""
        timestamp = datetime.now().isoformat()

        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            self.metrics_history[key].append({
                'epoch': epoch,
                'value': value,
                'timestamp': timestamp
            })
            self.recent_metrics[key].append(value)

    def log_loss_components(self, epoch, box_loss, obj_loss, cls_loss, total_loss):
        """Log individual loss components."""
        components = {
            'box_loss': box_loss.item() if isinstance(box_loss, torch.Tensor) else box_loss,
            'obj_loss': obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss,
            'cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        }

        for key, value in components.items():
            self.loss_components[key].append({
                'epoch': epoch,
                'value': value
            })

    def log_learning_rate(self, epoch, lr):
        """Log learning rate for an epoch."""
        self.lr_history.append({
            'epoch': epoch,
            'lr': lr
        })

    def log_gradient_stats(self, epoch, model):
        """Log gradient statistics."""
        total_norm = 0
        param_count = 0
        grad_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                grad_norms.append(param_norm.item())

        total_norm = total_norm ** (1. / 2)

        self.gradient_stats['total_norm'].append({
            'epoch': epoch,
            'value': total_norm
        })

        if grad_norms:
            self.gradient_stats['mean_norm'].append({
                'epoch': epoch,
                'value': np.mean(grad_norms)
            })
            self.gradient_stats['max_norm'].append({
                'epoch': epoch,
                'value': np.max(grad_norms)
            })

    def get_recent_trend(self, metric_name, num_points=10):
        """Get recent trend for a metric (improving, degrading, stable)."""
        if metric_name not in self.recent_metrics:
            return "no_data"

        recent_values = list(self.recent_metrics[metric_name])
        if len(recent_values) < num_points:
            return "insufficient_data"

        # Calculate trend
        recent_values = recent_values[-num_points:]
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]

        if abs(slope) < 0.001:  # Threshold for stability
            return "stable"
        elif slope < 0:
            return "improving" if "loss" in metric_name.lower() else "degrading"
        else:
            return "degrading" if "loss" in metric_name.lower() else "improving"

    def plot_training_curves(self, save_path=None):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Loss components
        if self.loss_components:
            ax = axes[0, 0]
            for loss_type in ['total_loss', 'box_loss', 'obj_loss', 'cls_loss']:
                if loss_type in self.loss_components:
                    epochs = [item['epoch']
                              for item in self.loss_components[loss_type]]
                    values = [item['value']
                              for item in self.loss_components[loss_type]]
                    ax.plot(epochs, values, label=loss_type,
                            marker='o', markersize=2)
            ax.set_title('Loss Components')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)

        # Learning rate
        if self.lr_history:
            ax = axes[0, 1]
            epochs = [item['epoch'] for item in self.lr_history]
            lrs = [item['lr'] for item in self.lr_history]
            ax.plot(epochs, lrs, 'g-', marker='o', markersize=2)
            ax.set_title('Learning Rate Schedule')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.grid(True)

        # Gradient norms
        if self.gradient_stats:
            ax = axes[0, 2]
            for stat_type in ['total_norm', 'mean_norm', 'max_norm']:
                if stat_type in self.gradient_stats:
                    epochs = [item['epoch']
                              for item in self.gradient_stats[stat_type]]
                    values = [item['value']
                              for item in self.gradient_stats[stat_type]]
                    ax.plot(epochs, values, label=stat_type,
                            marker='o', markersize=2)
            ax.set_title('Gradient Statistics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.legend()
            ax.grid(True)

        # mAP metrics
        metrics_to_plot = ['map_50', 'map', 'precision', 'recall']
        for i, metric in enumerate(metrics_to_plot):
            if i < 3 and metric in self.metrics_history:  # We have 3 remaining subplots
                ax = axes[1, i]
                epochs = [item['epoch']
                          for item in self.metrics_history[metric]]
                values = [item['value']
                          for item in self.metrics_history[metric]]
                ax.plot(epochs, values, 'b-', marker='o', markersize=2)
                ax.set_title(f'{metric.upper()}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(
                self.save_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def generate_training_report(self):
        """Generate a comprehensive training report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_epochs': len(self.lr_history) if self.lr_history else 0,
            'metrics_summary': {},
            'trends': {},
            'recommendations': []
        }

        # Summarize metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                values = [item['value'] for item in history]
                report['metrics_summary'][metric_name] = {
                    'current': values[-1],
                    'best': max(values) if 'map' in metric_name or 'precision' in metric_name or 'recall' in metric_name else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': self.get_recent_trend(metric_name)
                }

        # Add recommendations based on trends
        if 'total_loss' in self.metrics_history:
            loss_trend = self.get_recent_trend('total_loss')
            if loss_trend == "stable":
                report['recommendations'].append(
                    "Loss is stable - consider adjusting learning rate or trying different optimization strategies")
            elif loss_trend == "degrading":
                report['recommendations'].append(
                    "Loss is increasing - consider reducing learning rate or checking for overfitting")

        if 'map_50' in self.metrics_history:
            map_values = [item['value']
                          for item in self.metrics_history['map_50']]
            if all(v == 0 for v in map_values[-5:]):  # Last 5 epochs are 0
                report['recommendations'].append(
                    "mAP@0.5 is consistently 0 - check anchor sizes, detection thresholds, or label format")

        # Save report
        report_path = os.path.join(
            self.save_dir, f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report, report_path

    def print_training_summary(self):
        """Print a concise training summary."""
        print("\n" + "="*50)
        print("TRAINING DIAGNOSTICS SUMMARY")
        print("="*50)

        # Recent metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                current = history[-1]['value']
                trend = self.get_recent_trend(metric_name)
                print(f"{metric_name:15s}: {current:8.4f} (trend: {trend})")

        # Recent loss components
        if self.loss_components:
            print("\nLoss Components:")
            for loss_type in ['total_loss', 'box_loss', 'obj_loss', 'cls_loss']:
                if loss_type in self.loss_components and self.loss_components[loss_type]:
                    current = self.loss_components[loss_type][-1]['value']
                    print(f"  {loss_type:12s}: {current:8.4f}")

        print("="*50)


def analyze_dataset_statistics(dataset_loader):
    """Analyze dataset statistics for training optimization."""
    print("Analyzing dataset statistics...")

    box_sizes = []
    aspect_ratios = []
    class_counts = defaultdict(int)

    for batch_idx, (images, targets) in enumerate(dataset_loader):
        if batch_idx >= 100:  # Sample first 100 batches
            break

        for target in targets:
            boxes = target['boxes']
            labels = target['labels']

            if len(boxes) > 0:
                # Box sizes (width * height)
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                areas = widths * heights
                box_sizes.extend(areas.tolist())

                # Aspect ratios
                ratios = widths / heights
                aspect_ratios.extend(ratios.tolist())

                # Class distribution
                for label in labels:
                    class_counts[label.item()] += 1

    if box_sizes:
        print(f"\nDataset Statistics:")
        print(f"  Total boxes analyzed: {len(box_sizes)}")
        print(
            f"  Box area - Mean: {np.mean(box_sizes):.4f}, Std: {np.std(box_sizes):.4f}")
        print(
            f"  Box area - Min: {np.min(box_sizes):.4f}, Max: {np.max(box_sizes):.4f}")
        print(
            f"  Aspect ratio - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")
        print(
            f"  Most common classes: {dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")

        # Recommendations
        mean_area = np.mean(box_sizes)
        if mean_area < 0.01:
            print("  ⚠️  Small objects detected - consider increasing input resolution")
        if np.std(aspect_ratios) > 2.0:
            print("  ⚠️  High aspect ratio variance - consider data augmentation")

    return {
        'box_sizes': box_sizes,
        'aspect_ratios': aspect_ratios,
        'class_counts': class_counts
    }


def quick_model_health_check(model, data_loader, device, max_batches=3):
    """Quick health check to detect identical predictions issue"""
    print("\n🏥 QUICK MODEL HEALTH CHECK")
    print("=" * 50)

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= max_batches:
                break

            if isinstance(images, list):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)

            outputs = model(images)

            if isinstance(outputs, list):
                for output in outputs:
                    if 'boxes' in output and len(output['boxes']) > 0:
                        all_predictions.append({
                            'boxes': output['boxes'].cpu().numpy(),
                            'scores': output['scores'].cpu().numpy(),
                            'labels': output['labels'].cpu().numpy()
                        })

    if not all_predictions:
        print("❌ No predictions found - model may not be detecting objects")
        return False

    # Check for identical predictions
    if len(all_predictions) > 1:
        first_pred = all_predictions[0]
        identical_count = 0

        for pred in all_predictions[1:]:
            if (np.allclose(first_pred['boxes'], pred['boxes'], atol=1e-4) and
                    np.allclose(first_pred['scores'], pred['scores'], atol=1e-4)):
                identical_count += 1

        if identical_count > len(all_predictions) * 0.7:  # >70% identical
            print(
                f"❌ IDENTICAL PREDICTIONS DETECTED! {identical_count}/{len(all_predictions)} predictions are identical")
            print(
                f"   Sample prediction: boxes={first_pred['boxes'][:2]}, scores={first_pred['scores'][:2]}")
            return False
        else:
            print(
                f"✅ Good prediction diversity: {identical_count}/{len(all_predictions)} identical predictions")

    # Check prediction quality
    avg_confidence = np.mean([pred['scores'].max() if len(pred['scores']) > 0 else 0
                             for pred in all_predictions])
    unique_classes = len(set([label for pred in all_predictions
                              for label in pred['labels']]))

    print(f"📊 Model Health Summary:")
    print(f"   • Predictions found: {len(all_predictions)}")
    print(f"   • Average max confidence: {avg_confidence:.4f}")
    print(f"   • Unique classes predicted: {unique_classes}")

    if avg_confidence < 0.1:
        print("   ⚠️  Very low confidence scores detected")
        return False
    elif avg_confidence > 0.3:
        print("   ✅ Good confidence levels")
        return True
    else:
        print("   ⚠️  Moderate confidence levels")
        return True


def check_identical_predictions_simple(model, data_loader, device):
    """Simple check for identical predictions without external dependencies"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= 2:  # Only check first 2 batches
                break

            if isinstance(images, list):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, list) and len(outputs) > 0:
                first_output = outputs[0]
                if 'boxes' in first_output:
                    predictions.append({
                        'boxes': first_output['boxes'].cpu(),
                        'scores': first_output['scores'].cpu(),
                        'labels': first_output['labels'].cpu()
                    })

    # Simple identical check
    if len(predictions) > 1:
        first = predictions[0]
        second = predictions[1]
        if (torch.allclose(first['boxes'], second['boxes'], atol=1e-4) and
                torch.allclose(first['scores'], second['scores'], atol=1e-4)):
            return True  # Identical predictions detected

    return False  # Predictions are different
