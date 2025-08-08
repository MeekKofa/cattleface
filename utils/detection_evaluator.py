"""
Object Detection Evaluation Module
Following COCO evaluation standards for mAP calculation
"""

import torch
import numpy as np
from collections import defaultdict
import logging


class DetectionEvaluator:
    """
    Evaluates object detection models using mAP metrics following COCO standards.
    """

    def __init__(self, num_classes, iou_thresholds=None, device='cpu'):
        """
        Initialize the evaluator.

        Args:
            num_classes (int): Number of object classes
            iou_thresholds (list): IoU thresholds for mAP calculation. 
                                 Default: [0.5:0.95:0.05] (COCO standard)
            device (str): Device for computations
        """
        self.num_classes = num_classes
        self.device = device

        if iou_thresholds is None:
            # COCO standard: IoU from 0.5 to 0.95 with step 0.05
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)

        self.reset()

    def reset(self):
        """Reset all stored predictions and ground truth."""
        self.predictions = []
        self.ground_truths = []

    def add_batch(self, predictions, targets):
        """
        Add a batch of predictions and targets.

        Args:
            predictions (list): List of prediction dicts, one per image
                Each dict should have keys: 'boxes', 'scores', 'labels'
            targets (dict or list): Ground truth targets
        """
        # Handle case where targets is a dict with batch dimension
        if isinstance(targets, dict):
            # Convert single target dict to list format
            batch_size = len(predictions)
            targets_list = []
            for i in range(batch_size):
                target = {}
                for key, value in targets.items():
                    if isinstance(value, torch.Tensor):
                        if len(value.shape) > 1 and value.shape[0] >= batch_size:
                            # Multi-dimensional tensor with batch dimension
                            target[key] = value[i] if i < value.shape[0] else torch.empty(
                                0, dtype=value.dtype, device=value.device)
                        elif len(value.shape) == 1 and value.shape[0] >= batch_size:
                            # 1D tensor with batch dimension
                            target[key] = value[i:i+1] if i < value.shape[0] else torch.empty(
                                0, dtype=value.dtype, device=value.device)
                        else:
                            # Shared tensor for all samples in batch
                            target[key] = value
                    elif isinstance(value, list) and len(value) >= batch_size:
                        # List with batch dimension
                        target[key] = value[i] if i < len(value) else []
                    else:
                        # Scalar or shared value
                        target[key] = value
                targets_list.append(target)
            targets = targets_list
        elif isinstance(targets, list):
            # Already in list format
            pass
        else:
            # Convert other formats to list
            targets = [targets] * len(predictions)

        # Ensure we have the same number of predictions and targets
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]

        for pred, gt in zip(predictions, targets):
            self.predictions.append(self._format_prediction(pred))
            self.ground_truths.append(self._format_ground_truth(gt))

    def _format_prediction(self, pred):
        """Format prediction to standard format."""
        if isinstance(pred, dict):
            boxes = pred.get('boxes', torch.empty(0, 4))
            scores = pred.get('scores', torch.empty(0))
            labels = pred.get('labels', torch.empty(0))
        else:
            # Handle case where pred is not a dict
            boxes = torch.empty(0, 4)
            scores = torch.empty(0)
            labels = torch.empty(0)

        # Ensure tensors are on CPU and convert to numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        return {
            'boxes': boxes.reshape(-1, 4) if boxes.size > 0 else np.empty((0, 4)),
            'scores': scores.flatten() if scores.size > 0 else np.empty(0),
            'labels': labels.flatten().astype(int) if labels.size > 0 else np.empty(0, dtype=int)
        }

    def _format_ground_truth(self, gt):
        """Format ground truth to standard format."""
        if isinstance(gt, dict):
            boxes = gt.get('boxes', torch.empty(0, 4))
            labels = gt.get('labels', torch.empty(0))
        else:
            boxes = torch.empty(0, 4)
            labels = torch.empty(0)

        # Ensure tensors are on CPU and convert to numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        elif isinstance(boxes, (list, tuple)):
            boxes = np.array(boxes)

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, (list, tuple)):
            labels = np.array(labels)

        # Handle empty cases
        if not isinstance(boxes, np.ndarray):
            boxes = np.empty((0, 4))
        if not isinstance(labels, np.ndarray):
            labels = np.empty(0, dtype=int)

        return {
            'boxes': boxes.reshape(-1, 4) if boxes.size > 0 else np.empty((0, 4)),
            'labels': labels.flatten().astype(int) if labels.size > 0 else np.empty(0, dtype=int)
        }

    def compute_iou(self, box1, box2):
        """
        Compute IoU between two bounding boxes.

        Args:
            box1, box2: numpy arrays of shape (4,) in format [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        # Intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Intersection area
        intersection = (x2 - x1) * (y2 - y1)

        # Union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def compute_ap(self, recalls, precisions):
        """
        Compute Average Precision using the COCO methodology.

        Args:
            recalls: numpy array of recall values
            precisions: numpy array of precision values

        Returns:
            float: Average Precision
        """
        # Add sentinel values at the beginning and end
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Compute AP using 101-point interpolation (COCO standard)
        ap = 0.0
        for t in np.arange(0., 1.01, 0.01):
            if np.sum(mrec >= t) == 0:
                p = 0
            else:
                p = np.max(mpre[mrec >= t])
            ap += p / 101.0

        return ap

    def evaluate_class(self, class_id, iou_threshold):
        """
        Evaluate a specific class at a specific IoU threshold.

        Args:
            class_id (int): Class to evaluate
            iou_threshold (float): IoU threshold

        Returns:
            float: Average Precision for this class at this IoU threshold
        """
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_ground_truths = []

        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            # Get predictions for this class
            pred_mask = pred['labels'] == class_id
            if np.sum(pred_mask) > 0:
                for j in np.where(pred_mask)[0]:
                    class_predictions.append({
                        'image_id': i,
                        'box': pred['boxes'][j],
                        'score': pred['scores'][j]
                    })

            # Get ground truths for this class
            gt_mask = gt['labels'] == class_id
            if np.sum(gt_mask) > 0:
                for j in np.where(gt_mask)[0]:
                    class_ground_truths.append({
                        'image_id': i,
                        'box': gt['boxes'][j],
                        'matched': False
                    })

        if len(class_predictions) == 0:
            return 0.0

        if len(class_ground_truths) == 0:
            return 0.0

        # Sort predictions by confidence score (descending)
        class_predictions.sort(key=lambda x: x['score'], reverse=True)

        # Match predictions to ground truths
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))

        for i, pred in enumerate(class_predictions):
            # Find ground truths in the same image
            gt_in_image = [
                gt for gt in class_ground_truths if gt['image_id'] == pred['image_id']]

            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(gt_in_image):
                if gt['matched']:
                    continue

                iou = self.compute_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # Check if prediction matches a ground truth
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                gt_in_image[best_gt_idx]['matched'] = True
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(class_ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Compute AP
        ap = self.compute_ap(recalls, precisions)
        return ap

    def compute_map(self):
        """
        Compute mean Average Precision (mAP) across all classes and IoU thresholds.

        Returns:
            dict: Dictionary containing various mAP metrics
        """
        if len(self.predictions) == 0:
            logging.warning("No predictions to evaluate")
            return {
                'mAP': 0.0,
                'mAP@0.5': 0.0,
                'mAP@0.75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0,
                'per_class_ap': [0.0] * self.num_classes
            }

        # Compute AP for each class and IoU threshold
        aps = np.zeros((len(self.iou_thresholds), self.num_classes))

        for iou_idx, iou_thresh in enumerate(self.iou_thresholds):
            for class_id in range(self.num_classes):
                aps[iou_idx, class_id] = self.evaluate_class(
                    class_id, iou_thresh)

        # Compute mAP metrics
        # mAP averaged over IoU [0.5:0.95] and all classes
        map_all = np.mean(aps)
        map_50 = np.mean(aps[0])  # mAP at IoU=0.5
        map_75_idx = np.argmin(np.abs(self.iou_thresholds - 0.75))
        map_75 = np.mean(aps[map_75_idx]) if map_75_idx < len(
            self.iou_thresholds) else 0.0

        # Per-class AP (averaged over IoU thresholds)
        per_class_ap = np.mean(aps, axis=0)

        return {
            'mAP': float(map_all),
            'mAP@0.5': float(map_50),
            'mAP@0.75': float(map_75),
            'mAP_small': 0.0,  # Placeholder - would need area analysis
            'mAP_medium': 0.0,  # Placeholder - would need area analysis
            'mAP_large': 0.0,  # Placeholder - would need area analysis
            'per_class_ap': per_class_ap.tolist()
        }

    def compute_detection_loss(self, predictions, targets):
        """
        Compute a proper detection loss based on predictions and targets.

        Args:
            predictions (list): List of prediction dicts
            targets (dict or list): Ground truth targets

        Returns:
            float: Detection loss value
        """
        try:
            # Temporarily store current state
            old_predictions = self.predictions.copy()
            old_ground_truths = self.ground_truths.copy()

            # Reset and add current batch
            self.reset()
            self.add_batch(predictions, targets)

            # Check if we have valid data
            if len(self.predictions) == 0 or len(self.ground_truths) == 0:
                logging.warning(
                    "No valid predictions or ground truths for loss calculation")
                return 0.5

            # Compute mAP as a proxy for detection quality
            metrics = self.compute_map()
            map_score = metrics['mAP']

            # Convert mAP to loss (higher mAP = lower loss)
            # Loss = 1 - mAP, with some smoothing
            loss = max(0.1, 1.0 - map_score)

            # Restore previous state
            self.predictions = old_predictions
            self.ground_truths = old_ground_truths

            return loss

        except Exception as e:
            logging.warning(f"Error computing detection loss: {e}")
            return 0.5  # Fallback loss
