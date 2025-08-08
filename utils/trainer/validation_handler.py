"""
Validation Handler - specialized validation logic for different model types
"""
import torch
import logging
from torch.cuda.amp import autocast


class ValidationHandler:
    """Handles validation for different types of models"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        # Use a very low confidence threshold during early training
        self.confidence_threshold = getattr(
            config, 'confidence_threshold', 0.01)  # Lowered from 0.1 to 0.01
        logging.info(
            f"ValidationHandler using confidence_threshold={self.confidence_threshold}")

    def validate_object_detection(self, val_loader, criterion=None):
        """Validation specifically for object detection models"""
        self.model.eval()

        total_loss = 0
        num_batches = 0
        detection_stats = {
            'total_images': 0,
            'total_detections': 0,
            'total_confidence': 0.0,
            'valid_detections': 0,
            'boxes_with_area': 0,
            'avg_box_area': 0.0
        }

        logging.info("Starting detailed object detection validation...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                try:
                    images, targets = self._parse_batch(batch_data)
                    images, targets = self._move_to_device(images, targets)

                    # Get model outputs (inference mode - no targets)
                    self.model.eval()
                    logging.info(
                        f"DEBUG: About to call model.forward with images shape: {images.shape}")
                    outputs = self.model(images, None)
                    logging.info(
                        f"DEBUG: Model returned outputs type: {type(outputs)}, length: {len(outputs) if isinstance(outputs, list) else 'N/A'}")

                    # Count images
                    batch_size = self._get_batch_size(images)
                    detection_stats['total_images'] += batch_size

                    # Analyze detections
                    batch_detection_stats = self._analyze_detection_outputs(
                        outputs)
                    self._update_detection_stats(
                        detection_stats, batch_detection_stats)

                    # Compute loss if possible
                    logging.info(f"DEBUG: About to compute validation loss")
                    loss = self._compute_validation_loss(outputs, targets)
                    logging.info(f"DEBUG: Computed validation loss: {loss}")
                    total_loss += loss
                    num_batches += 1

                    # Log progress for first few batches
                    if batch_idx < 5:
                        logging.debug(
                            f"Batch {batch_idx}: {batch_detection_stats}")

                except Exception as e:
                    logging.error(
                        f"Error in validation batch {batch_idx}: {e}")
                    continue

        # Compute final metrics
        metrics = self._compute_detection_metrics(
            detection_stats, total_loss, num_batches)
        self._log_validation_results(metrics)

        return metrics['loss'], metrics

    def _parse_batch(self, batch_data):
        """Parse batch data into images and targets"""
        if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
            return batch_data[0], batch_data[1]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch_data)}")

    def _move_to_device(self, images, targets):
        """Move data to device"""
        # Handle images
        if isinstance(images, list):
            images = [img.to(self.device, non_blocking=True) for img in images]
        elif isinstance(images, torch.Tensor):
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

    def _get_batch_size(self, images):
        """Get batch size from images"""
        if isinstance(images, torch.Tensor):
            return images.shape[0]
        elif isinstance(images, list):
            return len(images)
        else:
            return 1

    def _analyze_detection_outputs(self, outputs):
        """Analyze detection outputs and return statistics"""
        logging.info(
            f"DEBUG: _analyze_detection_outputs called with outputs type: {type(outputs)}")
        if isinstance(outputs, list):
            logging.info(f"DEBUG: Number of predictions: {len(outputs)}")
            for i, pred in enumerate(outputs):
                if isinstance(pred, dict):
                    boxes = pred.get('boxes', torch.empty(0, 4))
                    scores = pred.get('scores', torch.empty(0))
                    labels = pred.get('labels', torch.empty(0))
                    logging.info(
                        f"DEBUG: Prediction {i} - boxes: {boxes.shape}, scores: {scores.shape}, labels: {labels.shape}")
                    if scores.numel() > 0:
                        logging.info(f"DEBUG: Score values: {scores}")
                        logging.info(
                            f"DEBUG: Confidence threshold: {self.confidence_threshold}")
                        valid_count = torch.sum(
                            scores > self.confidence_threshold).item()
                        logging.info(
                            f"DEBUG: Valid detections (>{self.confidence_threshold}): {valid_count}")

        stats = {
            'detections': 0,
            'confidence_sum': 0.0,
            'valid_detections': 0,
            'boxes_with_area': 0,
            'total_box_area': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 1.0,
            'has_predictions': False
        }

        if not isinstance(outputs, list):
            logging.warning(f"Expected list of outputs, got {type(outputs)}")
            return stats

        for i, pred in enumerate(outputs):
            if not isinstance(pred, dict):
                logging.warning(
                    f"Expected dict output, got {type(pred)} for prediction {i}")
                continue

            boxes = pred.get('boxes', torch.empty(0, 4))
            scores = pred.get('scores', torch.empty(0))
            labels = pred.get('labels', torch.empty(0))

            if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                num_dets = boxes.shape[0]
                stats['detections'] += num_dets
                stats['has_predictions'] = True

                # Analyze box areas
                if boxes.shape[1] >= 4:  # x1, y1, x2, y2 format
                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    areas = widths * heights
                    valid_areas = areas > 0
                    stats['boxes_with_area'] += torch.sum(valid_areas).item()
                    stats['total_box_area'] += torch.sum(
                        areas[valid_areas]).item()

                # Analyze confidence scores
                if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                    stats['confidence_sum'] += torch.sum(scores).item()
                    stats['valid_detections'] += torch.sum(
                        scores > self.confidence_threshold).item()
                    stats['max_confidence'] = max(
                        stats['max_confidence'], torch.max(scores).item())
                    stats['min_confidence'] = min(
                        stats['min_confidence'], torch.min(scores).item())

        return stats

    def _update_detection_stats(self, total_stats, batch_stats):
        """Update total detection statistics with batch statistics"""
        total_stats['total_detections'] += batch_stats['detections']
        total_stats['total_confidence'] += batch_stats['confidence_sum']
        total_stats['valid_detections'] += batch_stats['valid_detections']
        total_stats['boxes_with_area'] += batch_stats['boxes_with_area']
        total_stats['avg_box_area'] += batch_stats['total_box_area']

    def _compute_validation_loss(self, outputs, targets):
        """Compute validation loss"""
        logging.info(
            f"DEBUG: _compute_validation_loss called with outputs type: {type(outputs)}")
        try:
            if hasattr(self.model, 'compute_loss'):
                logging.info(
                    "DEBUG: Model has compute_loss method, calling it")
                loss = self.model.compute_loss(outputs, targets)
                logging.info(
                    f"DEBUG: compute_loss returned: {loss}, type: {type(loss)}")
                if isinstance(loss, torch.Tensor):
                    return loss.item()
                else:
                    return float(loss)
            else:
                logging.warning(
                    "DEBUG: Model does not have compute_loss method")
                # Fallback loss
                return 0.5
        except Exception as e:
            logging.error(f"DEBUG: Exception in _compute_validation_loss: {e}")
            import traceback
            logging.error(f"DEBUG: Traceback: {traceback.format_exc()}")
            return 0.5

    def _compute_detection_metrics(self, stats, total_loss, num_batches):
        """Compute final detection metrics"""
        avg_loss = total_loss / max(num_batches, 1)

        # Avoid division by zero
        total_images = max(stats['total_images'], 1)
        total_detections = max(stats['total_detections'], 1)

        avg_detections_per_image = stats['total_detections'] / total_images
        avg_confidence = stats['total_confidence'] / total_detections
        valid_detection_ratio = stats['valid_detections'] / total_detections
        avg_box_area = stats['avg_box_area'] / max(stats['boxes_with_area'], 1)

        # Compute quality scores
        # Expecting ~3 detections per image
        detection_quality = min(1.0, avg_detections_per_image / 3.0)
        confidence_quality = avg_confidence  # Already 0-1
        valid_quality = valid_detection_ratio  # Already 0-1

        # Overall quality score
        overall_quality = 0.4 * detection_quality + \
            0.4 * confidence_quality + 0.2 * valid_quality

        return {
            'loss': avg_loss,
            'accuracy': overall_quality,
            'detections_per_image': avg_detections_per_image,
            'avg_confidence': avg_confidence,
            'valid_detection_ratio': valid_detection_ratio,
            'avg_box_area': avg_box_area,
            'total_images': stats['total_images'],
            'total_detections': stats['total_detections']
        }

    def _log_validation_results(self, metrics):
        """Log validation results"""
        logging.info("=" * 50)
        logging.info("VALIDATION RESULTS")
        logging.info("=" * 50)
        logging.info(f"Loss: {metrics['loss']:.4f}")
        logging.info(f"Overall Quality Score: {metrics['accuracy']:.4f}")
        logging.info(f"Total Images: {metrics['total_images']}")
        logging.info(f"Total Detections: {metrics['total_detections']}")
        logging.info(
            f"Avg Detections/Image: {metrics['detections_per_image']:.2f}")
        logging.info(f"Avg Confidence: {metrics['avg_confidence']:.3f}")
        logging.info(
            f"Valid Detection Ratio: {metrics['valid_detection_ratio']:.3f}")
        logging.info(f"Avg Box Area: {metrics['avg_box_area']:.2f}")
        logging.info("=" * 50)

    def diagnose_model_outputs(self, val_loader, num_batches=3):
        """Diagnose model outputs for debugging"""
        logging.info("=" * 60)
        logging.info("DIAGNOSING MODEL OUTPUTS")
        logging.info("=" * 60)

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                try:
                    images, targets = self._parse_batch(batch_data)
                    images, targets = self._move_to_device(images, targets)

                    logging.info(f"\\nBatch {batch_idx + 1}:")
                    logging.info(f"  Images type: {type(images)}")

                    if isinstance(images, torch.Tensor):
                        logging.info(f"  Images shape: {images.shape}")
                    elif isinstance(images, list):
                        logging.info(f"  Images count: {len(images)}")
                        if len(images) > 0:
                            logging.info(
                                f"  First image shape: {images[0].shape}")

                    # Get model outputs
                    outputs = self.model(images, None)
                    logging.info(f"  Outputs type: {type(outputs)}")

                    if isinstance(outputs, list):
                        logging.info(
                            f"  Number of predictions: {len(outputs)}")
                        for i, pred in enumerate(outputs):
                            if isinstance(pred, dict):
                                boxes = pred.get('boxes', torch.empty(0))
                                scores = pred.get('scores', torch.empty(0))
                                labels = pred.get('labels', torch.empty(0))

                                logging.info(f"    Prediction {i}:")
                                logging.info(f"      Boxes: {boxes.shape}")
                                logging.info(f"      Scores: {scores.shape}")
                                logging.info(f"      Labels: {labels.shape}")

                                if boxes.numel() > 0:
                                    logging.info(
                                        f"      Box range: {boxes.min():.3f} to {boxes.max():.3f}")
                                if scores.numel() > 0:
                                    logging.info(
                                        f"      Score range: {scores.min():.3f} to {scores.max():.3f}")
                                    logging.info(
                                        f"      Max score: {scores.max():.3f}")

                except Exception as e:
                    logging.error(f"Error diagnosing batch {batch_idx}: {e}")

        logging.info("=" * 60)
