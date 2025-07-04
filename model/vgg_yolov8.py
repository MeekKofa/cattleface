import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision.models import vgg16, vgg16_bn
from torchvision.ops import box_iou


class VGG_YOLOv8(nn.Module):
    def __init__(self, num_classes, input_channels=3, dropout_rate=0.5, pretrained=True):
        super(VGG_YOLOv8, self).__init__()
        # Use pretrained VGG16 as backbone
        if pretrained:
            self.backbone = vgg16_bn(pretrained=True).features
        else:
            self.backbone = vgg16_bn(pretrained=False).features

        # Modify first layer if needed
        if input_channels != 3:
            self.backbone[0] = nn.Conv2d(
                input_channels, 64, kernel_size=3, padding=1)

        # Feature pyramid network
        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),  # Reduce channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Process features
        ])

        # Prediction head
        self.dropout = nn.Dropout(dropout_rate)
        # 4 values for bbox (x, y, w, h)
        self.bbox_head = nn.Conv2d(256, 4, kernel_size=1)
        self.cls_head = nn.Conv2d(
            256, num_classes, kernel_size=1)  # Class predictions

        self.num_classes = num_classes

    def forward(self, x, targets=None):
        # Feature extraction
        features = self.backbone(x)

        # Feature pyramid
        fpn_features = features
        for layer in self.fpn_layers:
            fpn_features = layer(fpn_features)

        # Apply dropout
        fpn_features = self.dropout(fpn_features)

        # Prediction heads
        bbox_pred = self.bbox_head(fpn_features)
        cls_pred = self.cls_head(fpn_features)

        # Reshape predictions
        batch_size = x.shape[0]
        bbox_pred = bbox_pred.permute(
            0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1, self.num_classes)

        # Apply sigmoid to get box coordinates in [0,1]
        bbox_pred = torch.sigmoid(bbox_pred)

        # Training or inference mode
        if self.training and targets is not None:
            # Calculate loss
            loss_dict = self.compute_loss(bbox_pred, cls_pred, targets)
            return loss_dict
        else:
            # Return predictions
            return self.make_detections(bbox_pred, cls_pred)

    def compute_loss(self, bbox_pred, cls_pred, targets):
        """Compute detection loss"""
        batch_size = bbox_pred.shape[0]
        loss_bbox = torch.tensor(0.0, device=bbox_pred.device)
        loss_cls = torch.tensor(0.0, device=cls_pred.device)
        num_positive_samples = 0

        for i in range(batch_size):
            target_boxes = targets[i]["boxes"]
            target_labels = targets[i]["labels"]

            # Skip if no objects in this image
            if len(target_boxes) == 0 or target_boxes.numel() == 0:
                continue

            # Ensure target_boxes has the correct shape
            if target_boxes.dim() == 1:
                target_boxes = target_boxes.reshape(-1, 4)
            elif target_boxes.dim() == 0:
                continue

            # Ensure we have valid predictions for this image
            if bbox_pred[i].numel() == 0:
                continue

            # Ensure predictions have correct shape
            pred_boxes = bbox_pred[i]
            pred_classes = cls_pred[i]

            if pred_boxes.dim() == 1:
                pred_boxes = pred_boxes.reshape(-1, 4)
            if pred_classes.dim() == 1:
                pred_classes = pred_classes.reshape(-1, self.num_classes)

            # Skip if either tensor is empty after reshaping
            if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
                continue

            try:
                # Compute IoU between predicted and target boxes
                iou = box_iou(pred_boxes, target_boxes)

                # Assign targets to predictions based on highest IoU
                max_iou, matched_idx = torch.max(iou, dim=1)
                positive_mask = max_iou > 0.5

                if positive_mask.sum() > 0:
                    # Ensure matched indices are valid
                    valid_matches = matched_idx[positive_mask]
                    valid_matches = valid_matches[valid_matches < len(
                        target_labels)]

                    if len(valid_matches) > 0:
                        # Update positive mask to only include valid matches
                        valid_positive_mask = torch.zeros_like(positive_mask)
                        valid_indices = torch.where(positive_mask)[0]
                        keep_indices = valid_indices[:len(valid_matches)]
                        valid_positive_mask[keep_indices] = True

                        # Classification loss (only for positive matches)
                        matched_targets = target_labels[valid_matches]
                        matched_targets = matched_targets.clamp(
                            0, self.num_classes - 1)  # Ensure valid class indices
                        loss_cls += F.cross_entropy(
                            pred_classes[valid_positive_mask], matched_targets)

                        # Regression loss (only for positive matches)
                        matched_boxes = target_boxes[valid_matches]
                        loss_bbox += F.smooth_l1_loss(
                            pred_boxes[valid_positive_mask], matched_boxes)

                        num_positive_samples += valid_matches.shape[0]

            except Exception as e:
                # Log the error and continue with next image
                logging.warning(f"Error computing loss for image {i}: {e}")
                logging.warning(
                    f"Pred boxes shape: {pred_boxes.shape}, Target boxes shape: {target_boxes.shape}")
                continue

        # Normalize losses by number of positive samples or batch size
        normalizer = max(num_positive_samples, 1)
        loss_dict = {
            'loss_cls': loss_cls / normalizer,
            'loss_bbox': loss_bbox / normalizer,
            'loss': (loss_cls + loss_bbox) / normalizer
        }

        return loss_dict

    def make_detections(self, bbox_pred, cls_pred):
        """Convert raw predictions to detections"""
        batch_size = bbox_pred.shape[0]
        results = []

        for i in range(batch_size):
            # Get class scores and predictions
            scores, labels = torch.max(F.softmax(cls_pred[i], dim=1), dim=1)
            boxes = bbox_pred[i]

            # Filter by confidence
            keep = scores > 0.05
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            results.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            })

        return results


def get_vgg_yolov8(num_classes, input_channels=3, depth=16, dropout_rate=0.5):
    """Factory function to create VGG YOLOv8 model"""
    assert depth == 16, "Only VGG16 backbone is supported for YOLOv8"
    model = VGG_YOLOv8(num_classes, input_channels, dropout_rate)
    return model
