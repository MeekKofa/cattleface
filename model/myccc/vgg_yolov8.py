import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from model.attention.base_robust_method import BaseRobustMethod


class SPP(nn.Module):
    """Spatial Pyramid Pooling layer"""

    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        outputs = [x]
        for pool_size in self.pool_sizes:
            pooled = F.max_pool2d(x, kernel_size=pool_size,
                                  stride=1, padding=pool_size//2)
            outputs
        return torch.cat(outputs, dim=1)


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer"""

    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(CSPLayer, self).__init__()
        hidden_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(2 * hidden_channels,
                               out_channels, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.blocks = nn.Sequential(*[
            self._make_block(hidden_channels) for _ in range(num_blocks)
        ])

    def _make_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, 0.1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, 0.1)

        x1 = self.blocks(x1)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.1)

        return x


class YOLOv8Head(nn.Module):
    """YOLOv8 Detection Head"""

    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super(YOLOv8Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # CSP-like structure
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Spatial Pyramid Pooling
        self.spp = SPP([5, 9, 13])
        spp_out_channels = 256 * 4  # original + 3 pooled

        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(spp_out_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # CSP layer
        self.csp = CSPLayer(256, 256, num_blocks=3)

        # Upsample path
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Downsample path
        self.downsample = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Multi-scale fusion
        self.multi_scale_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Final detection layers
        self.detection_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Output predictions: [x, y, w, h, obj, ...classes]
        output_filters = num_anchors * (5 + num_classes)
        self.output_conv = nn.Conv2d(256, output_filters, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Reduce channels
        x = self.reduce_conv(x)

        # Spatial pyramid pooling
        x = self.spp(x)

        # Feature fusion
        x = self.fusion_conv(x)

        # CSP processing
        x = self.csp(x)

        # Multi-scale feature processing
        # Store original size for matching
        original_size = x.shape[2:]

        # Upsample
        up = self.upsample(x)
        # Crop or pad upsampled features to match original size
        if up.shape[2:] != original_size:
            up = F.interpolate(up, size=original_size,
                               mode='bilinear', align_corners=False)

        # Downsample
        down = self.downsample(x)
        # Resize downsampled features to match original size
        if down.shape[2:] != original_size:
            down = F.interpolate(down, size=original_size,
                                 mode='bilinear', align_corners=False)

        # Concatenate multi-scale features (now all same size)
        x = torch.cat([up, down], dim=1)
        x = self.multi_scale_conv(x)

        # Final detection
        x = self.detection_conv(x)
        output = self.output_conv(x)

        return output


class VGGYOLOv8(nn.Module):
    """
    VGG16 backbone with YOLOv8 detection head for object detection.
    Combines VGG feature extraction with modern YOLO detection capabilities.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 80,
                 pretrained: bool = False, robust_method: Optional[BaseRobustMethod] = None,
                 input_size: int = 416):
        super(VGGYOLOv8, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.robust_method = robust_method

        # VGG16 Feature Extractor (modified for detection)
        self.features = self._make_vgg16_features(input_channels)

        # YOLOv8 Detection Head
        self.detection_head = YOLOv8Head(512, num_classes)

        # Initialize weights
        self._initialize_weights()

        if pretrained:
            self.load_pretrained_vgg_weights()

    def _make_vgg16_features(self, input_channels: int) -> nn.Sequential:
        """Create VGG16 feature extraction layers optimized for detection"""
        layers = []
        in_channels = input_channels

        # VGG16 configuration: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
               256, 'M', 512, 512, 512, 'M', 512, 512, 512]

        for i, v in enumerate(cfg):
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1, bias=False)
                layers.extend([
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ])
                in_channels = v

        # Remove the last maxpool to maintain spatial resolution for detection
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_vgg_weights(self):
        """Load pretrained VGG16 weights for the feature extractor"""
        try:
            from torch.hub import load_state_dict_from_url
            vgg16_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
            pretrained_dict = load_state_dict_from_url(
                vgg16_url, progress=True)

            model_dict = self.state_dict()

            # Map VGG16 weights to our feature extractor
            feature_dict = {}
            vgg_idx = 0
            our_idx = 0

            for name, param in pretrained_dict.items():
                if 'features' in name and 'weight' in name:
                    # Map VGG conv weights to our conv weights
                    our_name = f'features.{our_idx}.weight'
                    if our_name in model_dict and model_dict[our_name].shape == param.shape:
                        feature_dict[our_name] = param
                    our_idx += 3  # Skip BatchNorm and ReLU

            model_dict.update(feature_dict)
            self.load_state_dict(model_dict, strict=False)
            print("Loaded pretrained VGG16 weights for feature extractor")

        except Exception as e:
            print(f"Warning: Could not load pretrained VGG weights: {e}")

    def forward(self, x, targets=None):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            targets: Optional dict with keys 'boxes', 'labels', 'area', 'iscrowd', 'image_id'
        Returns:
            If self.training and targets is not None: dict of losses
            Else: list of detections (one dict per image with 'boxes', 'scores', 'labels')
        """
        features = self.features(x)

        if self.robust_method:
            batch_size = features.size(0)
            features_flat = features.view(batch_size, -1)
            features_flat, _ = self.robust_method(
                features_flat, features_flat, features_flat)
            spatial_size = int((features_flat.size(1) / 512) ** 0.5)
            features = features_flat.view(
                batch_size, 512, spatial_size, spatial_size)

        predictions = self.detection_head(features)
        batch, ch, h, w = predictions.shape
        num_anchors = 3
        bbox_ch = num_anchors * 4
        obj_ch = num_anchors * 1
        class_ch = num_anchors * self.num_classes  # Use dynamic class count

        # Reshape predictions
        predictions = predictions.permute(
            0, 2, 3, 1).contiguous()  # [batch, h, w, ch]
        # [batch, num_preds, ch_per_anchor]
        predictions = predictions.view(batch, h * w * num_anchors, -1)

        # Split predictions
        bbox_pred = predictions[:, :, :4]  # [batch, num_preds, 4]
        obj_pred = predictions[:, :, 4:5]  # [batch, num_preds, 1]
        class_pred = predictions[:, :, 5:]  # [batch, num_preds, num_classes]

        if self.training and targets is not None:
            loss_dict = self.compute_loss(
                bbox_pred, obj_pred, class_pred, targets, h, w)
            return loss_dict
        else:
            detections = []
            for i in range(batch):
                boxes = bbox_pred[i]  # [num_preds, 4]
                scores = torch.sigmoid(obj_pred[i])  # [num_preds, 1]
                labels = torch.argmax(class_pred[i], dim=1)  # [num_preds]
                detections.append({
                    'boxes': boxes,
                    'scores': scores.squeeze(),
                    'labels': labels
                })
            return detections

    def compute_loss(self, bbox_pred, obj_pred, class_pred, targets, h, w):
        """Compute YOLO-style loss"""
        device = bbox_pred.device
        batch_size = bbox_pred.size(0)
        num_preds = bbox_pred.size(1)

        # Generate anchor grid
        stride = self.input_size // h
        grid_x = torch.arange(w, device=device).repeat(h, 1).view(1, h, w, 1)
        grid_y = torch.arange(h, device=device).repeat(
            w, 1).t().view(1, h, w, 1)
        anchor_grid = torch.cat(
            [grid_x, grid_y], dim=-1).repeat(1, 1, 1, 3).view(1, h * w * 3, 2)

        # Decode predictions
        bbox_pred = bbox_pred.view(batch_size, num_preds, 4)
        # [batch, num_preds, 2]
        xy_pred = torch.sigmoid(bbox_pred[:, :, :2]) + anchor_grid
        wh_pred = torch.exp(bbox_pred[:, :, 2:])  # [batch, num_preds, 2]
        # [batch, num_preds, 4]
        boxes_pred = torch.cat(
            [xy_pred - wh_pred / 2, xy_pred + wh_pred / 2], dim=-1)

        # Initialize losses
        box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        class_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Match predictions to targets
        for i in range(batch_size):
            gt_boxes = targets['boxes'][i]  # [n, 4]
            gt_labels = targets['labels'][i]  # [n]
            if gt_boxes.shape[0] == 0:
                obj_target = torch.zeros((num_preds,), device=device)
                obj_loss += F.binary_cross_entropy_with_logits(
                    obj_pred[i].squeeze(), obj_target)
                continue

            iou = self.compute_iou(boxes_pred[i], gt_boxes)  # [num_preds, n]
            max_iou, max_idx = iou.max(dim=1)  # [num_preds]

            obj_target = (max_iou > 0.5).float()
            obj_loss += F.binary_cross_entropy_with_logits(
                obj_pred[i].squeeze(), obj_target)

            matched = max_iou > 0.5
            if matched.sum() > 0:
                matched_boxes_pred = boxes_pred[i][matched]
                matched_gt_boxes = gt_boxes[max_idx[matched]]
                box_loss += F.smooth_l1_loss(matched_boxes_pred,
                                             matched_gt_boxes)

                matched_labels = gt_labels[max_idx[matched]]
                class_loss += F.cross_entropy(
                    class_pred[i][matched], matched_labels)

        total_loss = box_loss + obj_loss + class_loss
        return {
            'total_loss': total_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'class_loss': class_loss
        }

    def compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

        inter_area = torch.clamp(
            inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = area1[:, None] + area2 - inter_area
        return inter_area / (union_area + 1e-6)


def get_vgg_yolov8(
    input_channels: int = 3,
    num_classes: int = 80,
    pretrained: bool = False,
    robust_method: None = None,
    input_size: int = 224
) -> VGGYOLOv8:
    """
    Create VGG-YOLOv8 model

    Args:
        input_channels: Number of input channels
        num_classes: Number of detection classes
        pretrained: Whether to load pretrained VGG weights
        robust_method: Optional robust method to apply
        input_size: Input image size

    Returns:
        VGGYOLOv8 model instance
    """
    model = VGGYOLOv8(
        input_channels=input_channels,
        num_classes=num_classes,
        input_size=input_size,
        pretrained=pretrained,
        robust_method=robust_method
    )
    return model

# Compatibility aliases


def get_vgg_yolo(input_channels: int = 3, num_classes: int = 80,
                 pretrained: bool = False, robust_method: Optional[BaseRobustMethod] = None) -> VGGYOLOv8:
    """Alias for get_vgg_yolov8"""
    return get_vgg_yolov8(input_channels, num_classes, pretrained, robust_method)


def get_vgg_detection(input_channels: int = 3, num_classes: int = 80,
                      pretrained: bool = False, robust_method: Optional[BaseRobustMethod] = None) -> VGGYOLOv8:
    """Alias for get_vgg_yolov8"""
    return get_vgg_yolov8(input_channels, num_classes, pretrained, robust_method)
