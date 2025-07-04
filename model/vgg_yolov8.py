import torch
import torch.nn as nn
import torch.nn.functional as F
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
            self.backbone[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        
        # Feature pyramid network
        self.fpn_layers = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),  # Reduce channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Process features
        ])
        
        # Prediction head
        self.dropout = nn.Dropout(dropout_rate)
        self.bbox_head = nn.Conv2d(256, 4, kernel_size=1)  # 4 values for bbox (x, y, w, h)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)  # Class predictions
        
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
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
        
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
        
        for i in range(batch_size):
            target_boxes = targets[i]["boxes"]
            target_labels = targets[i]["labels"]
            
            if len(target_boxes) == 0:
                # No objects in this image
                continue
                
            # Compute IoU between predicted and target boxes
            iou = box_iou(bbox_pred[i], target_boxes)
            
            # Assign targets to predictions based on highest IoU
            max_iou, matched_idx = torch.max(iou, dim=1)
            positive_mask = max_iou > 0.5
            
            if positive_mask.sum() > 0:
                # Classification loss (only for positive matches)
                matched_targets = target_labels[matched_idx[positive_mask]]
                loss_cls += F.cross_entropy(cls_pred[i][positive_mask], matched_targets)
                
                # Regression loss (only for positive matches)
                matched_boxes = target_boxes[matched_idx[positive_mask]]
                loss_bbox += F.smooth_l1_loss(bbox_pred[i][positive_mask], matched_boxes)
        
        # Normalize losses
        loss_dict = {
            'loss_cls': loss_cls / batch_size,
            'loss_bbox': loss_bbox / batch_size,
            'loss': loss_cls / batch_size + loss_bbox / batch_size
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
