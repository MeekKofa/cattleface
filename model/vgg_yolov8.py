import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision.models import resnet50, ResNet50_Weights

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def ciou_loss(boxes1, boxes2):
    # boxes: [N, 4] in [x1, y1, x2, y2] normalized
    # Implementation based on https://github.com/Zzh-tju/DIoU
    eps = 1e-7
    x1, y1, x2, y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1g, y1g, x2g, y2g = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    inter_x1 = torch.max(x1, x1g)
    inter_y1 = torch.max(y1, y1g)
    inter_x2 = torch.min(x2, x2g)
    inter_y2 = torch.min(y2, y2g)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1 + area2 - inter + eps
    iou = inter / union

    # center distance
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    cx2 = (x1g + x2g) / 2
    cy2 = (y1g + y2g) / 2
    center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # enclosing box
    enclose_x1 = torch.min(x1, x1g)
    enclose_y1 = torch.min(y1, y1g)
    enclose_x2 = torch.max(x2, x2g)
    enclose_y2 = torch.max(y2, y2g)
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # aspect ratio
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x2g - x1g, y2g - y1g
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v + eps)
    ciou = iou - center_dist / enclose_diag - alpha * v
    return 1 - ciou

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def diou_nms(boxes, scores, threshold=0.5):
    # boxes: [N, 4], scores: [N]
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        # DIoU penalty
        dious = ciou_loss(boxes[i].unsqueeze(0).repeat(idxs[1:].numel(), 1), boxes[idxs[1:]])
        mask = (ious < threshold) & (dious > threshold)
        idxs = idxs[1:][mask]
    return keep

class ResNet50_YOLOv8(nn.Module):
    def __init__(self, num_classes=1, input_size=448, dropout=0.3):
        super(ResNet50_YOLOv8, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Use all layers except the final classification layers
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  # Add layer4 back for richer features
        )
        
        # Remove channel reducer - keep 2048 channels
        # Add spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 2048, 1),
            nn.Sigmoid()
        )
        
        # FPN with more capacity
        self.fpn = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        # YOLOv8 head uses FPN output
        self.pred_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 5 + num_classes, kernel_size=1)
        )
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.focal_loss = FocalLoss()
        
        # Initialize weights
        self.apply(initialize_weights)
        
        nn.init.constant_(self.pred_head[-1].bias[:4], 0.0)
        nn.init.constant_(self.pred_head[-1].bias[4], -4.0)

    def forward(self, x, targets=None):
        x = x.float()
        x = torch.clamp(x, -10, 10)
        features = self.backbone(x)
        # Apply spatial attention
        attn = self.attention(features)
        features = features * attn
        fpn_features = self.fpn(features)
        outputs = self.pred_head(fpn_features)
        batch_size, _, H, W = outputs.shape
        outputs = outputs.permute(0, 2, 3, 1).reshape(batch_size, H * W, -1)

        # Split into components
        raw_boxes = outputs[..., :4]
        pred_obj = torch.sigmoid(outputs[..., 4:5])
        pred_cls = outputs[..., 5:]

        # Ensure valid bounding boxes: convert to center-size then min-max, clamp
        center_x = torch.sigmoid(raw_boxes[..., 0])
        center_y = torch.sigmoid(raw_boxes[..., 1])
        width = torch.sigmoid(raw_boxes[..., 2])
        height = torch.sigmoid(raw_boxes[..., 3])

        x_min = (center_x - width / 2).clamp(0.0, 1.0)
        y_min = (center_y - height / 2).clamp(0.0, 1.0)
        x_max = (center_x + width / 2).clamp(0.0, 1.0)
        y_max = (center_y + height / 2).clamp(0.0, 1.0)

        pred_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        detections = []
        for i in range(batch_size):
            boxes = pred_boxes[i]
            scores = pred_obj[i].squeeze(-1)
            labels = torch.argmax(pred_cls[i], dim=-1)
            keep = diou_nms(boxes, scores, threshold=0.5)
            detections.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep]
            })
        if self.training and targets is not None:
            return (outputs, detections, fpn_features)
        else:
            return detections

    def compute_loss(self, outputs, targets):
        # Multi-anchor matching and CIoU loss
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            outputs = outputs[0]
        batch_size = outputs.shape[0]
        pred_boxes = torch.sigmoid(outputs[..., :4])
        pred_obj = torch.sigmoid(outputs[..., 4])
        pred_cls = outputs[..., 5:]
        box_loss = torch.tensor(0., device=outputs.device)
        obj_loss = torch.tensor(0., device=outputs.device)
        cls_loss = torch.tensor(0., device=outputs.device)
        for i in range(batch_size):
            img_boxes = pred_boxes[i]
            img_obj = pred_obj[i]
            img_cls = pred_cls[i]
            target_boxes = targets['boxes'][i].to(img_boxes.device)
            target_labels = targets['labels'][i].to(img_cls.device)
            if len(target_boxes) == 0:
                obj_loss += F.binary_cross_entropy(img_obj, torch.zeros_like(img_obj))
                continue
            ious = box_iou(img_boxes, target_boxes)
            # Multi-anchor matching
            pos_mask = ious > 0.5
            best_iou, best_idx = ious.max(dim=0)
            pos_mask[best_idx, torch.arange(len(target_boxes))] = True
            pos_indices = pos_mask.nonzero(as_tuple=True)
            if pos_indices[0].numel() > 0:
                matched_boxes = img_boxes[pos_indices[0]]
                matched_targets = target_boxes[pos_indices[1]]
                box_loss += ciou_loss(matched_boxes, matched_targets).mean()
                obj_targets = torch.zeros_like(img_obj)
                obj_targets[pos_indices[0]] = 1.0
                obj_loss += F.binary_cross_entropy(img_obj, obj_targets)
                if self.num_classes > 1:
                    cls_loss += F.cross_entropy(img_cls[pos_indices[0]], target_labels[pos_indices[1]])
                else:
                    logits = img_cls[pos_indices[0]].squeeze()
                    target_tensor = torch.ones_like(logits)
                    cls_loss += self.focal_loss(logits, target_tensor)
            else:
                obj_loss += F.binary_cross_entropy(img_obj, torch.zeros_like(img_obj))
        box_loss /= batch_size
        obj_loss /= batch_size
        cls_loss /= batch_size
        total_loss = 5.0 * box_loss + 1.0 * obj_loss + 2.0 * cls_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error("total_loss is NaN or Inf, replacing with high value")
            total_loss = torch.tensor(1000.0, device=outputs.device)
        return total_loss

    @property
    def classes(self):
        return list(range(self.num_classes))

# ADD BACK THE GET_VGG_YOLOV8 FUNCTION FOR BACKWARD COMPATIBILITY
def get_vgg_yolov8(num_classes=0, depth=16, input_size=448, dropout=0.3):
    return ResNet50_YOLOv8(num_classes=num_classes, input_size=input_size, dropout=dropout)

# Updated model creation function
def create_model(arch, depth, num_classes, dropout=0.3):
    if arch == "yolov8n":
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.model[-1].nc = num_classes
        return model
    elif arch == "resnet50_yolov8":
        return ResNet50_YOLOv8(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")