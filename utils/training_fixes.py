"""
Training fixes for VGG YOLO model - addressing the identical predictions issue
"""

import torch
import torch.nn as nn
import logging


def diagnose_model_predictions(model, data_loader, device):
    """Diagnose why model is producing identical predictions - simplified"""
    print("🔍 DIAGNOSING MODEL PREDICTIONS...")

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= 3:  # Check first 3 batches
                break

            if isinstance(images, list):
                images = [img.to(device) for img in images]
                sample_input = images[0].unsqueeze(0)
            else:
                images = images.to(device)
                sample_input = images[0].unsqueeze(0)

            # Get simple output without complex hooks to avoid NaN
            try:
                outputs = model(sample_input)
                if isinstance(outputs, list) and len(outputs) > 0:
                    first_det = outputs[0]
                    print(f"Batch {i} - Final detection: "
                          f"boxes={first_det['boxes'][:1] if len(first_det['boxes']) > 0 else 'none'}, "
                          f"scores={first_det['scores'][:1] if len(first_det['scores']) > 0 else 'none'}, "
                          f"labels={first_det['labels'][:1] if len(first_det['labels']) > 0 else 'none'}")
                else:
                    print(f"Batch {i} - No valid output")
            except Exception as e:
                print(f"Batch {i} - Error: {e}")

    model.train()


def simple_fix_identical_predictions(model):
    """Simple fix for identical predictions - improved version"""
    print("🔧 APPLYING SIMPLE PREDICTION FIXES...")

    # Better initialization for prediction layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and ('pred_head' in name or 'head' in name):
            # Xavier initialization with smaller gain
            nn.init.xavier_normal_(module.weight, gain=0.01)
            if module.bias is not None:
                if 'obj' in name or name.endswith('4'):  # Objectness layer
                    # Bias toward no object
                    nn.init.constant_(module.bias, -2.0)
                else:
                    nn.init.constant_(module.bias, 0.0)

    # Add small noise to break symmetry
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:  # Only weight matrices
                param.add_(torch.randn_like(param) * 0.001)

    print("✅ Simple prediction fixes applied!")
    return model
