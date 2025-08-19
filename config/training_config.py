"""
Enhanced training configuration for better object detection performance.
"""

# Detection threshold optimization
DETECTION_CONFIG = {
    'confidence_threshold': 0.25,  # Lower threshold to catch more detections
    'nms_threshold': 0.45,         # Non-max suppression threshold
    'max_detections_per_image': 300,  # Allow more detections per image
    'score_threshold': 0.05,       # Minimum score for considering detections
}

# Training hyperparameters for better convergence
TRAINING_CONFIG = {
    'warmup_epochs': 5,            # Learning rate warmup
    'cosine_annealing': True,      # Use cosine annealing scheduler
    'gradient_clip_value': 10.0,   # Gradient clipping for stability
    'weight_decay': 0.0005,        # L2 regularization
    'momentum': 0.937,             # SGD momentum
}

# Loss function weights for balanced training
LOSS_CONFIG = {
    'box_loss_weight': 7.5,        # Increased weight for bounding box regression
    'obj_loss_weight': 1.0,        # Objectness loss weight
    'cls_loss_weight': 0.5,        # Classification loss weight (reduced)
    'focal_alpha': 0.25,           # Focal loss alpha
    'focal_gamma': 1.5,            # Focal loss gamma
    'label_smoothing': 0.0,        # Label smoothing for classification
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'mosaic_prob': 0.5,            # Probability of mosaic augmentation
    'mixup_prob': 0.15,            # Probability of mixup augmentation
    'hsv_gain': (0.015, 0.7, 0.4), # HSV color space augmentation gains
    'flip_prob': 0.5,              # Horizontal flip probability
    'rotation_range': 10,          # Rotation range in degrees
    'scale_range': (0.5, 1.5),     # Scale range for augmentation
}

# Anchor box optimization (for YOLO-style models)
ANCHOR_CONFIG = {
    'anchor_sizes': [
        [10, 13, 16, 30, 33, 23],      # Small objects
        [30, 61, 62, 45, 59, 119],     # Medium objects  
        [116, 90, 156, 198, 373, 326]  # Large objects
    ],
    'anchor_optimization': True,    # Whether to optimize anchors
    'iou_threshold': 0.20,         # IoU threshold for anchor matching
}

# Model architecture improvements
MODEL_CONFIG = {
    'use_attention': False,         # Whether to use attention mechanisms
    'dropout_rate': 0.1,           # Dropout rate for regularization
    'batch_norm': True,            # Use batch normalization
    'activation': 'silu',          # Activation function (silu/swish/relu)
    'use_fpn': True,              # Use Feature Pyramid Network
}

# Evaluation and monitoring
EVAL_CONFIG = {
    'eval_interval': 5,            # Evaluate every N epochs
    'save_interval': 10,           # Save checkpoint every N epochs
    'log_interval': 100,           # Log metrics every N batches
    'plot_interval': 25,           # Generate plots every N epochs
    'early_stopping_patience': 30, # Early stopping patience
}

def get_optimized_config():
    """Get the complete optimized configuration."""
    return {
        'detection': DETECTION_CONFIG,
        'training': TRAINING_CONFIG, 
        'loss': LOSS_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'anchors': ANCHOR_CONFIG,
        'model': MODEL_CONFIG,
        'evaluation': EVAL_CONFIG
    }

def print_config_summary():
    """Print a summary of the optimized configuration."""
    config = get_optimized_config()
    print("\n" + "="*60)
    print("OPTIMIZED TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    
    for section_name, section_config in config.items():
        print(f"\n{section_name.upper()} SETTINGS:")
        for key, value in section_config.items():
            print(f"  {key:25s}: {value}")
    
    print("\nKEY IMPROVEMENTS:")
    print("  • Lower confidence thresholds for better detection")
    print("  • Increased box loss weight for better localization") 
    print("  • Gradient clipping for training stability")
    print("  • Enhanced monitoring and diagnostics")
    print("  • Optimized anchor configurations")
    print("="*60)

if __name__ == "__main__":
    print_config_summary()
