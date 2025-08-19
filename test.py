import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Any, Optional

from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader
from utils.visual.visualization import Visualization
# Add metrics utilities for per-class metrics
from utils.metrics import Metrics
from utils.evaluator import Evaluator
# Import the centralized argument parser
from argument_parser import parse_args

import torchmetrics

# Setup logging - Fix the format string error and remove timestamp
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(args, num_classes):
    """Load model from specified path or use ModelLoader to find appropriate checkpoint"""

    # If the model path is provided directly, use that instead of ModelLoader's search
    if args.model_path and os.path.exists(args.model_path):
        logging.info(
            f"Loading model directly from specified path: {args.model_path}")

        model_loader = ModelLoader(args.device, args.arch, pretrained=False)
        depth_value = args.depth
        if isinstance(depth_value, str) and depth_value.startswith('{'):
            import json
            depth_value = json.loads(depth_value.replace("'", "\""))
        input_channels = getattr(args, 'input_channels', 3)
        model_name = args.arch[0] if isinstance(args.arch, list) else args.arch
        models_and_names = model_loader.get_model(
            model_name=model_name,
            depth=depth_value,
            input_channels=input_channels,
            num_classes=num_classes
        )
        model, model_name_with_depth = models_and_names[0]

        # --- Model Loading Mismatch Debug ---
        print("Model keys:", [k for k in model.state_dict().keys()][:5])  # First 5 keys

        try:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            print("Checkpoint keys:", [k for k in checkpoint.keys()][:5])  # First 5 keys

            # Key alignment if needed
            if any(k.startswith("module.") for k in checkpoint.keys()):
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

            result = model.load_state_dict(checkpoint, strict=False)
            if isinstance(result, tuple) or hasattr(result, 'missing_keys'):
                missing_keys = getattr(result, 'missing_keys', [])
                unexpected_keys = getattr(result, 'unexpected_keys', [])
            else:
                missing_keys, unexpected_keys = [], []

            if missing_keys or unexpected_keys:
                logging.warning(
                    "Model architecture mismatch detected. Attempting to load compatible weights...")
                model_dict = model.state_dict()
                compatible_dict = {}
                for k, v in checkpoint.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                model.load_state_dict(compatible_dict, strict=False)
                logging.info(
                    f"Loaded {len(compatible_dict)} compatible weights from {args.model_path}")
            else:
                logging.info(
                    f"Successfully loaded all model weights from {args.model_path}")

        except Exception as e:
            logging.error(f"Error loading model weights: {e}")
            raise RuntimeError(f"Failed to load model weights: {e}")

    else:
        # Use ModelLoader to find the latest checkpoint
        logging.info(
            "Model path not specified or doesn't exist. Using ModelLoader to find checkpoint.")
        task_name = args.task_name or "normal_training"
        model_loader = ModelLoader(args.device, args.arch)

        # Set default input_channels if not defined in args
        # Default to 3 if not specified
        input_channels = getattr(args, 'input_channels', 3)

        # Extract the first model name from the list if it's a list
        model_name = args.arch[0] if isinstance(args.arch, list) else args.arch

        models_and_names = model_loader.get_model(
            model_name=model_name,  # Use extracted string instead of list
            depth=args.depth,
            input_channels=input_channels,
            num_classes=num_classes,
            task_name=task_name,
            dataset_name=args.data[0] if isinstance(
                args.data, list) else args.data,
            adversarial=args.adversarial
        )

        if not models_and_names:
            raise ValueError(
                f"No models found for {model_name} with depth {args.depth}")

        model, model_name_with_depth = models_and_names[0]

    model.eval()
    # --- Model Architecture Verification ---
    original_params = 2327622  # Example from training log
    current_params = sum(p.numel() for p in model.parameters())
    print(f"Param count: Original={original_params}, Current={current_params}")
    if hasattr(model, "num_classes"):
        print(f"Model num_classes: {model.num_classes}")

    return model, model_name_with_depth


def test_model(model, test_loader, args):
    """Test the model on the provided test loader"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    device = args.device
    is_object_detection = hasattr(model, 'compute_loss') or 'yolo' in str(type(model)).lower()

    # --- Input/Output Mismatch Debug ---
    model.eval()
    with torch.no_grad():
        sample = next(iter(test_loader))
        gt = sample[1]
        # Defensive access for ground truth boxes (fix KeyError: 0)
        if isinstance(gt, dict) and 'boxes' in gt:
            print("Ground truth boxes:", gt['boxes'][:2] if len(gt['boxes']) > 0 else "No boxes")
        elif isinstance(gt, list) and len(gt) > 0 and isinstance(gt[0], dict) and 'boxes' in gt[0]:
            print("Ground truth boxes:", gt[0]['boxes'][:2] if len(gt[0]['boxes']) > 0 else "No boxes")
        else:
            print("Ground truth boxes: Not found or unexpected format")
        # Model output debug (unchanged)
        output = model(sample[0].to(device))
        print("Sample output boxes:", output[0]['boxes'][:2])
        print("Sample output scores:", output[0]['scores'][:2])
        print("Sample output labels:", output[0]['labels'][:2])
    adversarial_trainer = None
    if args.adversarial:
        logging.info("Setting up adversarial evaluation...")
        try:
            from gan.defense.adv_train import AdversarialTraining
            from gan.attack.attack_loader import AttackLoader

            # Create attack loader
            attack_loader = AttackLoader(model, args)
            attack = attack_loader.get_attack(args.attack_type or "fgsm")
            logging.info(f"Initialized {args.attack_type or 'fgsm'} attack")

            # Create adversarial trainer (for generation only, not for training)
            criterion = torch.nn.CrossEntropyLoss()
            adversarial_trainer = AdversarialTraining(model, criterion, args)

        except Exception as e:
            logging.error(f"Error setting up adversarial evaluation: {e}")
            raise RuntimeError(f"Failed to setup adversarial evaluation: {e}")

    from tqdm import tqdm
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # --- Match train/model file conventions for object detection ---
                if is_object_detection:
                    images, targets = batch_data
                    # Data sanitization (same as train.py)
                    def validate_batch(images, targets):
                        valid_batch = True
                        if torch.isnan(images).any() or torch.isinf(images).any():
                            print("Invalid image values detected")
                            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
                            valid_batch = False
                        if isinstance(targets, dict) and 'boxes' in targets:
                            boxes = targets['boxes']
                            if isinstance(boxes, list):
                                boxes = torch.stack([b if isinstance(b, torch.Tensor) else torch.tensor(b) for b in boxes])
                            elif not isinstance(boxes, torch.Tensor):
                                boxes = torch.tensor(boxes)
                            if boxes.numel() > 0:
                                if (boxes.min() < 0) or (boxes.max() > 1) or torch.isnan(boxes).any():
                                    print(f"Invalid boxes in batch: min={boxes.min()}, max={boxes.max()}, NaNs={torch.isnan(boxes).any()}")
                                    new_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=boxes.dtype)
                                    if isinstance(targets['boxes'], list):
                                        for i in range(len(targets['boxes'])):
                                            targets['boxes'][i] = new_boxes
                                    else:
                                        targets['boxes'] = new_boxes
                                    valid_batch = False
                        return images, targets, valid_batch
                    images, targets, valid = validate_batch(images, targets)
                    if not valid:
                        print(f"Skipping problematic batch {batch_idx}")
                        continue
                    # Move images to device
                    if isinstance(images, torch.Tensor):
                        images = images.to(device)
                    else:
                        images = [img.to(device) for img in images]
                    # Convert targets dict of lists to batch tensors for metrics
                    batch_labels = []
                    for i in range(len(targets['labels'])):
                        labels_tensor = targets['labels'][i]
                        if labels_tensor.numel() > 0:
                            batch_labels.append(labels_tensor[0].item())
                        else:
                            batch_labels.append(0)
                    target = torch.tensor(batch_labels, dtype=torch.long)
                else:
                    data, target = batch_data
                    images = data
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
                    if isinstance(images, torch.Tensor):
                        images = images.to(device)
                    else:
                        images = [img.to(device) for img in images]
                target = target.to(device)

                # --- Match model file: clamp input for robustness ---
                if is_object_detection:
                    images = torch.clamp(images, -10, 10)

                # Generate adversarial examples if needed
                if args.adversarial and adversarial_trainer is not None:
                    images = adversarial_trainer._pgd_attack(images, target)

                # Get model predictions
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    if is_object_detection:
                        detections = model(images)
                        batch_preds = []
                        batch_probs = []
                        num_classes = getattr(model, 'num_classes', 20)
                        if isinstance(detections, list):
                            for detection in detections:
                                # Clamp outputs for robustness (match model.py)
                                detection['boxes'] = torch.clamp(detection['boxes'], 0, 1)
                                detection['scores'] = torch.clamp(detection['scores'], 0, 1)
                                if 'labels' in detection and len(detection['labels']) > 0:
                                    best_idx = torch.argmax(detection['scores'])
                                    pred_label = detection['labels'][best_idx].item()
                                    pred_score = detection['scores'][best_idx].item()
                                else:
                                    pred_label = 0
                                    pred_score = 0.5
                                batch_preds.append(pred_label)
                                probs = torch.zeros(num_classes)
                                probs[pred_label] = pred_score
                                batch_probs.append(probs.numpy())
                        else:
                            batch_size = len(images) if isinstance(images, list) else images.size(0)
                            batch_preds = [0] * batch_size
                            batch_probs = [np.zeros(num_classes) for _ in range(batch_size)]
                        pred = torch.tensor(batch_preds, dtype=torch.long)
                        probs = torch.tensor(np.array(batch_probs), dtype=torch.float32)
                    else:
                        output = model(images)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        _, pred = torch.max(output, 1)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                if batch_idx % 10 == 0:
                    progress_bar.set_description(
                        f"Testing batch {batch_idx}/{len(test_loader)}")

            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Concatenate results
    try:
        if not all_preds:
            raise RuntimeError(
                "No predictions were generated - all batches failed")

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Fix for binary classification: ensure probabilities have shape (n_samples, n_classes)
        # This addresses the "axis 1 is out of bounds" error
        if all_probs.ndim == 1 or (all_probs.shape[1] == 1 and len(np.unique(all_targets)) == 2):
            logger.info("Reshaping probabilities for binary classification...")
            # For binary case with single probability output, convert to two-column format
            all_probs = np.column_stack((1 - all_probs, all_probs))

    except Exception as e:
        logging.error(f"Error concatenating results: {e}")
        raise RuntimeError(f"Failed to concatenate results: {e}")

    return all_preds, all_targets, all_probs


def generate_detailed_metrics(model_name: str,
                              all_preds: np.ndarray,
                              all_targets: np.ndarray,
                              all_probs: np.ndarray,
                              class_names: List[str],
                              args) -> Dict[str, Any]:
    """
    Generate comprehensive detailed metrics for testing

    Args:
        model_name: Name of the model
        all_preds: Predicted class indices
        all_targets: Ground truth labels
        all_probs: Predicted probabilities
        class_names: List of class names
        args: Command line arguments

    Returns:
        Dictionary containing detailed metrics
    """
    # Import necessary metrics functions directly
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    logger.info("Generating detailed metrics and visualizations...")

    # Ensure probabilities have correct shape for multi-class metrics
    num_classes = len(np.unique(all_targets))

    try:
        # Check and fix probability array dimensions
        if all_probs.ndim == 1:
            logger.info(
                f"Reshaping 1D probabilities array of shape {all_probs.shape} for binary classification")
            all_probs = np.column_stack((1 - all_probs, all_probs))
        elif all_probs.shape[1] != num_classes and num_classes > 1:
            logger.warning(
                f"Probabilities shape {all_probs.shape} doesn't match number of classes {num_classes}. Reshaping...")
            # If we have a binary problem with one probability column
            if num_classes == 2 and all_probs.shape[1] == 1:
                all_probs = np.column_stack((1 - all_probs, all_probs))
            else:
                # For multi-class case, use one-hot encoded probabilities
                # This is a fallback case and may not be accurate
                all_probs = label_binarize(
                    all_preds, classes=np.unique(all_targets))
    except Exception as e:
        logger.warning(f"Error reshaping probabilities: {e}")

    try:
        # Calculate all possible metrics using the Metrics utility
        detailed_metrics = Metrics.calculate_metrics(
            all_targets, all_preds, all_probs)
    except Exception as e:
        logger.warning(f"Error in calculate_metrics: {e}")
        # Create a basic metrics dictionary as fallback
        detailed_metrics = {
            'accuracy': (all_preds == all_targets).mean(),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        }

    # Calculate per-class metrics
    per_class = {}
    classes = np.unique(all_targets)

    # For binary classification, calculate single ROC AUC value
    binary_auc = None
    if len(classes) == 2:
        try:
            # Make sure we're using the probability for the positive class
            # For binary classification, ROC AUC is the same for both classes
            pos_class_prob = all_probs[:,
                                       1] if all_probs.shape[1] > 1 else all_probs
            binary_auc = roc_auc_score(all_targets, pos_class_prob)
            logger.info(
                f"Binary classification detected. Overall ROC AUC: {binary_auc:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate binary ROC AUC: {e}")

    for cls in classes:
        cls_mask = (all_targets == cls)
        if sum(cls_mask) == 0:
            continue

        # Binary classification for this class (one-vs-rest)
        cls_true = (all_targets == cls).astype(int)
        cls_pred = (all_preds == cls).astype(int)

        # Calculate comprehensive metrics directly with imported functions
        try:
            cls_metrics = {
                'accuracy': np.mean(cls_true == cls_pred),
                'precision': precision_score(cls_true, cls_pred, zero_division=0),
                'recall': recall_score(cls_true, cls_pred, zero_division=0),
                'f1': f1_score(cls_true, cls_pred, zero_division=0),
                # Manual calculation
                'specificity': np.sum((cls_true == 0) & (cls_pred == 0)) / max(1, np.sum(cls_true == 0)),
                'support': np.sum(cls_true)
            }

            # For binary classification, use the overall AUC
            if binary_auc is not None:
                cls_metrics['roc_auc'] = binary_auc
            # For multi-class, calculate class-specific AUC
            elif all_probs is not None and all_probs.shape[1] >= num_classes:
                try:
                    # Get index in classes array
                    cls_idx = np.where(classes == cls)[0][0]
                    if cls_idx < all_probs.shape[1]:
                        cls_probs = all_probs[:, cls_idx]
                        auc_value = roc_auc_score(cls_true, cls_probs)
                        cls_metrics['roc_auc'] = auc_value
                    else:
                        logger.debug(
                            f"Class index {cls_idx} out of bounds for probabilities with shape {all_probs.shape}")
                except Exception as e:
                    logger.debug(
                        f"Could not calculate ROC AUC for class {cls}: {e}")
                    # Use binary AUC as fallback
                    if binary_auc is not None:
                        cls_metrics['roc_auc'] = binary_auc
        except Exception as e:
            logger.debug(f"Error calculating metrics for class {cls}: {e}")
            # Create partial metrics if calculation fails
            cls_metrics = {
                'accuracy': np.mean(cls_true == cls_pred),
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': np.sum(cls_true)
            }
            # Use binary AUC as fallback
            if binary_auc is not None:
                cls_metrics['roc_auc'] = binary_auc

        class_name = class_names[cls] if cls < len(
            class_names) else f"Class {cls}"
        per_class[class_name] = cls_metrics

    detailed_metrics['per_class'] = per_class

    # Log per-class metrics with more detail than during training
    logger.info("Per-Class Metrics (Detailed):")
    for cls_name, metrics in per_class.items():
        base_metrics = (f"  {cls_name}: F1={metrics['f1']:.4f}, "
                        f"Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, "
                        f"Support={metrics['support']}")

        # Add AUC if available and valid
        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            base_metrics += f", AUC={metrics['roc_auc']:.4f}"
        else:
            base_metrics += ", AUC=N/A"  # Show N/A instead of nan

        logger.info(base_metrics)

    # Create all possible visualizations using the Visualization class
    visualization = Visualization()

    # Create confusion matrix
    try:
        visualization.visualize_metrics(
            metrics=detailed_metrics,
            task_name=args.task_name,
            dataset_name=args.data[0],
            model_name=model_name,
            phase="test",
            class_names=class_names
        )
    except Exception as e:
        logger.warning(f"Error generating confusion matrix: {e}")

    # For multi-class, visualize all pairwise ROC curves
    if len(classes) > 2:
        logger.info("Creating multi-class ROC curves...")
        # Use visualize_normal which handles multi-class ROC curves
        try:
            visualization.visualize_normal(
                model_names=[model_name],
                data=(
                    {model_name: all_targets},  # true labels dict
                    {model_name: all_preds},    # predictions dict
                    {model_name: all_probs}     # probabilities dict
                ),
                task_name=args.task_name,
                dataset_name=args.data[0],
                class_names=class_names
            )
        except Exception as e:
            logger.warning(f"Error generating ROC curves: {e}")
    # For binary classification, create threshold optimization curve
    elif len(classes) == 2 and all_probs.shape[1] >= 2:
        logger.info(
            "Creating binary classification threshold optimization curves...")
        try:
            optimal_threshold = visualization.create_threshold_curve(
                true_labels=all_targets,
                probabilities=all_probs,
                task_name=args.task_name,
                dataset_name=args.data[0],
                model_name=model_name
            )
            logger.info(
                f"Optimal threshold for binary classification: {optimal_threshold:.4f}")
        except Exception as e:
            logger.warning(
                f"Error generating threshold optimization curves: {e}")

    return detailed_metrics


def test_single_image(model, image_path, class_names, device, args):
    """Test model on a single image"""
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    logger.info(f"Testing single image: {image_path}")

    # Check if file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return

    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')

        # Create standard preprocessing transform
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # Preprocess image
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        # Create a non-normalized version for display
        display_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        display_tensor = display_transform(img)

        # Set model to evaluation mode
        model.eval()

        # Generate adversarial version if requested
        adv_input = None
        if args.adversarial:
            from gan.defense.adv_train import AdversarialTraining
            # Create dummy targets (this is just for visualization)
            dummy_target = torch.tensor([0], device=device)
            adv_trainer = AdversarialTraining(
                model, torch.nn.CrossEntropyLoss(), args)

            with torch.enable_grad():
                input_batch.requires_grad_(True)
                _, adv_batch, _ = adv_trainer.attack.attack(
                    input_batch, dummy_target)
                adv_input = adv_batch.detach()

                # Create perturbation visualization
                perturbation = adv_batch - input_batch
                # Scale for visibility
                enhanced_pert = (perturbation * 10) + 0.5
                enhanced_pert = torch.clamp(enhanced_pert, 0, 1)

        # Forward pass
        with torch.no_grad():
            output = model(input_batch)

            # Get adversarial prediction if available
            adv_output = model(adv_input) if adv_input is not None else None

        # Get predictions
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        adv_probs = torch.nn.functional.softmax(adv_output, dim=1)[
            0] if adv_output is not None else None

        # Get top prediction
        top_prob, top_class = torch.topk(probabilities, 1)
        predicted_class = class_names[top_class.item()]
        confidence = top_prob.item()

        # Get adversarial prediction if available
        adv_class = None
        if adv_probs is not None:
            adv_top_prob, adv_top_class = torch.topk(adv_probs, 1)
            adv_class = class_names[adv_top_class.item()]
            adv_confidence = adv_top_prob.item()

        # Print results
        logger.info(
            f"Prediction: {predicted_class} with {confidence:.4f} confidence")
        if adv_class is not None:
            logger.info(
                f"Adversarial prediction: {adv_class} with {adv_confidence:.4f} confidence")

        # Create output directory for visualizations
        os.makedirs('out/single_image_tests', exist_ok=True)

        # Create output image with prediction overlay
        fig, axs = plt.subplots(
            1, 3 if adv_input is not None else 1, figsize=(15, 5))
        if adv_input is None:
            axs = [axs]  # Make axs a list for consistent indexing

        # Display original image with prediction
        img_np = display_tensor.cpu().numpy().transpose(1, 2, 0)
        axs[0].imshow(img_np)
        axs[0].set_title(
            f"Prediction: {predicted_class}\nConfidence: {confidence:.4f}")
        axs[0].axis('off')

        # If we have adversarial version, show it too
        if adv_input is not None:
            # Display adversarial image
            adv_np = adv_input[0].cpu().detach().numpy(
            ).transpose(1, 2, 0)  # Added detach()
            # Normalize for display
            adv_np = (adv_np - adv_np.min()) / \
                (adv_np.max() - adv_np.min() + 1e-8)
            axs[1].imshow(adv_np)
            axs[1].set_title(
                f"Adversarial: {adv_class}\nConfidence: {adv_confidence:.4f}")
            axs[1].axis('off')

            # Display perturbation - Fixed the detach() issue here
            pert_np = enhanced_pert[0].cpu().detach(
            ).numpy().transpose(1, 2, 0)  # Added detach()
            axs[2].imshow(pert_np)
            axs[2].set_title(f"Perturbation (Enhanced)")
            axs[2].axis('off')

        # Save figure
        output_path = os.path.join(
            'out/single_image_tests', os.path.basename(image_path))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        # Generate class activation map (heatmap) if requested
        if args.show_heatmap:
            generate_heatmap(model, input_tensor, img, output_path, device)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error testing single image: {e}")
        import traceback
        traceback.print_exc()


def generate_heatmap(model, input_tensor, original_img, output_path, device):
    """Generate a class activation heatmap for network visualization"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        # Create GradCAM-like visualization
        # Find the last convolutional layer
        last_conv_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
                last_conv_name = name
                break

        if last_conv_layer is None:
            logger.warning(
                "Could not find convolutional layer for heatmap generation")
            return

        # Define hook to get feature maps
        feature_maps = None
        gradients = None

        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        # Register hooks
        forward_handle = last_conv_layer.register_forward_hook(forward_hook)
        backward_handle = last_conv_layer.register_full_backward_hook(
            backward_hook)

        # Set model to eval mode
        model.eval()

        # Forward pass with gradient calculation
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        pred_class = output.argmax().item()

        # Zero all gradients
        model.zero_grad()

        # Backward pass for the predicted class
        output[0, pred_class].backward()

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Generate heatmap
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * feature_maps,
                            dim=1).squeeze().cpu().numpy()

        # Make positive values only and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)

        # Resize heatmap to match original image size
        heatmap = cv2.resize(
            heatmap, (original_img.width, original_img.height))

        # Convert heatmap to RGB colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert original image to numpy array
        original_np = np.array(original_img)

        # Convert RGB to BGR for OpenCV
        original_np = original_np[:, :, ::-1]

        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

        # Save heatmap visualization
        heatmap_path = output_path.replace('.png', '_heatmap.png')
        # Convert back to RGB
        cv2.imwrite(heatmap_path, superimposed[:, :, ::-1])

        logger.info(f"Heatmap saved to {heatmap_path}")

    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")


def _is_classification_model(arch: str) -> bool:
    """Check if the given architecture is a classification model"""
    classification_models = [
        'resnet', 'densenet', 'vgg', 'vgg_myccc', 'meddef1']
    object_detection_models = ['vgg_yolov8', 'yolo', 'ssd', 'faster_rcnn']

    # Handle case where arch might be a list or string
    if isinstance(arch, list):
        arch = arch[0] if arch else ''
    arch_str = str(arch).lower()

    # Check if it's explicitly an object detection model
    if any(model in arch_str for model in object_detection_models):
        return False

    # Check if it's a classification model
    return any(model in arch_str for model in classification_models)


def _should_force_classification(dataset_loader, dataset_name: str, arch: str) -> bool:
    """Determine if we should force classification mode for object detection dataset"""
    is_obj_detection_dataset = dataset_loader._is_object_detection_dataset(
        dataset_name)
    is_classification_model = _is_classification_model(arch)

    # Only force classification if dataset is object detection AND model is classification
    # For our case with vgg_yolov8, we don't want to force classification since it's an OD model
    return is_obj_detection_dataset and is_classification_model


def compute_object_detection_metrics(model, test_loader, device, num_classes):
    """
    Compute mAP, IoU, and precision/recall for object detection models using torchmetrics.
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            images, targets = batch_data
            if isinstance(images, torch.Tensor):
                images = images.to(device)
            else:
                images = [img.to(device) for img in images]
            gt = []
            # Updated to match dataset_loader output (dict of lists)
            if isinstance(targets, dict):
                for i in range(len(targets['boxes'])):
                    gt.append({
                        'boxes': targets['boxes'][i].cpu(),
                        'labels': targets['labels'][i].cpu()
                    })
            elif isinstance(targets, list):
                gt = [{k: v[i].cpu() for k, v in targets.items()} for i in range(len(images))]
            else:
                continue
            preds = model(images)
            if isinstance(preds, dict):
                preds = [preds]
            preds = [{k: v.cpu() for k, v in pred.items()} for pred in preds]
            metric.update(preds, gt)
    results = metric.compute()
    return results


def save_yolo_text_results(output_dir, detection_metrics, model_name, args):
    """
    Save YOLO-style text results summarizing train, model analysis, and visualization.
    """
    result_path = os.path.join(output_dir, "yolo_results.txt")
    with open(result_path, "w") as f:
        f.write(f"YOLO Model: {model_name}\n")
        f.write(f"Dataset: {args.data[0] if isinstance(args.data, list) else args.data}\n")
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Depth: {args.depth}\n")
        f.write(f"Train batch size: {args.batch_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write("\n--- Model Analysis ---\n")
        for k, v in detection_metrics.items():
            if hasattr(v, "item"):
                v = float(v.item())
            f.write(f"{k}: {v}\n")
        f.write("\n--- Visualization ---\n")
        f.write(
            "Note: For object detection (YOLO), visualizations like confusion matrix and PR curves are not generated automatically.\n"
            "To create these, you need to:\n"
            "  1. Enable per-class metrics and plotting in your config (e.g., set per_class_metrics=True).\n"
            "  2. Use or implement visualization utilities that process detection outputs (boxes, labels, scores) and ground truth to generate plots.\n"
            "  3. For confusion matrix, you must convert detection results to class predictions and compare with ground truth labels.\n"
            "  4. For PR curves, aggregate detection scores and labels across the dataset and use sklearn or torchmetrics functions.\n"
            "  5. Check your codebase for functions like visualize_detection, visualize_metrics, or similar, and call them after testing.\n"
            "  6. Output folders like out/visualizations/ will only be created if these utilities are run.\n"
        )
        f.write("See out/test_results/detection_metrics.json for full metrics.\n")
    logger.info(f"YOLO text results saved to {result_path}")

def main():
    # Parse arguments - use the unified parser instead of mode-specific one
    args = parse_args()

    # Define output directory for saving results and logs
    output_dir = os.path.join('out', 'test_results')
    os.makedirs(output_dir, exist_ok=True)

    # Set default attribute values that might be missing from args
    if not hasattr(args, 'input_channels'):
        args.input_channels = 3  # Default to RGB images

    if not hasattr(args, 'fp16'):
        args.fp16 = False  # Default to FP32 precision

    if not hasattr(args, 'save_predictions'):
        args.save_predictions = False

    if not hasattr(args, 'batch_size'):
        args.batch_size = 32  # Default batch size

    if not hasattr(args, 'num_workers'):
        args.num_workers = 4  # Default number of workers

    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Suppress all warnings at the beginning
    from utils.file_handler import FileHandler
    FileHandler.suppress_warnings()

    # Explicitly suppress PyTorch warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*Named tensors.*")
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*Empty data.*")

    # Set matplotlib backend only once at the very top (before any plotting)
    import matplotlib
    if not hasattr(main, "_mpl_backend_set"):
        matplotlib.use('Agg')
        main._mpl_backend_set = True
        logging.info("Setting matplotlib backend to 'Agg'")

    # Print arguments for debugging
    logging.info(f"Running with arguments: {args}")

    # For single image testing
    if args.image_path:
        dataset_loader = DatasetLoader()
        # Remove force_classification argument (not supported)
        _, _, test_loader = dataset_loader.load_data(
            dataset_name=args.data[0],
            batch_size={'train': 1, 'val': 1, 'test': 1},
            num_workers=1,
            pin_memory=False
        )

        # Get class names and number of classes
        if hasattr(test_loader.dataset, 'classes') and test_loader.dataset.classes:
            class_names = [f"Class {i}" for i in range(
                len(test_loader.dataset.classes))]
            num_classes = len(test_loader.dataset.classes)
        elif hasattr(test_loader.dataset, 'class_to_idx'):
            class_names = list(test_loader.dataset.class_to_idx.keys())
            num_classes = len(test_loader.dataset.class_to_idx)
        else:
            # Use programmatic class count for object detection
            from class_mapping import get_num_classes
            num_classes = get_num_classes()
            class_names = [f"Class {i}" for i in range(num_classes)]

        # Load model
        model, model_name = load_model(args, num_classes)

        # Test single image
        test_single_image(model, args.image_path, class_names, device, args)
        return

    # Continue with regular dataset testing
    # Load the dataset with appropriate batch sizes
    dataset_loader = DatasetLoader()
    # Remove force_classification argument (not supported)
    _, _, test_loader = dataset_loader.load_data(
        dataset_name=args.data[0],
        batch_size={'train': args.batch_size,
                    'val': args.batch_size, 'test': args.batch_size},
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if hasattr(args, 'pin_memory') else False
    )

    # --- FIX: Ensure test matches train/model setup for class count and architecture ---
    # Load the model and get num_classes from the training config or model weights
    from class_mapping import get_num_classes
    # Always use the same num_classes as used in training/model
    # If you have a saved config or can extract from the model, use that
    # Otherwise, fallback to 1 for single-class detection (as in train)
    # Example: For cattlebody, force single-class detection
    force_single_class = False
    if hasattr(args, "data") and args.data and "cattlebody" in str(args.data[0]):
        force_single_class = True
    if force_single_class:
        num_classes = 1
        class_names = ["Class 0"]
        logging.info("Forcing single-class detection for cattlebody (to match training/model).")
    else:
        num_classes = get_num_classes()
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Load the model with correct num_classes
    model, model_name = load_model(args, num_classes)

    # Test the model
    all_preds, all_targets, all_probs = test_model(model, test_loader, args)

    # --- FIX: Object detection model check matches train/model ---
    is_object_detection = (
        hasattr(model, 'detection_head') or
        hasattr(model, 'pred_head') or
        'yolo' in str(type(model)).lower()
    )

    if is_object_detection:
        logger.info(
            "Object Detection Model - Using detection-specific evaluation")

        # Compute detection metrics
        detection_metrics = compute_object_detection_metrics(model, test_loader, device, num_classes)
        logger.info(f"Detection metrics (torchmetrics):")
        for k, v in detection_metrics.items():
            logger.info(f"{k}: {v}")

        # Print mAP, IoU, and per-class AP
        if 'map' in detection_metrics:
            logger.info(f"mAP: {detection_metrics['map']:.4f}")
        if 'map_50' in detection_metrics:
            logger.info(f"mAP@0.5: {detection_metrics['map_50']:.4f}")
        if 'map_75' in detection_metrics:
            logger.info(f"mAP@0.75: {detection_metrics['map_75']:.4f}")
        if 'mar_100' in detection_metrics:
            logger.info(f"mAR@100: {detection_metrics['mar_100']:.4f}")
        if 'classes' in detection_metrics:
            logger.info(f"Per-class AP: {detection_metrics['classes']}")

        # Save detection metrics to file
        import json
        metrics_to_save = {k: float(v) if hasattr(v, "item") else v for k, v in detection_metrics.items()}
        with open(os.path.join(output_dir, "detection_metrics.json"), "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        # Save YOLO-style text results
        save_yolo_text_results(output_dir, detection_metrics, model_name, args)

        logger.info("Detection metrics saved to out/test_results/detection_metrics.json")
        logger.info("For more detailed PR curves and threshold analysis, use torchmetrics outputs or implement custom plotting.")

        logger.info(f"Testing complete. Results saved to {output_dir}")
        return

    # Save predictions if requested (adapted for object detection)
    if args.save_predictions:
        if is_object_detection:
            # For object detection, save simpler prediction format
            pred_df = pd.DataFrame({
                'sample_id': range(len(all_preds)),
                'predicted_class': all_preds,
                'confidence': all_probs.flatten() if len(all_probs) > 0 else [0.5] * len(all_preds)
            })
        else:
            # Original classification format
            pred_df = pd.DataFrame({
                'true': all_targets,
                'pred': all_preds
            })
            # Add probability columns for each class
            for i, class_name in enumerate(class_names):
                pred_df[f'prob_{class_name}'] = all_probs[:, i]

        pred_df.to_csv(os.path.join(
            output_dir, f"predictions.csv"), index=False)

    # Enhanced visualizations section using existing Visualization class (skip for object detection)
    if not is_object_detection:
        visualization = Visualization()

        try:
            # Suppress matplotlib category warnings
            import warnings
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="matplotlib")

            # Basic visualizations with your existing method
            try:
                visualization.visualize_normal(
                    model_names=[model_name],
                    data=(
                        {model_name: all_targets},  # true labels dict
                        {model_name: all_preds},    # predictions dict
                        {model_name: all_probs}     # probabilities dict
                    ),
                    task_name=args.task_name,
                    dataset_name=args.data[0],
                    class_names=class_names
                )
            except Exception as e:
                logger.warning(f"Error in visualize_normal: {e}")

            # Add class distribution visualization from your imported class
            try:
                from utils.visual.train.class_distribution import save_class_distribution
                save_class_distribution(
                    {model_name: all_targets},
                    class_names,
                    args.task_name,
                    args.data[0]
                )
                logger.info("Generated class distribution visualization")
            except Exception as e:
                logger.warning(
                    f"Error in class distribution visualization: {e}")

            # Add ROC and precision-recall curves
            try:
                from utils.visual.train.roc_curve import save_roc_curve
                from utils.visual.train.precision_recall_curve import save_precision_recall_curve

                save_roc_curve(
                    [model_name],
                    {model_name: all_targets},
                    {model_name: all_probs},
                    class_names,
                    args.task_name,
                    args.data[0]
                )

                save_precision_recall_curve(
                    [model_name],
                    {model_name: all_targets},
                    {model_name: all_probs},
                    class_names,
                    args.task_name,
                    args.data[0]
                )
            except Exception as e:
                logger.warning(f"Error generating ROC or PR curves: {e}")

            # For binary classification, also show threshold optimization
            if len(np.unique(all_targets)) == 2:
                try:
                    from utils.visual.train.threshold_optimization import save_threshold_optimization

                    optimal_thresh = save_threshold_optimization(
                        all_targets,
                        all_probs,
                        args.task_name,
                        args.data[0],
                        model_name
                    )
                    logger.info(f"Optimal threshold: {optimal_thresh:.4f}")
                except Exception as e:
                    logger.warning(f"Error in threshold optimization: {e}")

            # If adversarial testing is enabled
            if args.adversarial and args.evaluate_robustness:
                # Create adversarial examples for visualization
                from utils.visual.attack.perturbation_visualization import save_perturbation_visualization
                from utils.visual.attack.adversarial_examples import save_adversarial_examples
                from gan.defense.adv_train import AdversarialTraining
                from itertools import islice

                # Create adversarial trainer
                adv_trainer = AdversarialTraining(
                    model, torch.nn.CrossEntropyLoss(), args
                )

                # Generate and visualize adversarial examples
                vis_data, vis_targets = next(iter(test_loader))
                vis_data = vis_data[:min(5, len(vis_data))].to(device)
                vis_targets = vis_targets[:min(5, len(vis_targets))].to(device)

                with torch.enable_grad():
                    original_data = vis_data.clone()
                    _, adv_data, _ = adv_trainer.attack.attack(
                        original_data, vis_targets)

                adv_examples = (original_data.detach().cpu(),
                                adv_data.detach().cpu(), vis_targets.cpu())

                attack_name = args.attack_type[0] if isinstance(
                    args.attack_type, list) else args.attack_type

                # Use your existing visualization methods
                save_adversarial_examples(
                    adv_examples,
                    [model_name],
                    args.task_name,
                    args.data[0],
                    attack_name
                )

                save_perturbation_visualization(
                    adv_examples,
                    [model_name],
                    args.task_name,
                    args.data[0]
                )

                # Generate robustness curve
                from utils.visual.defense.robustness_evaluation import save_defense_robustness_plot

                # Test model at multiple epsilon values for robustness curve
                epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
                robustness_results = {}

                # Get clean accuracy first
                correct = 0
                total = 0
                test_subset = list(islice(test_loader, 5))

                with torch.no_grad():
                    for data, target in test_subset:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        pred = output.argmax(dim=1)
                        correct += (pred == target).sum().item()
                        total += target.size(0)

                robustness_results[0.0] = correct / total if total > 0 else 0

                # Test with adversarial examples at each epsilon
                for eps in epsilons:
                    args.attack_eps = eps
                    adv_trainer = AdversarialTraining(
                        model, torch.nn.CrossEntropyLoss(), args
                    )

                    correct = 0
                    total = 0

                    for data, target in test_subset:
                        data, target = data.to(device), target.to(device)

                        with torch.enable_grad():
                            _, adv_data, _ = adv_trainer.attack.attack(
                                data, target)

                        with torch.no_grad():
                            output = model(adv_data)
                            pred = output.argmax(dim=1)
                            correct += (pred == target).sum().item()
                            total += target.size(0)

                    robustness_results[eps] = correct / total if total > 0 else 0

                # Plot and save robustness curve using your visualization method
                save_defense_robustness_plot(
                    [model_name],
                    [attack_name],
                    {"robustness": {attack_name: robustness_results}},
                    args.data[0],
                    args.task_name
                )

        except Exception as e:
            logger.warning(f"Error generating additional visualizations: {e}")
            import traceback
            traceback.print_exc()
# Remove duplicated and mis-indented code at the end of the file
    else:
        logger.info(
            "Skipping visualization generation for object detection model")

    logger.info(f"Testing complete. Results saved to {output_dir}")

    # The reason out\test_results is empty after testing:
    # - The code only saves predictions to out\test_results if args.save_predictions is True.
    # - Visualizations and metrics are saved by their respective utility functions, not directly in out\test_results.
    # - For object detection, only logs are printed, no files are saved unless you implement saving in compute_object_detection_metrics or set args.save_predictions=True.
    # - For classification, visualizations are saved by Visualization and other utility functions, usually in their own subfolders (e.g., out/visualizations, out/single_image_tests).

    # To ensure files are saved in out\test_results:
    # 1. Set args.save_predictions = True (either in your config or command line).
    # 2. Check the output paths in your visualization and metrics utilities; update them to save in out\test_results if desired.
    # 3. For object detection, add code to save detection_metrics to a file:
    if args.save_predictions and is_object_detection:
        import json
        with open(os.path.join(output_dir, "detection_metrics.json"), "w") as f:
            json.dump({k: float(v) if hasattr(v, "item") else v for k, v in detection_metrics.items()}, f, indent=2)

# End of main() function
if __name__ == "__main__":
    main()

