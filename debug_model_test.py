#!/usr/bin/env python3
"""
Simple test to debug the VGGYOLOv8 model directly
"""
from model.myccc.vgg_yolov8 import VGGYOLOv8
import torch
import logging
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_model():
    """Test the VGGYOLOv8 model directly"""
    print("=" * 60)
    print("DIRECT MODEL TEST")
    print("=" * 60)

    # Create model
    model = VGGYOLOv8(num_classes=20, pretrained=False)
    model.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Model training mode: {model.training}")

    # Test forward pass
    try:
        print("\n--- FORWARD PASS TEST ---")
        with torch.no_grad():
            outputs = model(dummy_input, None)  # No targets = inference mode

        print(f"Output type: {type(outputs)}")
        if isinstance(outputs, list):
            print(f"Number of predictions: {len(outputs)}")
            for i, pred in enumerate(outputs):
                if isinstance(pred, dict):
                    print(f"  Prediction {i}:")
                    for key, value in pred.items():
                        if isinstance(value, torch.Tensor):
                            print(f"    {key}: {value.shape}")
                        else:
                            print(f"    {key}: {value}")
        else:
            print(f"Unexpected output type: {type(outputs)}")

    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Test the detection head directly
    try:
        print("\n--- DETECTION HEAD TEST ---")
        with torch.no_grad():
            features = model.features(dummy_input)
            print(f"Features shape: {features.shape}")

            predictions = model.detection_head(features)
            print(f"Raw predictions shape: {predictions.shape}")

            # Test _convert_to_detections directly
            detections = model._convert_to_detections(predictions)
            print(f"Converted detections: {len(detections)} predictions")

    except Exception as e:
        print(f"ERROR in detection head test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()
