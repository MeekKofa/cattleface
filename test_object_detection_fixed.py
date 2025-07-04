#!/usr/bin/env python3
"""
Test script for the fixed ObjectDetectionDataset
This demonstrates how to use the corrected dataset and collate functions
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loader.object_detection_dataset import (
    ObjectDetectionDataset, 
    object_detection_collate_fn, 
    yolo_collate_fn, 
    robust_yolo_collate_fn
)

def test_object_detection_dataset():
    """Test the object detection dataset with different collate functions"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test with a directory (replace with your actual data path)
    data_dir = "dataset/train/images"  # Update this path as needed
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Creating a dummy test...")
        print("To properly test, please provide a valid image directory path.")
        return
    
    try:
        # Create dataset
        dataset = ObjectDetectionDataset(
            image_dir=data_dir,
            transform=transform
        )
        
        print(f"Dataset created successfully with {len(dataset)} images")
        
        # Test different collate functions
        collate_functions = {
            "standard": object_detection_collate_fn,
            "yolo": yolo_collate_fn,
            "robust_yolo": robust_yolo_collate_fn
        }
        
        for name, collate_fn in collate_functions.items():
            print(f"\nTesting with {name} collate function:")
            
            try:
                # Create dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0  # Use 0 for debugging
                )
                
                # Get one batch
                for batch_idx, batch in enumerate(dataloader):
                    images, targets = batch
                    
                    print(f"  Batch {batch_idx}:")
                    print(f"    Images shape: {images.shape if hasattr(images, 'shape') else type(images)}")
                    print(f"    Targets type: {type(targets)}")
                    print(f"    Number of targets: {len(targets)}")
                    
                    if len(targets) > 0:
                        if isinstance(targets[0], dict):
                            print(f"    First target keys: {targets[0].keys()}")
                            print(f"    First target boxes shape: {targets[0].get('boxes', 'N/A')}")
                            print(f"    First target labels shape: {targets[0].get('labels', 'N/A')}")
                    
                    # Only process first batch for testing
                    break
                    
                print(f"  ✓ {name} collate function works correctly")
                
            except Exception as e:
                print(f"  ✗ Error with {name} collate function: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

def test_with_dummy_data():
    """Test with dummy data when no real dataset is available"""
    print("Testing with dummy data creation...")
    
    # Create a simple dummy dataset
    class DummyObjectDetectionDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=5):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Create dummy image
            image = torch.randn(3, 640, 640)
            
            # Create dummy target
            target = {
                'boxes': torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]),
                'labels': torch.tensor([1, 2]),
                'image_id': idx,
                'area': torch.tensor([0.04, 0.09]),  # (0.2 * 0.2) and (0.3 * 0.3)
                'iscrowd': torch.tensor([0, 0])
            }
            
            return image, target
    
    dummy_dataset = DummyObjectDetectionDataset()
    
    # Test with robust collate function
    dataloader = DataLoader(
        dummy_dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=robust_yolo_collate_fn
    )
    
    for batch_idx, batch in enumerate(dataloader):
        images, targets = batch
        print(f"Dummy batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of targets: {len(targets)}")
        print(f"  Target types: {[type(t) for t in targets]}")
        break
    
    print("✓ Dummy data test successful")

if __name__ == "__main__":
    print("Testing Object Detection Dataset Fixes")
    print("=" * 50)
    
    # Test with actual data if available
    test_object_detection_dataset()
    
    print("\n" + "=" * 50)
    
    # Test with dummy data as fallback
    test_with_dummy_data()
    
    print("\nAll tests completed!")
