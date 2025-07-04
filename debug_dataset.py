#!/usr/bin/env python3
"""
Debug script to test the object detection data loading
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath('.'))

from loader.object_detection_dataset import ObjectDetectionDataset, collate_fn_object_detection
from torchvision import transforms

def test_single_batch():
    """Test loading a single batch from the dataset"""
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    try:
        dataset = ObjectDetectionDataset(
            image_dir='processed_data/cattleface/train/images',
            annotation_dir='processed_data/cattleface/train/annotations',
            transform=transform
        )
        
        print(f"Dataset created successfully with {len(dataset)} images")
        
        # Test a single item
        if len(dataset) > 0:
            item = dataset[0]
            print(f"Single item structure: {type(item)}, length: {len(item)}")
            
            img, labels, bboxes, path = item
            print(f"Image shape: {img.shape}")
            print(f"Image type: {type(img)}")
            print(f"Labels: {labels}")
            print(f"Bboxes: {bboxes}")
            print(f"Path: {path}")
        
        # Create data loader with batch size 2
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate_fn_object_detection
            )
            
            print(f"DataLoader created successfully")
            
            # Try to get first batch
            for batch_idx, batch in enumerate(dataloader):
                print(f"Batch {batch_idx}:")
                images, labels, bboxes, paths = batch
                print(f"  Images shape: {images.shape if isinstance(images, torch.Tensor) else type(images)}")
                print(f"  Labels: {len(labels)} items")
                print(f"  Bboxes: {len(bboxes)} items")
                print(f"  Paths: {len(paths)} items")
                
                if batch_idx >= 2:  # Only test first 3 batches
                    break
                    
        except Exception as e:
            print(f"Error creating DataLoader or getting batch: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_batch()
