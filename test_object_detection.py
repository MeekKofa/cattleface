#!/usr/bin/env python3
"""
Test script for object detection dataset integration
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loader.dataset_handler import DatasetHandler
from loader.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO)

def test_object_detection_integration():
    """Test the object detection dataset integration"""
    dataset_name = "cattleface"
    
    try:
        # Test DatasetHandler
        print("=" * 50)
        print("Testing DatasetHandler for object detection...")
        handler = DatasetHandler(dataset_name)
        
        print(f"Dataset: {handler.dataset_name}")
        print(f"Structure type: {handler.structure_type}")
        print(f"Is object detection: {handler.is_object_detection_dataset()}")
        
        if handler.dataset_config:
            print(f"Images dir: {handler.dataset_config['structure']['images']}")
            print(f"Annotations dir: {handler.dataset_config['structure']['annotations']}")
        
        # Test DatasetLoader
        print("=" * 50)
        print("Testing DatasetLoader for object detection...")
        loader = DatasetLoader()
        
        print(f"Is object detection: {loader._is_object_detection_dataset(dataset_name)}")
        
        dataset_config = loader._get_dataset_config(dataset_name)
        if dataset_config:
            print(f"Dataset config found: {dataset_config['name']}")
            print(f"Modality: {dataset_config.get('modality', 'unknown')}")
        
        print("=" * 50)
        print("Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_object_detection_integration()
    sys.exit(0 if success else 1)
