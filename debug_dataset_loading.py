#!/usr/bin/env python3
"""
Debug script to verify data loading and class mapping for cattleface dataset
"""

import logging
from class_mapping import CLASS_MAPPING, get_num_classes
from loader.dataset_loader import DatasetLoader
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def main():
    print("=== Cattleface Dataset Debug Script ===")

    # Test class mapping
    print(f"\n1. Class Mapping Test:")
    print(f"Total unique classes in mapping: {get_num_classes()}")
    print(f"Sample mappings:")
    sample_ids = [0, 105, 288, 305, 306, 350, 380]
    for class_id in sample_ids:
        mapped_id = CLASS_MAPPING.get(class_id, -1)
        print(f"  Raw ID {class_id} -> Mapped ID {mapped_id}")

    # Test dataset loading
    print(f"\n2. Dataset Loading Test:")
    try:
        dataset_loader = DatasetLoader()
        train_loader, val_loader, test_loader = dataset_loader.load_data(
            dataset_name='cattleface',
            batch_size={'train': 2, 'val': 1, 'test': 1},
            num_workers=0,  # Use 0 for debugging
            pin_memory=False
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test a single batch
        print(f"\n3. Batch Content Test:")
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Targets type: {type(targets)}")

            if isinstance(targets, dict):
                print(f"  Target keys: {targets.keys()}")
                if 'boxes' in targets:
                    print(f"  Boxes: {len(targets['boxes'])} items")
                    for i, boxes in enumerate(targets['boxes']):
                        print(f"    Image {i}: {boxes.shape} boxes")
                        if boxes.numel() > 0:
                            print(f"      First box: {boxes[0]}")

                if 'labels' in targets:
                    print(f"  Labels: {len(targets['labels'])} items")
                    for i, labels in enumerate(targets['labels']):
                        print(f"    Image {i}: {labels.shape} labels")
                        if labels.numel() > 0:
                            print(f"      Labels: {labels}")
                            print(
                                f"      Label range: {labels.min().item()} to {labels.max().item()}")

            # Only test first batch
            break

        print(f"\n4. Data Integrity Check:")
        total_samples = 0
        total_objects = 0
        label_counts = {}

        for loader_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            loader_samples = 0
            loader_objects = 0

            for images, targets in loader:
                loader_samples += images.shape[0]

                if isinstance(targets, dict) and 'labels' in targets:
                    for labels in targets['labels']:
                        loader_objects += labels.numel()
                        for label in labels:
                            label_id = label.item()
                            label_counts[label_id] = label_counts.get(
                                label_id, 0) + 1

            print(
                f"  {loader_name.upper()}: {loader_samples} samples, {loader_objects} objects")
            total_samples += loader_samples
            total_objects += loader_objects

        print(f"  TOTAL: {total_samples} samples, {total_objects} objects")
        print(f"  Label distribution: {dict(sorted(label_counts.items()))}")

        # Verify all labels are within expected range
        if label_counts:
            min_label = min(label_counts.keys())
            max_label = max(label_counts.keys())
            expected_classes = get_num_classes()

            print(f"\n5. Label Validation:")
            print(f"  Expected classes: 0 to {expected_classes-1}")
            print(f"  Actual label range: {min_label} to {max_label}")

            if min_label < 0 or max_label >= expected_classes:
                print(f"  ⚠️  WARNING: Labels outside expected range!")
            else:
                print(f"  ✓ All labels within expected range")

        print(f"\n✓ Dataset debug completed successfully!")

    except Exception as e:
        print(f"✗ Error during dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
