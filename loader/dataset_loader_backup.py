from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import logging
from typing import Dict, Tuple
import os
from PIL import Image
from loader.object_detection_dataset import ObjectDetectionDataset, object_detection_collate_fn
import yaml
from class_mapping import CLASS_MAPPING, get_num_classes

# Constants
PROCESSED_DATA_DIR = 'processed_data'


def object_detection_collate(batch):
    """Collate function for object detection datasets."""
    # Filter out invalid batch items
    batch = [item for item in batch if item is not None and len(item) == 2]
    if not batch:
        logging.warning("Empty or invalid batch, returning dummy batch")
        return torch.zeros((1, 3, 224, 224)), {
            'boxes': [torch.empty((0, 4), dtype=torch.float32)],
            'labels': [torch.empty((0,), dtype=torch.long)]
        }

    images, targets = zip(*batch)

    # Stack images
    try:
        images = torch.stack(images)  # Shape: [batch_size, 3, H, W]
    except Exception as e:
        logging.error(f"Error stacking images: {e}")
        raise

    # Process targets to match model expectations
    processed_targets = {
        'boxes': [],
        'labels': []
    }

    for target in targets:
        if isinstance(target, dict):
            boxes = target.get('boxes', torch.empty(
                (0, 4), dtype=torch.float32))
            labels = target.get('labels', torch.empty((0,), dtype=torch.long))
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)

        processed_targets['boxes'].append(boxes)
        processed_targets['labels'].append(labels)

    return images, processed_targets


class DatasetLoader:
    def __init__(self):
        # Add any necessary initialization here
        self.processed_data_path = Path("processed_data")
        # You may want to set self.transform here if needed
        self.transform = None

    def validate_dataset(self, dataset_name):
        """
        Validate that the dataset exists and is structured correctly.
        """
        dataset_path = self.processed_data_path / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {dataset_path}")

        # Different validation for object detection vs classification
        if self._is_object_detection_dataset(dataset_name):
            self._validate_object_detection_structure(dataset_path)
        else:
            self._validate_classification_structure(dataset_path, dataset_name)

        return True

    def _is_object_detection_dataset(self, dataset_name):
        """
        Determine if the dataset is for object detection based on its name.
        """
        od_keywords = ['coco', 'voc', 'yolo', 'cattleface']
        return any(k in str(dataset_name).lower() for k in od_keywords)

    def _validate_object_detection_structure(self, dataset_path: Path):
        """Validate object detection dataset structure"""
        try:
            splits = ['train', 'val', 'test']
            for split in splits:
                split_path = dataset_path / split
                images_path = split_path / 'images'
                annotations_path = split_path / 'annotations'

                if not images_path.exists():
                    raise FileNotFoundError(
                        f"Images directory missing: {images_path}")
                if not annotations_path.exists():
                    raise FileNotFoundError(
                        f"Annotations directory missing: {annotations_path}")

                # Check for some basic files
                image_files = list(images_path.glob('*.*'))
                annotation_files = list(annotations_path.glob('*.txt'))

                if not image_files:
                    logging.warning(f"No image files found in {images_path}")
                if not annotation_files:
                    logging.warning(
                        f"No annotation files found in {annotations_path}")

                # Check for empty or malformed annotation files
                for anno_file in annotation_files:
                    # Adjust extension if needed
                    img_file = images_path / (anno_file.stem + '.jpg')
                    if not img_file.exists():
                        logging.warning(
                            f"No matching image for annotation: {anno_file}")
                    with open(anno_file, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            logging.warning(
                                f"Empty annotation file: {anno_file}")
                        for line in lines:
                            data = line.strip().split()
                            if len(data) < 5:
                                logging.warning(
                                    f"Malformed annotation in {anno_file}: {line}")
        except Exception as e:
            logging.error(
                f"Could not validate object detection structure: {e}")
            raise

    def _validate_classification_structure(self, dataset_path: Path, dataset_name: str):
        """Validate classification dataset structure"""
        try:
            # Check for channel consistency across splits
            splits = ['train', 'val', 'test']
            channels = []
            for split in splits:
                split_path = dataset_path / split
                if split_path.exists():
                    # Find first image to check channels
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir():
                            for img_path in class_dir.glob('*.*'):
                                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                                    with Image.open(img_path) as img:
                                        channels.append(
                                            (split, len(img.getbands()), img.getbands()))
                                    break
                            if channels and channels[-1][0] == split:
                                break

            # Check if all splits have the same number of channels
            if len(set(ch[1] for ch in channels)) > 1:
                channel_info = ", ".join(
                    f"{s}: {c} {b}" for s, c, b in channels)
                logging.warning(
                    f"Inconsistent channel counts detected in {dataset_name}: {channel_info}")
        except Exception as e:
            logging.warning(f"Could not verify channel consistency: {e}")

    def load_data(self, dataset_name, batch_size, num_workers, pin_memory, force_classification=False):
        """Load processed datasets"""
        # Only log once per call, and do not log inside loops or repeatedly
        logging.info(f"Loading {dataset_name} dataset")
        self.validate_dataset(dataset_name)
        dataset_path = self.processed_data_path / dataset_name

        # Check if this is an object detection dataset
        is_od_dataset = self._is_object_detection_dataset(dataset_name)

        # If object detection dataset and not forcing classification
        if is_od_dataset and not force_classification:
            logging.info(f"Loading {dataset_name} as object detection dataset")
            train_dataset = ObjectDetectionDataset(
                image_dir=str(self.processed_data_path /
                              dataset_name / 'train' / 'images'),
                annotation_dir=str(self.processed_data_path /
                                   dataset_name / 'train' / 'annotations'),
                transform=get_transform(is_train=True)
            )
            val_dataset = ObjectDetectionDataset(
                image_dir=str(self.processed_data_path /
                              dataset_name / 'val' / 'images'),
                annotation_dir=str(self.processed_data_path /
                                   dataset_name / 'val' / 'annotations'),
                transform=get_transform(is_train=False)
            )
            test_dataset = ObjectDetectionDataset(
                image_dir=str(self.processed_data_path /
                              dataset_name / 'test' / 'images'),
                annotation_dir=str(self.processed_data_path /
                                   dataset_name / 'test' / 'annotations'),
                transform=get_transform(is_train=False)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size['train'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=object_detection_collate
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size['val'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=object_detection_collate
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size['test'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=object_detection_collate
            )
            logging.info(f"Found {len(train_dataset)} training images")
            logging.info(f"Found {len(val_dataset)} validation images")
            logging.info(f"Found {len(test_dataset)} test images")
            # Get number of classes programmatically
            num_classes = get_num_classes()
            logging.info(f"Classes: {list(range(num_classes))}")

            # Note: Class weights computation moved to training script to avoid hanging
            # during dataset loading. The compute_class_weights function can cause
            # issues with object detection datasets that have complex collate functions.

            return train_loader, val_loader, test_loader

        # If object detection dataset but forcing classification
        elif is_od_dataset and force_classification:
            logging.info(
                f"Loading {dataset_name} as classification dataset (forced)")
            return self._load_object_detection_as_classification(dataset_name, dataset_path, batch_size, num_workers, pin_memory)

        # Regular classification dataset
        else:
            logging.info(f"Loading {dataset_name} as classification dataset")
            return self._load_classification_data(dataset_name, dataset_path, batch_size, num_workers, pin_memory)

    def _load_classification_data(self, dataset_name: str, dataset_path: Path,
                                  batch_size: Dict[str, int], num_workers: int, pin_memory: bool):
        """Load classification datasets"""
        train_dataset = ImageFolder(dataset_path / 'train', self.transform)
        val_dataset = ImageFolder(dataset_path / 'val', self.transform)
        test_dataset = ImageFolder(dataset_path / 'test', self.transform)

        logging.info(f"Loading {dataset_name} classification dataset:")
        logging.info(f"Found {len(train_dataset)} training samples")
        logging.info(f"Found {len(val_dataset)} validation samples")
        logging.info(f"Found {len(test_dataset)} test samples")
        logging.info(f"Classes: {train_dataset.classes}")

        # Check for potentially problematic split sizes
        if len(val_dataset) < 10:
            logging.warning(f"Validation set is very small ({len(val_dataset)} samples). "
                            "Consider using --enforce_split option with dataset_processing.py")
        if len(test_dataset) < 10:
            logging.warning(
                f"Test set is very small ({len(test_dataset)} samples).")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size['train'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size['val'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, val_loader, test_loader

    def _load_object_detection_as_classification(self, dataset_name: str, dataset_path: Path,
                                                 batch_size: Dict[str, int], num_workers: int, pin_memory: bool):
        """Load object detection datasets as classification datasets for classification models"""
        # For object detection datasets used with classification models,
        # we need to check if the processed data has been structured as classification data

        # Check if the dataset has been processed with classification structure
        train_path = dataset_path / 'train'
        if not train_path.exists():
            raise FileNotFoundError(f"Train directory not found: {train_path}")

        # Check if it has class subdirectories (classification structure)
        class_dirs = [d for d in train_path.iterdir() if d.is_dir() and d.name not in [
            'images', 'annotations']]

        if class_dirs:
            # Dataset has been processed with classification structure
            return self._load_classification_data(dataset_name, dataset_path, batch_size, num_workers, pin_memory)
        else:
            # Dataset still has object detection structure, but we need classification
            # This shouldn't happen if dataset_processing.py processes it correctly
            raise ValueError(
                f"Object detection dataset {dataset_name} cannot be used with classification models. "
                f"The dataset structure is not compatible. Please ensure the dataset is processed "
                f"correctly or use an object detection model instead."
            )

    def _load_object_detection_data(self, dataset_name: str, dataset_path: Path,
                                    batch_size: Dict[str, int], num_workers: int, pin_memory: bool):
        """Load object detection datasets"""

        # Define transforms for object detection
        train_transform = get_transform(is_train=True)
        val_transform = get_transform(is_train=False)

        train_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'train' / 'images'),
            annotation_dir=str(dataset_path / 'train' / 'annotations'),
            transform=train_transform
        )
        val_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'val' / 'images'),
            annotation_dir=str(dataset_path / 'val' / 'annotations'),
            transform=val_transform
        )
        test_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'test' / 'images'),
            annotation_dir=str(dataset_path / 'test' / 'annotations'),
            transform=val_transform
        )

        logging.info(f"Loading {dataset_name} object detection dataset:")
        logging.info(f"Found {len(train_dataset)} training images")
        logging.info(f"Found {len(val_dataset)} validation images")
        logging.info(f"Found {len(test_dataset)} test images")
        logging.info(f"Classes: {train_dataset.classes}")

        # Check for potentially problematic split sizes
        if len(val_dataset) < 10:
            logging.warning(
                f"Validation set is very small ({len(val_dataset)} images).")
        if len(test_dataset) < 10:
            logging.warning(
                f"Test set is very small ({len(test_dataset)} images).")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size['train'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=object_detection_collate
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size['val'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=object_detection_collate
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=object_detection_collate
        )

        return train_loader, val_loader, test_loader


class DatasetLoader:
    def __init__(self):
        # Add any necessary initialization here
        self.processed_data_path = Path("processed_data")
        # You may want to set self.transform here if needed
        self.transform = None


def collate_fn(batch):
    """
    Custom collate function for object detection batches
    """
    return tuple(zip(*batch))


def load_dataset(dataset_name, **kwargs):
    # ...existing code...

    if dataset_name.lower() == 'cattleface' or 'cattle' in dataset_name.lower():
        # Object detection dataset
        logging.info(f"Loading {dataset_name} object detection dataset:")

        base_path = os.path.join(PROCESSED_DATA_DIR, dataset_name)

    # ...existing code...


def get_transform(is_train):
    """
    Get transforms for object detection datasets - ensures consistent sizing
    """
    transform_list = []

    if is_train:
        # Data augmentation for training (applied before resize to maintain consistency)
        transform_list.append(transforms.RandomHorizontalFlip(0.5))

    # Ensure consistent image size - resize to (224, 224) to match preprocessing
    # This is CRITICAL for batching - all images must have the same size
    transform_list.append(transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BILINEAR))

    # Convert PIL image to tensor
    transform_list.append(transforms.ToTensor())

    print(
        f"Created transform pipeline: {[type(t).__name__ for t in transform_list]}")

    return transforms.Compose(transform_list)
