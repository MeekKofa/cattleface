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
from loader.object_detection_dataset import ObjectDetectionDataset, collate_fn_object_detection
import yaml
from class_mapping import CLASS_MAPPING, get_num_classes

# Constants - Use dynamic path resolution
# PROCESSED_DATA_DIR now handled by path_config.py


def object_detection_collate(batch):
    """Collate function for object detection datasets."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.zeros(0), torch.zeros(0)

    images, targets = zip(*batch)
    images = torch.stack(images)
    formatted_targets = []
    for target in targets:
        formatted = {
            'boxes': target['boxes'],
            'labels': target['labels'],
            'image_id': target['image_id']
        }
        formatted_targets.append(formatted)
    return images, formatted_targets

# Removed compute_class_weights function as it was causing hanging issues
# with object detection datasets during loading phase.


class DatasetLoader:
    def __init__(self):
        # Use path configuration for dynamic path resolution
        from loader.path_config import get_path_config
        self.path_config = get_path_config()
        self.transform = None

    def _is_object_detection_dataset(self, dataset_name):
        """
        Determine if the dataset is for object detection based on configuration.
        """
        dataset_config = self.path_config.get_dataset_config(dataset_name)
        if dataset_config:
            return dataset_config.get('modality') == 'object_detection'

        # Fallback to keyword matching
        od_keywords = ['coco', 'voc', 'yolo', 'cattleface', 'cattlebody']
        return any(k in str(dataset_name).lower() for k in od_keywords)

    def load_data(self, dataset_name, batch_size, num_workers, pin_memory, processed_path=None):
        logging.info(f"Loading {dataset_name} dataset")
        from loader.path_config import get_dataset_paths, validate_dataset

        # Validate the dataset structure first
        is_valid, message = validate_dataset(dataset_name)
        if not is_valid:
            raise FileNotFoundError(f"Dataset validation failed: {message}")

        # Get the properly structured paths
        dataset_paths = get_dataset_paths(dataset_name)

        # Use ObjectDetectionDataset for this structure
        train_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_paths['train']['images']),
            annotation_dir=str(dataset_paths['train']['labels']),
            transform=get_transform(is_train=True)
        )
        val_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_paths['val']['images']),
            annotation_dir=str(dataset_paths['val']['labels']),
            transform=get_transform(is_train=False)
        )
        test_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_paths['test']['images']),
            annotation_dir=str(dataset_paths['test']['labels']),
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
        logging.info(f"Classes: {train_dataset.classes}")

        return train_loader, val_loader, test_loader

    def _load_classification_data(self, dataset_name: str, dataset_path: Path,
                                  batch_size: Dict[str, int], num_workers: int, pin_memory: bool):
        """Load classification datasets using ObjectDetectionDataset instead of ImageFolder"""
        train_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'train' / 'images'),
            annotation_dir=str(dataset_path / 'train' / 'labels'),
            transform=get_transform(is_train=True)
        )
        val_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'val' / 'images'),
            annotation_dir=str(dataset_path / 'val' / 'labels'),
            transform=get_transform(is_train=False)
        )
        test_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'test' / 'images'),
            annotation_dir=str(dataset_path / 'test' / 'labels'),
            transform=get_transform(is_train=False)
        )

        logging.info(
            f"Loading {dataset_name} classification dataset (object detection format):")
        logging.info(f"Found {len(train_dataset)} training images")
        logging.info(f"Found {len(val_dataset)} validation images")
        logging.info(f"Found {len(test_dataset)} test images")
        logging.info(f"Classes: {train_dataset.classes}")

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
            'images', 'annotations', 'labels']]

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
            # annotation_dir=str(dataset_path / 'train' / 'annotations'),
            labels_dir=str(dataset_path / 'train' / 'labels'),
            transform=train_transform
        )
        val_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'val' / 'images'),
            # annotation_dir=str(dataset_path / 'val' / 'annotations'),
            labels_dir=str(dataset_path / 'val' / 'labels'),
            transform=val_transform
        )
        test_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_path / 'test' / 'images'),
            # annotation_dir=str(dataset_path / 'test' / 'annotations'),
            labels_dir=str(dataset_path / 'test' / 'labels'),
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


class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(
            image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths = [os.path.join(self.image_dir, f)
                            for f in self.image_files]
        self.annotation_paths = [os.path.join(self.annotation_dir, os.path.splitext(f)[
                                              0] + '.txt') for f in self.image_files]
        self.classes = []
        self._extract_classes()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        import numpy as np
        img_path = self.image_paths[index]
        annot_path = self.annotation_paths[index]

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None

        target = {
            'boxes': [],
            'labels': [],
            'image_id': torch.tensor([index], dtype=torch.int64)
        }

        if os.path.exists(annot_path):
            try:
                with open(annot_path, 'r') as f:
                    lines = f.readlines()
                    boxes = []
                    labels = []
                    for line in lines:
                        data = line.strip().split()
                        if len(data) < 5:
                            continue
                        class_id = int(data[0])
                        coords = list(map(float, data[1:5]))
                        x_center, y_center, w, h = coords
                        x_min = x_center - w/2
                        y_min = y_center - h/2
                        x_max = x_center + w/2
                        y_max = y_center + h/2
                        # Clamp coordinates to [0, 1]
                        x_min = max(0.0, min(1.0, x_min))
                        y_min = max(0.0, min(1.0, y_min))
                        x_max = max(0.0, min(1.0, x_max))
                        y_max = max(0.0, min(1.0, y_max))
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)
                    target['boxes'] = boxes
                    target['labels'] = labels
            except Exception as e:
                logging.error(f"Error reading annotations: {e}")

        if self.transform:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=target['boxes'],
                    labels=target['labels']
                )
                image = transformed['image']
                target['boxes'] = torch.tensor(
                    transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.tensor(
                    transformed['labels'], dtype=torch.int64)
            except Exception as e:
                logging.error(f"Transform error: {e}")
                image = torch.zeros((3, 448, 448))
                target = {
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'image_id': torch.tensor([index])
                }

        return image, target

    def _extract_classes(self):
        """Extract unique classes from annotation files"""
        class_set = set()
        for img_file in self.image_files:
            anno_path = os.path.join(
                self.annotation_dir, os.path.splitext(img_file)[0] + '.txt')
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as f:
                    for line in f:
                        data = line.strip().split()
                        if len(data) >= 1:
                            class_set.add(int(data[0]))
        self.classes = sorted(list(class_set))


def collate_fn(batch):
    """
    Custom collate function for object detection batches
    """
    return tuple(zip(*batch))


def load_dataset(dataset_name, **kwargs):
    """
    Legacy function - use DatasetLoader.load_data() instead
    """
    logging.warning(
        "load_dataset() is deprecated, use DatasetLoader.load_data() instead")
    loader = DatasetLoader()
    return loader.load_data(dataset_name, **kwargs)


def get_transform(is_train=True):
    """
    Get transforms for object detection datasets - ensures consistent sizing
    Uses Albumentations for advanced augmentation (mosaic, etc.)
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    target_size = (448, 448)  # Height and width as a tuple

    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=target_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


def clamp_boxes(output):
    # Clamp predicted boxes and filter invalid ones
    output['boxes'][:, [0, 2]] = output['boxes'][:, [0, 2]].clamp(0.0, 1.0)
    output['boxes'][:, [1, 3]] = output['boxes'][:, [1, 3]].clamp(0.0, 1.0)
    valid = (output['boxes'][:, 0] < output['boxes'][:, 2]) & \
            (output['boxes'][:, 1] < output['boxes'][:, 3])
    for k in output:
        output[k] = output[k][valid]
    return output

# Use clamp_boxes in your inference or post-processing pipeline
