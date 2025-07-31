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

# Constants
PROCESSED_DATA_DIR = 'processed_data'


def object_detection_collate(batch):
    """Collate function for object detection datasets."""
    # Filter out invalid batch items
    batch = [item for item in batch if item is not None and len(
        item) == 2 and isinstance(item[1], dict)]
    if not batch:
        logging.warning("Empty or invalid batch, returning dummy batch")
        return torch.zeros((1, 3, 224, 224)), {
            'boxes': torch.zeros((1, 1, 4), dtype=torch.float32),
            'labels': torch.zeros((1, 1), dtype=torch.int64),
            'area': torch.zeros((1, 1), dtype=torch.float32),
            'iscrowd': torch.zeros((1, 1), dtype=torch.uint8),
            'image_id': torch.zeros((1,), dtype=torch.int64)
        }

    images, targets = zip(*batch)
    try:
        images = torch.stack(images)  # Shape: [batch_size, 3, 224, 224]
    except Exception as e:
        logging.error(f"Error stacking images: {e}")
        raise

    # Find maximum number of boxes
    max_boxes = max(t['boxes'].shape[0]
                    for t in targets if t['boxes'].numel() > 0)
    max_boxes = max(max_boxes, 1)  # Ensure at least one box

    batch_boxes = []
    batch_labels = []
    batch_areas = []
    batch_iscrowd = []
    batch_image_ids = []

    for idx, t in enumerate(targets):
        try:
            boxes = t['boxes'].to(dtype=torch.float32)
            labels = t['labels'].to(dtype=torch.int64)
            areas = t['area'].to(dtype=torch.float32)
            iscrowd = t['iscrowd'].to(dtype=torch.uint8)
            image_id = t['image_id'].to(dtype=torch.int64)

            n = boxes.shape[0]
            if n == 0:
                boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
                labels = torch.zeros((max_boxes,), dtype=torch.int64)
                areas = torch.zeros((max_boxes,), dtype=torch.float32)
                iscrowd = torch.zeros((max_boxes,), dtype=torch.uint8)
            elif n < max_boxes:
                pad_boxes = torch.zeros(
                    (max_boxes - n, 4), dtype=torch.float32)
                boxes = torch.cat([boxes, pad_boxes], dim=0)
                pad_labels = torch.zeros((max_boxes - n,), dtype=torch.int64)
                labels = torch.cat([labels, pad_labels], dim=0)
                pad_areas = torch.zeros((max_boxes - n,), dtype=torch.float32)
                areas = torch.cat([areas, pad_areas], dim=0)
                pad_iscrowd = torch.zeros((max_boxes - n,), dtype=torch.uint8)
                iscrowd = torch.cat([iscrowd, pad_iscrowd], dim=0)

            if boxes.shape[0] != max_boxes:
                logging.error(
                    f"Sample {idx}: Boxes shape mismatch, got {boxes.shape}, expected [{max_boxes}, 4]")
                raise ValueError(f"Boxes shape mismatch for sample {idx}")

            batch_boxes.append(boxes)
            batch_labels.append(labels)
            batch_areas.append(areas)
            batch_iscrowd.append(iscrowd)
            batch_image_ids.append(image_id)

        except Exception as e:
            logging.error(
                f"Error processing target for sample {idx}: {t}, Error: {e}")
            raise

    try:
        # Shape: [batch_size, max_boxes, 4]
        batch_boxes = torch.stack(batch_boxes)
        # Shape: [batch_size, max_boxes]
        batch_labels = torch.stack(batch_labels)
        # Shape: [batch_size, max_boxes]
        batch_areas = torch.stack(batch_areas)
        # Shape: [batch_size, max_boxes]
        batch_iscrowd = torch.stack(batch_iscrowd)
        batch_image_ids = torch.stack(batch_image_ids)  # Shape: [batch_size]
    except Exception as e:
        logging.error(f"Stacking error: {e}")
        for i, (boxes, labels) in enumerate(zip(batch_boxes, batch_labels)):
            logging.error(
                f"Sample {i}: boxes={boxes.shape}, labels={labels.shape}")
        raise

    return images, {
        'boxes': batch_boxes,
        'labels': batch_labels,
        'area': batch_areas,
        'iscrowd': batch_iscrowd,
        'image_id': batch_image_ids
    }

# Removed compute_class_weights function as it was causing hanging issues
# with object detection datasets during loading phase.


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


class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        # Get list of image files
        self.image_files = [f for f in os.listdir(
            image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths = [os.path.join(self.image_dir, f)
                            for f in self.image_files]
        self.annotation_paths = [os.path.join(self.annotation_dir, os.path.splitext(f)[
                                              0] + '.txt') for f in self.image_files]

        # Initialize classes attribute
        self.classes = []
        self._extract_classes()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        annot_path = self.annotation_paths[index]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None

        target = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
            'image_id': torch.tensor([index], dtype=torch.int64)
        }

        try:
            with open(annot_path, 'r') as f:
                lines = f.readlines()
                boxes = []
                labels = []
                areas = []
                for line in lines:
                    data = line.strip().split()
                    if len(data) != 5:
                        logging.warning(
                            f"Invalid annotation in {annot_path}: {line.strip()}")
                        continue
                    try:
                        class_id = int(data[0])
                        # Consolidate class using mapping
                        mapped_class = CLASS_MAPPING.get(class_id, class_id)

                        # Use programmatic validation instead of hardcoded range
                        max_class = get_num_classes() - 1  # 0-indexed
                        if not (0 <= mapped_class <= max_class):
                            logging.warning(
                                f"Invalid mapped class {mapped_class} (max: {max_class}) in {annot_path}: {line.strip()}")
                            continue
                        x_center, y_center, width, height = map(
                            float, data[1:5])
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            logging.warning(
                                f"Non-normalized coordinates in {annot_path}: {line.strip()}")
                            continue
                        if width <= 0 or height <= 0:
                            logging.warning(
                                f"Invalid box dimensions in {annot_path}: {line.strip()}")
                            continue
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        boxes.append([x1, y1, x2, y2])
                        labels.append(mapped_class)
                        areas.append(width * height * 224 * 224)  # Pixel area
                    except ValueError as e:
                        logging.warning(
                            f"Error parsing annotation in {annot_path}: {e}")
                        continue

                if boxes:
                    target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                    target['labels'] = torch.tensor(labels, dtype=torch.int64)
                    target['area'] = torch.tensor(areas, dtype=torch.float32)
                    target['iscrowd'] = torch.zeros(
                        len(labels), dtype=torch.uint8)
                else:
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
                    target['area'] = torch.zeros((0,), dtype=torch.float32)
                    target['iscrowd'] = torch.zeros((0,), dtype=torch.uint8)

        except Exception as e:
            logging.warning(f"Error reading annotation {annot_path}: {e}")
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.uint8)

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
                        if len(data) >= 1:  # At least class_id
                            class_set.add(int(data[0]))
        self.classes = sorted(list(class_set))


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
