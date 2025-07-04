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

# Constants
PROCESSED_DATA_DIR = 'processed_data'


def object_detection_collate(batch):
    """
    Collate function for object detection datasets.
    Handles (image, target) pairs where target is a dictionary.
    Ensures all images have the same size.
    """
    import torch
    import torch.nn.functional as F
    
    images = []
    targets = []
    
    for item in batch:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            image, target = item
            images.append(image)
            targets.append(target)
        else:
            # Handle unexpected batch format
            print(f"Warning: Unexpected batch item format: {type(item)}")
            continue
    
    if not images:
        print("Warning: No valid images in batch")
        return torch.empty(0), []
    
    # Ensure all images have the same size before stacking
    try:
        # Check if all images have the same shape
        shapes = [img.shape for img in images]
        unique_shapes = list(set(shapes))
        
        if len(unique_shapes) > 1:
            print(f"Warning: Images have different shapes: {unique_shapes}")
            print(f"Individual shapes: {shapes}")
            
            # Find the most common shape or use the first one
            target_shape = shapes[0]
            print(f"Resizing all images to: {target_shape}")
            
            # Resize all images to the same size
            resized_images = []
            for i, img in enumerate(images):
                if img.shape != target_shape:
                    print(f"Resizing image {i} from {img.shape} to {target_shape}")
                    # Use interpolation to resize
                    if len(img.shape) == 3:  # CHW format
                        img_resized = F.interpolate(
                            img.unsqueeze(0), 
                            size=(target_shape[1], target_shape[2]),
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                        resized_images.append(img_resized)
                    else:
                        print(f"Error: Unexpected image dimensions: {img.shape}")
                        resized_images.append(img)
                else:
                    resized_images.append(img)
            images = resized_images
        
        # Final check before stacking
        final_shapes = [img.shape for img in images]
        if len(set(final_shapes)) > 1:
            print(f"Error: Still have inconsistent shapes after resizing: {final_shapes}")
            # Create dummy tensors with consistent shape
            target_shape = final_shapes[0]
            consistent_images = []
            for img in images:
                if img.shape == target_shape:
                    consistent_images.append(img)
                else:
                    # Create a zero tensor with the target shape
                    dummy_img = torch.zeros(target_shape, dtype=img.dtype)
                    consistent_images.append(dummy_img)
                    print(f"Warning: Created dummy tensor with shape {target_shape}")
            images = consistent_images
        
        # Stack images into a batch tensor
        images_tensor = torch.stack(images, 0)
        
        # Debug: Print target information
        print(f"Batch debug info:")
        print(f"  - Images tensor shape: {images_tensor.shape}")
        print(f"  - Number of targets: {len(targets)}")
        for i, target in enumerate(targets):
            if isinstance(target, dict) and 'boxes' in target and 'labels' in target:
                print(f"  - Target {i}: boxes {target['boxes'].shape}, labels {target['labels'].shape}")
        
        # For object detection, targets remain as a list of dictionaries
        # Each target dict contains: boxes, labels, image_id, area, iscrowd
        # We DON'T stack the targets because each image has different numbers of objects
        
        return images_tensor, targets
        
    except Exception as e:
        print(f"Error in collate function: {e}")
        print(f"Image shapes: {[img.shape if hasattr(img, 'shape') else type(img) for img in images]}")
        print(f"Image types: {[type(img) for img in images]}")
        # Return empty batch to avoid crashing
        return torch.empty(0), []

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
            raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

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
                    raise FileNotFoundError(f"Images directory missing: {images_path}")
                if not annotations_path.exists():
                    raise FileNotFoundError(f"Annotations directory missing: {annotations_path}")
                
                # Check for some basic files
                image_files = list(images_path.glob('*.*'))
                annotation_files = list(annotations_path.glob('*.*'))
                
                if not image_files:
                    logging.warning(f"No image files found in {images_path}")
                if not annotation_files:
                    logging.warning(f"No annotation files found in {annotations_path}")
        except Exception as e:
            logging.warning(f"Could not validate object detection structure: {e}")

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

    def load_data(self, dataset_name: str, batch_size: Dict[str, int],
                  num_workers: int = 0, pin_memory: bool = False, 
                  force_classification: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load processed datasets"""
        self.validate_dataset(dataset_name)
        dataset_path = self.processed_data_path / dataset_name

        # Handle object detection datasets differently
        if self._is_object_detection_dataset(dataset_name):
            if force_classification:
                # Use object detection dataset as classification dataset
                return self._load_object_detection_as_classification(dataset_name, dataset_path, batch_size, num_workers, pin_memory)
            else:
                return self._load_object_detection_data(dataset_name, dataset_path, batch_size, num_workers, pin_memory)
        else:
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
        class_dirs = [d for d in train_path.iterdir() if d.is_dir() and d.name not in ['images', 'annotations']]
        
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
            logging.warning(f"Validation set is very small ({len(val_dataset)} images).")
        if len(test_dataset) < 10:
            logging.warning(f"Test set is very small ({len(test_dataset)} images).")

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
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Initialize classes attribute
        self.classes = []
        self._extract_classes()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load annotation
        anno_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.txt')
        boxes = []
        labels = []
        
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) >= 5:  # class_id, x_center, y_center, width, height
                        label = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
                        
                        # Convert from normalized YOLO format to [x1, y1, x2, y2] format
                        x1 = (x_center - width/2)
                        y1 = (y_center - height/2)
                        x2 = (x_center + width/2)
                        y2 = (y_center + height/2)
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(label)
        
        # Convert to tensor format
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transform:
            # Apply transform only to image, not target
            image = self.transform(image)
        
        return image, target
    
    def _extract_classes(self):
        """Extract unique classes from annotation files"""
        class_set = set()
        for img_file in self.image_files:
            anno_path = os.path.join(self.annotation_dir, os.path.splitext(img_file)[0] + '.txt')
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
        
        # Define transforms
        transform = get_transform(is_train=True)
        val_transform = get_transform(is_train=False)
        
        # Create datasets
        train_dataset = ObjectDetectionDataset(
            image_dir=os.path.join(base_path, 'train', 'images'),
            annotation_dir=os.path.join(base_path, 'train', 'annotations'),
            transform=transform
        )
        
        val_dataset = ObjectDetectionDataset(
            image_dir=os.path.join(base_path, 'val', 'images'),
            annotation_dir=os.path.join(base_path, 'val', 'annotations'),
            transform=val_transform
        )
        
        test_dataset = ObjectDetectionDataset(
            image_dir=os.path.join(base_path, 'test', 'images'),
            annotation_dir=os.path.join(base_path, 'test', 'annotations'),
            transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=kwargs.get('train_batch', 32),
            shuffle=True,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            collate_fn=object_detection_collate  # Use proper object detection collate function
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=kwargs.get('test_batch', 32),
            shuffle=False,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            collate_fn=object_detection_collate  # Use proper object detection collate function
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=kwargs.get('test_batch', 32),
            shuffle=False,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            collate_fn=object_detection_collate  # Use proper object detection collate function
        )
        
        # Get classes
        classes = list(range(384))  # Based on the error message showing 384 classes (0-383)
        
        logging.info(f"Found {len(train_dataset)} training images")
        logging.info(f"Found {len(val_dataset)} validation images")
        logging.info(f"Found {len(test_dataset)} test images")
        logging.info(f"Classes: {classes}")
        
        return train_loader, val_loader, test_loader, classes
        
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
    transform_list.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR))
    
    # Convert PIL image to tensor
    transform_list.append(transforms.ToTensor())
    
    print(f"Created transform pipeline: {[type(t).__name__ for t in transform_list]}")
    
    return transforms.Compose(transform_list)
