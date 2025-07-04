# object_detection_dataset.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union
import xml.etree.ElementTree as ET
import json
import logging
from torchvision import transforms


SUPPORTED_IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
    '.tif', '.tiff', '.webp'
)
SUPPORTED_ANNOTATION_EXTENSIONS = ('.txt', '.xml', '.json')


def parse_annotation(annotation_path):
    """Parse annotation files to extract object classes and bounding boxes"""
    classes = []
    bboxes = []
    if annotation_path.endswith('.txt'):
        # YOLO format: class_id center_x center_y width height (normalized coordinates)
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) >= 5:
                    classes.append(int(data[0]))
                    # YOLO format is normalized (0-1), need to convert to actual coordinates
                    center_x, center_y, width, height = [float(x) for x in data[1:5]]
                    # Convert from center format to corner format and assume image size for now
                    # Note: For proper use, you should pass image dimensions to this function
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    bboxes.append([x1, y1, x2, y2])
    elif annotation_path.endswith('.xml'):
        # Pascal VOC format
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        # Get image dimensions for normalization if needed
        size = root.find('size')
        if size is not None:
            img_width = float(size.find('width').text)
            img_height = float(size.find('height').text)
        else:
            img_width = img_height = 1.0  # fallback
            
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            # Convert class name to integer if it's a string
            if isinstance(cls_name, str):
                # You might want to maintain a class name to ID mapping
                classes.append(hash(cls_name) % 1000)  # Simple hash for demo
            else:
                classes.append(int(cls_name))
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / img_width  # Normalize
            ymin = float(bbox.find('ymin').text) / img_height
            xmax = float(bbox.find('xmax').text) / img_width
            ymax = float(bbox.find('ymax').text) / img_height
            bboxes.append([xmin, ymin, xmax, ymax])
    return classes, bboxes


def find_matching_annotation(image_path, annotation_dir):
    """Find matching annotation file for an image"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for ext in SUPPORTED_ANNOTATION_EXTENSIONS:
        annotation_path = os.path.join(annotation_dir, base_name + ext)
        if os.path.exists(annotation_path):
            return annotation_path
    return None


def get_all_classes(annotation_dir):
    """Get all unique classes from annotation files"""
    class_set = set()
    for root, _, files in os.walk(annotation_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_ANNOTATION_EXTENSIONS):
                annotation_path = os.path.join(root, file)
                classes, _ = parse_annotation(annotation_path)
                class_set.update(classes)
    return sorted(list(class_set))


class ObjectDetectionDataset(Dataset):
    """Dataset class for object detection that returns proper format for YOLO models"""
    
    def __init__(self, image_dir, annotation_file=None, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        
        # Load images
        self.images = []
        self.targets = []
        self.classes = []  # Initialize classes list
        
        if os.path.isdir(image_dir):
            # Load from directory structure
            self._load_from_directory()
        elif annotation_file and os.path.exists(annotation_file):
            # Load from annotation file (COCO format)
            self._load_from_annotations(annotation_file)
        else:
            raise ValueError(f"Invalid image_dir: {image_dir} or annotation_file: {annotation_file}")
        
        # Extract unique classes after loading data
        self._extract_classes()
    
    def _load_from_directory(self):
        """Load images from directory structure with annotation support"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Check if there's an annotations directory
        annotation_dir = os.path.join(os.path.dirname(self.image_dir), 'annotations')
        if not os.path.exists(annotation_dir):
            annotation_dir = os.path.join(self.image_dir, '..', 'annotations')
            if not os.path.exists(annotation_dir):
                annotation_dir = None
        
        for filename in os.listdir(self.image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.image_dir, filename)
                self.images.append(image_path)
                
                # Try to load actual annotations if available
                if annotation_dir:
                    annotation_path = find_matching_annotation(image_path, annotation_dir)
                    if annotation_path:
                        try:
                            classes, bboxes = parse_annotation(annotation_path)
                            
                            # Convert to tensors
                            if bboxes:
                                boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
                                labels_tensor = torch.tensor(classes, dtype=torch.int64)
                                areas = torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes], dtype=torch.float32)
                            else:
                                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                                labels_tensor = torch.zeros((0,), dtype=torch.int64)
                                areas = torch.zeros((0,), dtype=torch.float32)
                            
                            target = {
                                'boxes': boxes_tensor,
                                'labels': labels_tensor,
                                'image_id': len(self.images) - 1,
                                'area': areas,
                                'iscrowd': torch.zeros((len(labels_tensor),), dtype=torch.int64)
                            }
                        except Exception as e:
                            print(f"Error loading annotation for {filename}: {e}")
                            # Fall back to empty target
                            target = {
                                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                                'labels': torch.zeros((0,), dtype=torch.int64),
                                'image_id': len(self.images) - 1,
                                'area': torch.zeros((0,), dtype=torch.float32),
                                'iscrowd': torch.zeros((0,), dtype=torch.int64)
                            }
                    else:
                        # No annotation found - create empty target
                        target = {
                            'boxes': torch.zeros((0, 4), dtype=torch.float32),
                            'labels': torch.zeros((0,), dtype=torch.int64),
                            'image_id': len(self.images) - 1,
                            'area': torch.zeros((0,), dtype=torch.float32),
                            'iscrowd': torch.zeros((0,), dtype=torch.int64)
                        }
                else:
                    # No annotation directory - create dummy target
                    target = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'labels': torch.zeros((0,), dtype=torch.int64),
                        'image_id': len(self.images) - 1,
                        'area': torch.zeros((0,), dtype=torch.float32),
                        'iscrowd': torch.zeros((0,), dtype=torch.int64)
                    }
                
                self.targets.append(target)
    
    def _extract_classes(self):
        """Extract unique classes from loaded targets"""
        class_set = set()
        for target in self.targets:
            if 'labels' in target and len(target['labels']) > 0:
                class_set.update(target['labels'].tolist())
        self.classes = sorted(list(class_set))
        
        # If no classes found, create a default empty list
        if not self.classes:
            self.classes = []
    
    def _load_from_annotations(self, annotation_file):
        """Load from COCO-style annotation file"""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Process annotations - simplified version
        for image_info in annotations.get('images', []):
            image_path = os.path.join(self.image_dir, image_info['file_name'])
            if os.path.exists(image_path):
                self.images.append(image_path)
                
                # Create target dict
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'image_id': image_info['id'],
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.int64)
                }
                self.targets.append(target)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get target - make a proper copy of the dictionary
        target = {}
        for key, value in self.targets[idx].items():
            if isinstance(value, torch.Tensor):
                target[key] = value.clone()
            else:
                target[key] = value
        
        # Apply transforms to image only
        if self.transform:
            image = self.transform(image)
        
        # Apply target transforms if provided (should not use ToTensor on targets)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


def object_detection_collate_fn(batch):
    """Custom collate function for object detection datasets"""
    images, targets = zip(*batch)
    
    # Stack images into a batch tensor
    images = torch.stack(images, 0)
    
    # Keep targets as a list of dictionaries
    # Each target dict contains: boxes, labels, image_id, area, iscrowd
    targets = list(targets)
    
    return images, targets


# Alias for backward compatibility
collate_fn_object_detection = object_detection_collate_fn


def yolo_collate_fn(batch):
    """Alternative collate function that returns lists for YOLO models"""
    images, targets = zip(*batch)
    
    # Return as lists for YOLO compatibility  
    return list(images), list(targets)


def robust_yolo_collate_fn(batch):
    """
    Robust collate function for YOLO models that handles various input formats
    and ensures consistent tensor shapes
    """
    import logging
    import torch.nn.functional as F
    
    images = []
    all_labels = []
    all_bboxes = []
    all_paths = []
    
    # Process each item in the batch
    for item in batch:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            img, labels, bboxes, path = item[:4]
            images.append(img)
            all_labels.append(labels)
            all_bboxes.append(bboxes)
            all_paths.append(path)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Handle (image, target) format from ObjectDetectionDataset
            img, target = item
            images.append(img)
            # Extract labels and boxes from target dict
            if isinstance(target, dict):
                labels = target.get('labels', torch.tensor([]))
                bboxes = target.get('boxes', torch.tensor([]).reshape(0, 4))
            else:
                labels = torch.tensor([])
                bboxes = torch.tensor([]).reshape(0, 4)
            all_labels.append(labels)
            all_bboxes.append(bboxes)
            all_paths.append("unknown_path")
        else:
            logging.warning(f"Unexpected batch item format: {type(item)}, length: {len(item) if hasattr(item, '__len__') else 'N/A'}")
            # Create dummy tensors with proper dimensions to avoid errors
            if len(images) > 0:
                # Clone the first image to ensure consistent dimensions
                dummy_img = images[0].clone()
                dummy_img.zero_()  # Zero out the values
                images.append(dummy_img)
                all_labels.append(torch.tensor([]))
                all_bboxes.append(torch.tensor([]).reshape(0, 4))
                all_paths.append("dummy_path")
            else:
                # Skip this batch entirely if we can't create proper dummy data
                logging.error("Invalid batch format and no valid images to create dummy tensors")
                return torch.empty(0), [], [], []
    
    if not images:
        return torch.empty(0), [], [], []
    
    # Ensure all images are tensors with the same dimension and type
    try:
        # Convert non-tensor images to tensors and ensure all have float dtype
        processed_images = []
        for i, img in enumerate(images):
            if not isinstance(img, torch.Tensor):
                logging.warning(f"Image {i} is not a tensor: {type(img)}")
                # Try to convert to tensor
                try:
                    img = torch.tensor(img, dtype=torch.float32)
                except:
                    logging.error(f"Could not convert image {i} to tensor")
                    continue
            
            # Ensure tensor is float type for interpolation operations
            if img.dtype != torch.float32:
                img = img.to(torch.float32)
            
            # Ensure image has proper dimensions for CHW format
            if len(img.shape) == 2:  # Handle grayscale images
                img = img.unsqueeze(0)  # Add channel dimension
            elif len(img.shape) != 3:
                logging.error(f"Invalid image dimensions {img.shape} for image {i}")
                continue
                
            processed_images.append(img)
        
        if not processed_images:
            logging.error("No valid images after preprocessing")
            return torch.empty(0), [], [], []
            
        images = processed_images
        
        # Get the target size from the first image
        target_shape = images[0].shape
        
        # Resize images to match the target shape if needed
        resized_images = []
        for i, img in enumerate(images):
            if img.shape != target_shape:
                # Ensure we have a batch dimension for interpolate
                if len(img.shape) == 3:  # CHW format
                    # Handle different channel counts
                    if img.shape[0] != target_shape[0]:
                        # Convert to target number of channels
                        if target_shape[0] == 3 and img.shape[0] == 1:
                            img = img.repeat(3, 1, 1)  # Expand grayscale to RGB
                        elif target_shape[0] == 1 and img.shape[0] == 3:
                            img = img.mean(dim=0, keepdim=True)  # Convert RGB to grayscale
                    
                    # Resize height and width
                    if img.shape[1:] != target_shape[1:]:
                        try:
                            img_resized = F.interpolate(
                                img.unsqueeze(0), 
                                size=(target_shape[1], target_shape[2]),
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                            resized_images.append(img_resized)
                        except Exception as e:
                            logging.error(f"Interpolation failed for image {i}: {e}")
                            # Use original image as fallback
                            resized_images.append(img)
                    else:
                        resized_images.append(img)
                else:
                    logging.error(f"Unexpected image shape: {img.shape}")
                    continue
            else:
                resized_images.append(img)
        
        if not resized_images:
            logging.error("No valid images after resizing")
            return torch.empty(0), [], [], []
            
        # Try to stack the images, with additional error handling
        try:
            # Check for consistent shapes before stacking
            shapes = [img.shape for img in resized_images]
            if len(set(shapes)) > 1:
                logging.error(f"Inconsistent shapes after resizing: {shapes}")
                # Try to ensure consistent shapes one more time
                uniform_images = []
                for img in resized_images:
                    if img.shape != target_shape:
                        # Reshape or pad to match target
                        if len(img.shape) == 3 and len(target_shape) == 3:
                            # Create new tensor with target shape
                            new_img = torch.zeros(target_shape, dtype=img.dtype)
                            # Copy as much as possible from original
                            c = min(img.shape[0], target_shape[0])
                            h = min(img.shape[1], target_shape[1])
                            w = min(img.shape[2], target_shape[2])
                            new_img[:c, :h, :w] = img[:c, :h, :w]
                            uniform_images.append(new_img)
                        else:
                            continue
                    else:
                        uniform_images.append(img)
                        
                if not uniform_images:
                    logging.error("No valid images after shape correction")
                    return torch.empty(0), [], [], []
                    
                resized_images = uniform_images
            
            images = torch.stack(resized_images, 0)
        except RuntimeError as e:
            logging.error(f"Stack failed with error: {e}")
            for i, img in enumerate(resized_images):
                logging.error(f"Image {i} shape: {img.shape}, dtype: {img.dtype}")
            # Return empty batch as fallback
            return torch.empty(0), [], [], []
            
    except Exception as e:
        logging.error(f"Error in collate function: {e}")
        logging.error(f"Image shapes: {[img.shape if isinstance(img, torch.Tensor) else type(img) for img in images]}")
        # Return empty tensors to avoid crashing
        return torch.empty(0), [], [], []
    
    # For training loop compatibility with YOLO models
    targets = []
    for labels, bboxes in zip(all_labels, all_bboxes):
        # Ensure labels is a tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels) if labels else torch.tensor([])
        
        # Ensure bboxes is a tensor with proper shape (n, 4)
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.tensor(bboxes) if bboxes else torch.tensor([]).reshape(0, 4)
        elif len(bboxes.shape) == 1 and bboxes.numel() > 0:
            # Handle case where bboxes might be flattened
            bboxes = bboxes.reshape(-1, 4)
        elif len(bboxes.shape) == 0:
            # Handle scalar tensor case
            bboxes = torch.tensor([]).reshape(0, 4)
        
        # Create dict that matches YOLO model's expected format
        target_dict = {
            'labels': labels,
            'bboxes': bboxes
        }
        targets.append(target_dict)
    
    return images, targets


class ObjectDetectionSplitter:
    """
    Utility class to split object detection datasets while maintaining
    annotation file relationships.
    """
    
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_paths = self._get_image_paths()
    
    def _get_image_paths(self):
        """Get all image paths"""
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    def split_dataset(self, output_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
        """
        Split dataset into train/val/test while maintaining image-annotation pairs.
        
        Args:
            output_dir: Directory to save split datasets
            split_ratios: Tuple of (train, val, test) ratios
            seed: Random seed for reproducibility
        """
        import random
        import shutil
        from pathlib import Path
        
        random.seed(seed)
        np.random.seed(seed)
        
        output_path = Path(output_dir)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'annotations').mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split image paths
        shuffled_paths = self.image_paths.copy()
        random.shuffle(shuffled_paths)
        
        n_total = len(shuffled_paths)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        train_paths = shuffled_paths[:n_train]
        val_paths = shuffled_paths[n_train:n_train + n_val]
        test_paths = shuffled_paths[n_train + n_val:]
        
        # Copy files to respective splits
        splits = {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }
        
        for split_name, paths in splits.items():
            for img_path in paths:
                # Copy image
                img_filename = os.path.basename(img_path)
                dst_img_path = output_path / split_name / 'images' / img_filename
                shutil.copy2(img_path, dst_img_path)
                
                # Copy corresponding annotation
                annotation_path = find_matching_annotation(img_path, self.annotation_dir)
                if annotation_path:
                    ann_filename = os.path.basename(annotation_path)
                    dst_ann_path = output_path / split_name / 'annotations' / ann_filename
                    shutil.copy2(annotation_path, dst_ann_path)
        
        return splits
    
    def split_dataset_with_progress(self, output_dir, split_ratios=(0.7, 0.15, 0.15), seed=42, transforms=None):
        """
        Split dataset into train/val/test while maintaining image-annotation pairs with progress tracking.
        Now includes image transformation support.
        
        Args:
            output_dir: Directory to save split datasets
            split_ratios: Tuple of (train, val, test) ratios
            seed: Random seed for reproducibility
            transforms: Dictionary of transforms for each split (e.g., {'train': transform, 'val': transform, 'test': transform})
        """
        import random
        import shutil
        from pathlib import Path
        from tqdm import tqdm
        from PIL import Image

        random.seed(seed)
        np.random.seed(seed)

        output_path = Path(output_dir)

        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'annotations').mkdir(parents=True, exist_ok=True)

        # Shuffle and split image paths
        shuffled_paths = self.image_paths.copy()
        random.shuffle(shuffled_paths)

        n_total = len(shuffled_paths)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        train_paths = shuffled_paths[:n_train]
        val_paths = shuffled_paths[n_train:n_train + n_val]
        test_paths = shuffled_paths[n_train + n_val:]

        # Process files to respective splits with progress tracking and transformations
        splits = {
            'train': train_paths,
            'val': val_paths,
            'test': test_paths
        }

        total_files = n_total * 2  # Each image has a corresponding annotation
        processed_files = 0

        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for split_name, paths in splits.items():
                # Get transform for this split
                transform = transforms.get(split_name) if transforms else None

                for img_path in paths:
                    # Process and save image (with transformation if provided)
                    img_filename = os.path.basename(img_path)
                    dst_img_path = output_path / split_name / 'images' / img_filename

                    if transform:
                        # Load, transform, and save image
                        try:
                            with Image.open(img_path) as img:
                                # Convert to RGB if necessary
                                if img.mode != "RGB":
                                    img = img.convert("RGB")

                                # Apply transforms
                                transformed_img = transform(img)

                                # Convert tensor back to PIL for saving
                                if hasattr(transformed_img, 'permute'):  # It's a tensor
                                    from torchvision import transforms as T
                                    transformed_img = T.ToPILImage()(transformed_img)

                                # Save transformed image
                                transformed_img.save(dst_img_path)
                        except Exception as e:
                            print(f"Error processing image {img_path}: {e}")
                            # Fallback to copying
                            shutil.copy2(img_path, dst_img_path)
                    else:
                        # Just copy the image without transformation
                        shutil.copy2(img_path, dst_img_path)

                    processed_files += 1
                    pbar.update(1)
                    pbar.set_description(f"Processing {split_name} images")

                    # Copy corresponding annotation
                    annotation_path = find_matching_annotation(img_path, self.annotation_dir)
                    if annotation_path:
                        ann_filename = os.path.basename(annotation_path)
                        dst_ann_path = output_path / split_name / 'annotations' / ann_filename
                        shutil.copy2(annotation_path, dst_ann_path)
                        processed_files += 1
                        pbar.update(1)
                        pbar.set_description(f"Copying {split_name} annotations")
                    else:
                        # Still update progress bar even if no annotation found
                        pbar.update(1)

        return splits
