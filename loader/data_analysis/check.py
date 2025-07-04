# check.py  ## Check Datasets before preprocessing

import logging
import os
import gc
import csv
import cv2
import torch
import numpy as np
import hashlib
from datetime import datetime
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter, defaultdict
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Directory to save analysis figures
FIGURES_DIR = "loader/data_analysis/analysis/analysis_figures"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Define filenames for results
TXT_RESULTS_FILE = "loader/data_analysis/analysis/dataset_analysis_results.txt"
CSV_RESULTS_FILE = "loader/data_analysis/analysis/dataset_analysis_results.csv"

SUPPORTED_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
    '.tif', '.tiff', '.webp', '.nii', '.nii.gz'  # added .nii and .nii.gz
)

# Ensure the log directory exists
log_file = os.path.join(os.path.dirname(__file__),
                        "analysis/analysis_errors.log")
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# --------------------------------------------------------------------
# Helper Functions for Dataset Analysis
# --------------------------------------------------------------------

# Function to compute mean and std of dataset images


def compute_stats(data_dir, batch_size=32, max_images=500):
    """
    Compute stats using only image directory.
    Handles both class-folder structure and flat directory with images.
    Uses a random sample of images for speed if dataset is large.
    """
    import random
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # Check if there are any subfolders (class folders)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if subdirs:
        # Use ImageFolder if class folders exist
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        total = len(dataset)
        indices = list(range(total))
        if total > max_images:
            indices = random.sample(indices, max_images)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        # Flat directory: create a dataset from all supported images
        image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                       if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
        if len(image_paths) > max_images:
            image_paths = random.sample(image_paths, max_images)
        class FlatImageDataset(torch.utils.data.Dataset):
            def __init__(self, image_paths, transform=None):
                self.image_paths = image_paths
                self.transform = transform
            def __len__(self):
                return len(self.image_paths)
            def __getitem__(self, idx):
                img = Image.open(self.image_paths[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0  # dummy label
        subset = FlatImageDataset(image_paths, transform=transform)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False)
    mean = 0.0
    std = 0.0
    num_images = 0

    for images, _ in tqdm(loader, desc="Computing Stats"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_images += batch_samples

    mean /= num_images
    std /= num_images

    return mean.numpy(), std.numpy()

# Incremental pixel range analysis
def analyze_pixel_range(loader):
    running_max = -float("inf")
    running_min = float("inf")
    for batch in tqdm(loader, desc="Pixel Range Analysis"):
        # Handle both regular dataloader and object detection dataloader
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            if len(batch) == 4:  # Object detection format: images, labels, bboxes, paths
                images = batch[0]
            else:  # Regular format: images, labels
                images = batch[0]
        else:
            images = batch
            
        batch_max = images.max().item()
        batch_min = images.min().item()
        running_max = max(running_max, batch_max)
        running_min = min(running_min, batch_min)
    msg = f"Pixel range: [{running_min}, {running_max}]. "
    if running_max > 1.0:
        msg += "Pixel values are in range [0,255]. Normalization is recommended."
    else:
        msg += "Pixel values are in range [0,1]. Normalization may not be needed."
    print(msg)
    return msg

# Function to analyze class distribution


def analyze_class_distribution(dataset, dataset_name):
    """Analyze class distribution for object detection"""
    class_counts = defaultdict(int)
    for _, labels, _, _ in tqdm(dataset, desc="Analyzing class distribution"):
        for label in labels:
            class_counts[label] += 1
    if not class_counts:
        print("No classes found in annotations")
        return "No classes found in annotations"
    print("\nClass Distribution Analysis:")
    print(f"Class counts: {dict(class_counts)}")
    class_names = [str(dataset.classes[idx]) for idx in class_counts.keys()]
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts.values(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'{dataset_name} - Class Distribution')
    plt.xticks(rotation=45, ha='right')
    class_dist_fig = os.path.join(
        FIGURES_DIR, f"{dataset_name}_class_distribution.png")
    plt.tight_layout()
    plt.savefig(class_dist_fig)
    print(f"Saved class distribution figure to {class_dist_fig}")
    plt.close()  # Use plt.close() instead of plt.show() to avoid freezing in non-interactive environments
    rec = ""
    if len(class_counts) > 1:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 5:
            rec = (f"Warning: Severe class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                   "Consider oversampling, undersampling, or weighted loss functions.")
        else:
            rec = "Class distribution is relatively balanced. No special handling required."
    else:
        rec = "Only one class found. Ensure this is intentional."
    print(rec)
    return rec

# Function to visualize sample images and save the figure


def visualize_samples(dataset, dataset_name, num_samples=9):
    """Visualize samples with bounding boxes"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for i, idx in enumerate(indices):
        img, labels, bboxes, img_path = dataset[idx]
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img
        img = img.copy()
        for label, bbox in zip(labels, bboxes):
            class_name = str(dataset.classes[label])  # Ensure class_name is a string
            if len(bbox) == 4 and all(0 <= x <= 1 for x in bbox):  # YOLO format
                h, w, _ = img.shape
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
            else:  # Pascal VOC format
                x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        axes[i].imshow(img)
        axes[i].set_title(f"Image: {os.path.basename(img_path)}", fontsize=10)
        axes[i].axis('off')
    for i in range(len(indices), 9):
        axes[i].axis('off')
    plt.suptitle(f'{dataset_name} - Sample Images with Annotations',
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sample_fig_path = os.path.join(
        FIGURES_DIR, f"{dataset_name}_sample_images.png")
    plt.savefig(sample_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved sample images figure to {sample_fig_path}")
    plt.close()

# Function to check image dimensions from the raw image files


def check_image_dimensions(dataset):
    """Check image dimensions for object detection dataset"""
    original_sizes = []
    for _, _, _, img_path in dataset:
        with Image.open(img_path) as img:
            original_sizes.append(img.size)
    unique_sizes = list(set(original_sizes))
    msg = "\nImage Dimension Analysis:\n"
    if len(unique_sizes) > 10:
        msg += f"Found {len(unique_sizes)} unique sizes. First 10: {unique_sizes[:10]}\n"
    else:
        msg += f"Unique image sizes: {unique_sizes}\n"
    if len(unique_sizes) > 1:
        msg += "Warning: Images have varying dimensions. Ensure resizing or padding is applied.\n"
    else:
        msg += "All images have consistent dimensions. No resizing needed.\n"
    print(msg)
    return msg

# Function to check color channels from the raw image files


def check_color_channels(dataset):
    channels = []
    for img_path, _ in dataset.imgs:
        with Image.open(img_path) as img:
            channels.append(len(img.getbands()))
    unique_channels = set(channels)
    msg = "\nColor Channel Analysis:\n"
    msg += f"Unique channel counts: {unique_channels}\n"
    if 1 in unique_channels:
        msg += "Warning: Grayscale images detected. Consider converting to 3-channel or using 1-channel models.\n"
    else:
        msg += "All images are RGB (3-channel). No conversion needed.\n"
    print(msg)
    return msg

# Function to check for corrupted files in the dataset


def check_corrupted_files(dataset):
    corrupted_files = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted_files.append(img_path)
    msg = "\nCorrupted File Check:\n"
    if corrupted_files:
        msg += f"Warning: {len(corrupted_files)} corrupted files found.\n"
        msg += "Suggestions: Remove or replace corrupted files before training.\n"
    else:
        msg += "No corrupted files found.\n"
    print(msg)
    return msg


# Duplicate Images Check


def check_duplicate_images(dataset, limit=2, show_full=False):
    hash_dict = defaultdict(list)
    for img_path, _ in dataset.imgs:
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hash_dict[file_hash].append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    msg = ""
    if duplicates:
        msg += f"Found {len(duplicates)} sets of duplicate images. Consider removing duplicates.\n"
        count = 0
        for h, paths in duplicates.items():
            if count < limit:
                msg += f"Hash: {h} -> {len(paths)} duplicates\n"
                count += 1
            else:
                msg += "Only showing 2 duplicate sets. Use full log for more details if needed.\n"
                break
    else:
        msg = "No duplicate images found.\n"
    print(msg)
    return msg

# Noise and Artifact Detection


def detect_noise_and_artifacts(dataset):
    laplacian_vars = []
    for img_path, _ in dataset.imgs:
        try:
            img = cv2.imdecode(np.fromfile(
                img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                laplacian_vars.append(lap_var)
        except Exception as e:
            print(f"Error processing {img_path} for noise: {e}")

    if laplacian_vars:
        avg_lap = np.mean(laplacian_vars)
        msg = f"Average Laplacian variance (noise/sharpness measure): {avg_lap:.2f}\n"
        if avg_lap < 50:
            msg += "Warning: Images may be too blurry/noisy. Consider enhancing image quality.\n"
        else:
            msg += "Image sharpness/noise levels appear acceptable.\n"
    else:
        msg = "Could not compute noise/artifact metrics.\n"
    print(msg)
    return msg

# Contrast and Brightness Analysis


def analyze_contrast_and_brightness(dataset):
    brightness_vals = []
    contrast_vals = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness_vals.append(np.mean(arr))
                contrast_vals.append(np.std(arr))
        except Exception as e:
            print(f"Error processing {img_path} for brightness/contrast: {e}")

    if brightness_vals and contrast_vals:
        avg_brightness = np.mean(brightness_vals)
        avg_contrast = np.mean(contrast_vals)
        msg = f"Average brightness: {avg_brightness:.2f}\nAverage contrast: {avg_contrast:.2f}\n"
        if avg_brightness < 50:
            msg += "Warning: Overall brightness is low. Consider image enhancement.\n"
        elif avg_brightness > 200:
            msg += "Warning: Overall brightness is very high. Check for overexposure.\n"
        if avg_contrast < 30:
            msg += "Warning: Low contrast detected. Consider contrast enhancement.\n"
    else:
        msg = "Could not analyze brightness/contrast.\n"
    print(msg)
    return msg

# File Format and Metadata Consistency


def check_file_format_and_metadata(dataset):
    formats = []
    exif_errors = []
    for img_path, _ in dataset.imgs:
        ext = os.path.splitext(img_path)[1].lower()
        formats.append(ext)
        try:
            with Image.open(img_path) as img:
                exif = img._getexif()
        except Exception as e:
            exif_errors.append(img_path)
    format_counts = Counter(formats)
    msg = "File format distribution:\n"
    for fmt, count in format_counts.items():
        msg += f"  {fmt}: {count}\n"
    if exif_errors:
        msg += f"Warning: Could not extract metadata from {len(exif_errors)} images.\n"
    print(msg)
    return msg

# Resolution and Aspect Ratio


def analyze_resolution_and_aspect_ratio(dataset):
    aspect_ratios = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if h != 0:
                    aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Error processing {img_path} for aspect ratio: {e}")
    if aspect_ratios:
        mean_ratio = np.mean(aspect_ratios)
        min_ratio = np.min(aspect_ratios)
        max_ratio = np.max(aspect_ratios)
        msg = f"Aspect Ratio - Mean: {mean_ratio:.2f}, Min: {min_ratio:.2f}, Max: {max_ratio:.2f}\n"
        if max_ratio - min_ratio > 0.5:
            msg += "Warning: Significant variation in aspect ratios. Consider standardizing.\n"
    else:
        msg = "Could not compute aspect ratios.\n"
    print(msg)
    return msg

# Data Augmentation Feasibility


def evaluate_data_augmentation_feasibility(dataset):
    brightness_vals = []
    contrast_vals = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness_vals.append(np.mean(arr))
                contrast_vals.append(np.std(arr))
        except Exception as e:
            print(
                f"Error processing {img_path} for augmentation feasibility: {e}")
    if brightness_vals and contrast_vals:
        brightness_range = np.max(brightness_vals) - np.min(brightness_vals)
        contrast_range = np.max(contrast_vals) - np.min(contrast_vals)
        msg = f"Brightness range: {brightness_range:.2f}\nContrast range: {contrast_range:.2f}\n"
        if brightness_range < 30 or contrast_range < 10:
            msg += "Warning: Limited variation. Data augmentation recommended.\n"
    else:
        msg = "Could not evaluate data augmentation feasibility.\n"
    print(msg)
    return msg

# New Check 1: Image Orientation Consistency


def check_image_orientation(dataset):
    orientations = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w > h:
                    orientations.append("landscape")
                elif h > w:
                    orientations.append("portrait")
                else:
                    orientations.append("square")
        except Exception as e:
            print(f"Error processing {img_path} for orientation: {e}")
    orientation_counts = Counter(orientations)
    msg = "\nImage Orientation Analysis:\n"
    for orient, count in orientation_counts.items():
        msg += f"  {orient}: {count}\n"
    if len(orientation_counts) > 1:
        msg += "Warning: Varying orientations detected. Consider standardizing.\n"
    else:
        msg += "All images have consistent orientation.\n"
    print(msg)
    return msg

# New Check 2: Color Space Verification


def check_color_space(dataset):
    color_spaces = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                color_spaces.append(img.mode)
        except Exception as e:
            print(f"Error processing {img_path} for color space: {e}")
    unique_color_spaces = set(color_spaces)
    msg = "\nColor Space Analysis:\n"
    msg += f"Unique color spaces: {unique_color_spaces}\n"
    if len(unique_color_spaces) > 1:
        msg += "Warning: Inconsistent color spaces detected. Ensure uniformity.\n"
    else:
        msg += "All images are in the same color space.\n"
    print(msg)
    return msg

# New Check 3: Basic Label Verification


def check_label_accuracy(dataset):
    defined_classes = set(range(len(dataset.classes)))
    labels = [label for _, label in dataset]
    unique_labels = set(labels)
    invalid_labels = unique_labels - defined_classes
    msg = "\nLabel Accuracy Check:\n"
    if invalid_labels:
        msg += f"Warning: Found labels outside defined classes: {invalid_labels}\n"
    else:
        msg += "All labels correspond to defined classes.\n"
    print(msg)
    return msg

# New Check 4: Image Quality Flags


def flag_low_quality_images(dataset, brightness_threshold=50, contrast_threshold=30, noise_threshold=50):
    low_quality_flags = []
    for img_path, _ in dataset.imgs:
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness = np.mean(arr)
                contrast = np.std(arr)
                img_cv = cv2.imdecode(np.fromfile(
                    img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                lap_var = cv2.Laplacian(
                    img_cv, cv2.CV_64F).var() if img_cv is not None else 0
                if brightness < brightness_threshold or contrast < contrast_threshold or lap_var < noise_threshold:
                    low_quality_flags.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path} for quality flags: {e}")
    msg = "\nImage Quality Flags:\n"
    if low_quality_flags:
        msg += f"Found {len(low_quality_flags)} images with potential quality issues.\n"
    else:
        msg += "No images flagged for quality issues.\n"
    print(msg)
    return msg

# --------------------------------------------------------------------
# Helper Functions for Object Detection Datasets
# --------------------------------------------------------------------

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
        # YOLO format: class_id center_x center_y width height
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) >= 5:
                    classes.append(int(data[0]))
                    bboxes.append([float(x) for x in data[1:5]])
    elif annotation_path.endswith('.xml'):
        # Pascal VOC format
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls = obj.find('name').text
            classes.append(cls)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
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

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    self.image_paths.append(os.path.join(root, file))
        self.classes = get_all_classes(annotation_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.imgs = [(path, 0) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        annotation_path = find_matching_annotation(img_path, self.annotation_dir)
        if annotation_path:
            classes, bboxes = parse_annotation(annotation_path)
            labels = [self.class_to_idx[cls] if cls in self.class_to_idx else 0 for cls in classes]
        else:
            labels = []
            bboxes = []
        if self.transform:
            img = self.transform(img)
        return img, labels, bboxes, img_path

def visualize_samples(dataset, dataset_name, num_samples=9):
    """Visualize samples with bounding boxes"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for i, idx in enumerate(indices):
        img, labels, bboxes, img_path = dataset[idx]
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img
        img = img.copy()
        for label, bbox in zip(labels, bboxes):
            class_name = str(dataset.classes[label])  # Ensure class_name is a string
            if len(bbox) == 4 and all(0 <= x <= 1 for x in bbox):  # YOLO format
                h, w, _ = img.shape
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
            else:  # Pascal VOC format
                x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        axes[i].imshow(img)
        axes[i].set_title(f"Image: {os.path.basename(img_path)}", fontsize=10)
        axes[i].axis('off')
    for i in range(len(indices), 9):
        axes[i].axis('off')
    plt.suptitle(f'{dataset_name} - Sample Images with Annotations',
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sample_fig_path = os.path.join(
        FIGURES_DIR, f"{dataset_name}_sample_images.png")
    plt.savefig(sample_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved sample images figure to {sample_fig_path}")
    plt.close()

def check_image_dimensions(dataset):
    """Check image dimensions for object detection dataset"""
    original_sizes = []
    print("Checking image dimensions...")
    total_images = len(dataset)
    for i, (_, _, _, img_path) in enumerate(tqdm(dataset, desc="Checking dimensions")):
        try:
            with Image.open(img_path) as img:
                original_sizes.append(img.size)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue
        
        # Limit to first 1000 images for large datasets
        if i >= 1000:
            print(f"Sampled first 1000 images out of {total_images}")
            break
    
    unique_sizes = list(set(original_sizes))
    msg = "\nImage Dimension Analysis:\n"
    if len(unique_sizes) > 10:
        msg += f"Found {len(unique_sizes)} unique sizes. First 10: {unique_sizes[:10]}\n"
    else:
        msg += f"Unique image sizes: {unique_sizes}\n"
    if len(unique_sizes) > 1:
        msg += "Warning: Images have varying dimensions. Ensure resizing or padding is applied.\n"
    else:
        msg += "All images have consistent dimensions. No resizing needed.\n"
    print(msg)
    return msg

def check_color_channels(dataset):
    channels = []
    print("Checking color channels...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking channels"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking channels"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                channels.append(len(img.getbands()))
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    
    unique_channels = set(channels)
    msg = "\nColor Channel Analysis:\n"
    msg += f"Unique channel counts: {unique_channels}\n"
    if 1 in unique_channels:
        msg += "Warning: Grayscale images detected. Consider converting to 3-channel or using 1-channel models.\n"
    else:
        msg += "All images are RGB (3-channel). No conversion needed.\n"
    print(msg)
    return msg

def check_corrupted_files(dataset):
    corrupted_files = []
    print("Checking for corrupted files...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking corruption"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking corruption"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted_files.append(img_path)
        
        # Limit to first 1000 images for large datasets
        if i >= 1000:
            print(f"Sampled first 1000 images out of {len(items)}")
            break
    
    msg = "\nCorrupted File Check:\n"
    if corrupted_files:
        msg += f"Warning: {len(corrupted_files)} corrupted files found.\n"
        msg += "Suggestions: Remove or replace corrupted files before training.\n"
    else:
        msg += "No corrupted files found.\n"
    print(msg)
    return msg

def check_duplicate_images(dataset, limit=2, show_full=False):
    hash_dict = defaultdict(list)
    print("Checking for duplicate images...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking duplicates"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking duplicates"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            hash_dict[file_hash].append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
        
        # Limit to first 1000 images for large datasets
        if i >= 1000:
            print(f"Sampled first 1000 images out of {len(items)}")
            break
    
    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    msg = ""
    if duplicates:
        msg += f"Found {len(duplicates)} sets of duplicate images. Consider removing duplicates.\n"
        count = 0
        for h, paths in duplicates.items():
            if count < limit:
                msg += f"Hash: {h} -> {len(paths)} duplicates\n"
                count += 1
            else:
                msg += "Only showing 2 duplicate sets. Use full log for more details if needed.\n"
                break
    else:
        msg = "No duplicate images found.\n"
    print(msg)
    return msg

def detect_noise_and_artifacts(dataset):
    laplacian_vars = []
    print("Detecting noise and artifacts...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking noise/artifacts"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking noise/artifacts"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            img = cv2.imdecode(np.fromfile(
                img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                laplacian_vars.append(lap_var)
        except Exception as e:
            print(f"Error processing {img_path} for noise: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    if laplacian_vars:
        avg_lap = np.mean(laplacian_vars)
        msg = f"Average Laplacian variance (noise/sharpness measure): {avg_lap:.2f}\n"
        if avg_lap < 50:
            msg += "Warning: Images may be too blurry/noisy. Consider enhancing image quality.\n"
        else:
            msg += "Image sharpness/noise levels appear acceptable.\n"
    else:
        msg = "Could not compute noise/artifact metrics.\n"
    print(msg)
    return msg

def analyze_contrast_and_brightness(dataset):
    brightness_vals = []
    contrast_vals = []
    print("Analyzing contrast and brightness...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking brightness/contrast"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking brightness/contrast"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness_vals.append(np.mean(arr))
                contrast_vals.append(np.std(arr))
        except Exception as e:
            print(f"Error processing {img_path} for brightness/contrast: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    if brightness_vals and contrast_vals:
        avg_brightness = np.mean(brightness_vals)
        avg_contrast = np.mean(contrast_vals)
        msg = f"Average brightness: {avg_brightness:.2f}\nAverage contrast: {avg_contrast:.2f}\n"
        if avg_brightness < 50:
            msg += "Warning: Overall brightness is low. Consider image enhancement.\n"
        elif avg_brightness > 200:
            msg += "Warning: Overall brightness is very high. Check for overexposure.\n"
        if avg_contrast < 30:
            msg += "Warning: Low contrast detected. Consider contrast enhancement.\n"
    else:
        msg = "Could not analyze brightness/contrast.\n"
    print(msg)
    return msg

def check_file_format_and_metadata(dataset):
    formats = []
    exif_errors = []
    print("Checking file formats and metadata...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking file formats"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking file formats"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        ext = os.path.splitext(img_path)[1].lower()
        formats.append(ext)
        try:
            with Image.open(img_path) as img:
                exif = img._getexif()
        except Exception as e:
            exif_errors.append(img_path)
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    format_counts = Counter(formats)
    msg = "File format distribution:\n"
    for fmt, count in format_counts.items():
        msg += f"  {fmt}: {count}\n"
    if exif_errors:
        msg += f"Warning: Could not extract metadata from {len(exif_errors)} images.\n"
    print(msg)
    return msg

def analyze_resolution_and_aspect_ratio(dataset):
    aspect_ratios = []
    print("Analyzing resolution and aspect ratios...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking aspect ratios"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking aspect ratios"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if h != 0:
                    aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Error processing {img_path} for aspect ratio: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    if aspect_ratios:
        mean_ratio = np.mean(aspect_ratios)
        min_ratio = np.min(aspect_ratios)
        max_ratio = np.max(aspect_ratios)
        msg = f"Aspect Ratio - Mean: {mean_ratio:.2f}, Min: {min_ratio:.2f}, Max: {max_ratio:.2f}\n"
        if max_ratio - min_ratio > 0.5:
            msg += "Warning: Significant variation in aspect ratios. Consider standardizing.\n"
    else:
        msg = "Could not compute aspect ratios.\n"
    print(msg)
    return msg

def evaluate_data_augmentation_feasibility(dataset):
    brightness_vals = []
    contrast_vals = []
    print("Evaluating data augmentation feasibility...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking augmentation feasibility"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking augmentation feasibility"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness_vals.append(np.mean(arr))
                contrast_vals.append(np.std(arr))
        except Exception as e:
            print(
                f"Error processing {img_path} for augmentation feasibility: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    if brightness_vals and contrast_vals:
        brightness_range = np.max(brightness_vals) - np.min(brightness_vals)
        contrast_range = np.max(contrast_vals) - np.min(contrast_vals)
        msg = f"Brightness range: {brightness_range:.2f}\nContrast range: {contrast_range:.2f}\n"
        if brightness_range < 30 or contrast_range < 10:
            msg += "Warning: Limited variation. Data augmentation recommended.\n"
    else:
        msg = "Could not evaluate data augmentation feasibility.\n"
    print(msg)
    return msg

def check_image_orientation(dataset):
    orientations = []
    print("Checking image orientations...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking orientations"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking orientations"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w > h:
                    orientations.append("landscape")
                elif h > w:
                    orientations.append("portrait")
                else:
                    orientations.append("square")
        except Exception as e:
            print(f"Error processing {img_path} for orientation: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    orientation_counts = Counter(orientations)
    msg = "\nImage Orientation Analysis:\n"
    for orient, count in orientation_counts.items():
        msg += f"  {orient}: {count}\n"
    if len(orientation_counts) > 1:
        msg += "Warning: Varying orientations detected. Consider standardizing.\n"
    else:
        msg += "All images have consistent orientation.\n"
    print(msg)
    return msg

def check_color_space(dataset):
    color_spaces = []
    print("Checking color spaces...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking color spaces"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking color spaces"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                color_spaces.append(img.mode)
        except Exception as e:
            print(f"Error processing {img_path} for color space: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    unique_color_spaces = set(color_spaces)
    msg = "\nColor Space Analysis:\n"
    msg += f"Unique color spaces: {unique_color_spaces}\n"
    if len(unique_color_spaces) > 1:
        msg += "Warning: Inconsistent color spaces detected. Ensure uniformity.\n"
    else:
        msg += "All images are in the same color space.\n"
    print(msg)
    return msg

def check_label_accuracy(dataset):
    defined_classes = set(range(len(dataset.classes)))
    labels = []
    print("Checking label accuracy...")
    for _, label_list, _, _ in tqdm(dataset, desc="Checking labels"):
        labels.extend(label_list)
    unique_labels = set(labels)
    invalid_labels = unique_labels - defined_classes
    msg = "\nLabel Accuracy Check:\n"
    if invalid_labels:
        msg += f"Warning: Found labels outside defined classes: {invalid_labels}\n"
    else:
        msg += "All labels correspond to defined classes.\n"
    print(msg)
    return msg

def flag_low_quality_images(dataset, brightness_threshold=50, contrast_threshold=30, noise_threshold=50):
    low_quality_flags = []
    print("Flagging low quality images...")
    
    # Handle different dataset types
    if hasattr(dataset, 'imgs'):
        # Traditional ImageFolder dataset
        items = dataset.imgs
        progress_desc = "Checking image quality"
    else:
        # Object detection dataset
        items = [(path, 0) for path in dataset.image_paths]
        progress_desc = "Checking image quality"
    
    for i, (img_path, _) in enumerate(tqdm(items, desc=progress_desc)):
        try:
            with Image.open(img_path) as img:
                gray = img.convert("L")
                arr = np.array(gray)
                brightness = np.mean(arr)
                contrast = np.std(arr)
                img_cv = cv2.imdecode(np.fromfile(
                    img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                lap_var = cv2.Laplacian(
                    img_cv, cv2.CV_64F).var() if img_cv is not None else 0
                if brightness < brightness_threshold or contrast < contrast_threshold or lap_var < noise_threshold:
                    low_quality_flags.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path} for quality flags: {e}")
        
        # Limit to first 500 images for large datasets
        if i >= 500:
            print(f"Sampled first 500 images out of {len(items)}")
            break
    msg = "\nImage Quality Flags:\n"
    if low_quality_flags:
        msg += f"Found {len(low_quality_flags)} images with potential quality issues.\n"
    else:
        msg += "No images flagged for quality issues.\n"
    print(msg)
    return msg



def collate_fn_object_detection(batch):
    """Custom collate function for object detection datasets"""
    images = []
    labels = []
    bboxes = []
    paths = []
    
    for img, label_list, bbox_list, path in batch:
        images.append(img)
        labels.append(label_list)
        bboxes.append(bbox_list)
        paths.append(path)
    
    # Stack images only
    images = torch.stack(images, 0)
    return images, labels, bboxes, paths

def analyze_dataset(dataset_name, data_dir):
    """
    Comprehensive dataset analysis function that handles both classification 
    and object detection datasets
    """
    print(f"\nAnalyzing dataset: {dataset_name} at {data_dir}")
    
    # Check if this is an object detection dataset
    is_object_detection = dataset_name in object_detection_datasets
    
    if is_object_detection:
        # Handle object detection dataset
        dataset_config = object_detection_datasets[dataset_name]
        image_dir = os.path.join(data_dir, dataset_config['images'])
        annotation_dir = os.path.join(data_dir, dataset_config['annotations'])
        
        if not os.path.exists(image_dir):
            print(f"Error: Image directory not found: {image_dir}")
            return
        if not os.path.exists(annotation_dir):
            print(f"Error: Annotation directory not found: {annotation_dir}")
            return
            
        print(f"Found annotation directory: {annotation_dir}")
        
        # Count valid image files
        dataset_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                    dataset_files.append(os.path.join(root, file))
        
        print(f"Found {len(dataset_files)} valid image files in {dataset_name}")
        
    else:
        # Handle traditional classification dataset
        subdirs = [d for d in os.listdir(
            data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_files = defaultdict(list)
        for sub in subdirs:
            sub_path = os.path.join(data_dir, sub)
            valid_files = []
            for root, _, files in os.walk(sub_path):
                for file in files:
                    if file.lower().endswith(SUPPORTED_EXTENSIONS):
                        valid_files.append(os.path.join(root, file))
            if valid_files:
                class_files[sub].extend(valid_files)

        if class_files:
            print(f"Found {len(class_files)} classes in {dataset_name}:")
            dataset_files = sum(class_files.values(), [])
        else:
            dataset_files = []
            if subdirs:
                for sub in subdirs:
                    sub_path = os.path.join(data_dir, sub)
                    for root, _, files in os.walk(sub_path):
                        for file in files:
                            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                                dataset_files.append(os.path.join(root, file))
                print(
                    f"Found {len(dataset_files)} valid files in sample folders of {dataset_name}.")
            else:
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.lower().endswith(SUPPORTED_EXTENSIONS):
                            dataset_files.append(os.path.join(root, file))
                print(
                    f"Found {len(dataset_files)} valid files directly in {dataset_name}.")

    if not dataset_files:
        print(
            f"Error analyzing {dataset_name}: Found no valid files. Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        if is_object_detection:
            # Create object detection dataset
            dataset = ObjectDetectionDataset(image_dir, annotation_dir, transform=transform)
        else:
            # Try to find annotation directory for object detection datasets
            annotation_dir = None
            for candidate in ["annotations", "annotation", "labels", "label"]:
                candidate_path = os.path.join(data_dir, candidate)
                if os.path.isdir(candidate_path):
                    annotation_dir = candidate_path
                    break
            
            if annotation_dir:
                print(f"Found annotation directory: {annotation_dir}")
                dataset = ObjectDetectionDataset(data_dir, annotation_dir, transform=transform)
            else:
                dataset = datasets.ImageFolder(root=data_dir, transform=transform)
            
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False,
            collate_fn=collate_fn_object_detection if (is_object_detection or hasattr(dataset, 'image_paths')) else None)

        print("Computing dataset statistics...")
        if is_object_detection:
            mean, std = compute_stats(image_dir)
        else:
            mean, std = compute_stats(data_dir)
        print(f"Mean: {mean}, Std: {std}")
        
        print("Analyzing class distribution...")
        class_dist_rec = analyze_class_distribution(dataset, dataset_name)
        
        print("Generating sample visualizations...")
        if hasattr(dataset, 'classes'):
            visualize_samples(dataset, dataset_name)
        plt.close('all')
        
        print("Running comprehensive dataset checks...")
        dims_msg = check_image_dimensions(dataset)
        color_msg = check_color_channels(dataset)
        corrupted_msg = check_corrupted_files(dataset)
        pixel_range_msg = analyze_pixel_range(loader)
        dup_msg = check_duplicate_images(dataset)
        noise_msg = detect_noise_and_artifacts(dataset)
        brightness_contrast_msg = analyze_contrast_and_brightness(dataset)
        file_meta_msg = check_file_format_and_metadata(dataset)
        aspect_ratio_msg = analyze_resolution_and_aspect_ratio(dataset)
        aug_msg = evaluate_data_augmentation_feasibility(dataset)
        orientation_msg = check_image_orientation(dataset)
        color_space_msg = check_color_space(dataset)
        quality_flags_msg = flag_low_quality_images(dataset)
        
        # For object detection datasets, also check label accuracy
        if hasattr(dataset, 'classes') and len(dataset.classes) > 0:
            label_accuracy_msg = check_label_accuracy(dataset)
        else:
            label_accuracy_msg = "Label accuracy check not applicable for this dataset type."

        results = {
            "Mean": mean,
            "Std": std,
            "Class Distribution": class_dist_rec,
            "Unique Image Sizes": dims_msg,
            "Unique Color Channels": color_msg,
            "Corrupted Files": corrupted_msg,
            "Pixel Value Range": pixel_range_msg,
            "Duplicate Check": dup_msg,
            "Noise/Artifacts": noise_msg,
            "Brightness/Contrast": brightness_contrast_msg,
            "File Format/Metadata": file_meta_msg,
            "Aspect Ratio": aspect_ratio_msg,
            "Data Augmentation Feasibility": aug_msg,
            "Image Orientation": orientation_msg,
            "Color Space": color_space_msg,
            "Label Accuracy": label_accuracy_msg,
            "Image Quality Flags": quality_flags_msg
        }

        print("Saving analysis results...")
        save_results(dataset_name, results)
        save_results_csv(dataset_name, results)

        del dataset, loader, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("---- Analysis Complete ----")

    except Exception as e:
        print(f"Error analyzing {dataset_name}: {str(e)}")
        with open(log_file, "a") as f:
            f.write(f"{dataset_name} failed: {str(e)}\n")

# Function to save results to a text file
def save_results(dataset_name, results, file_path=TXT_RESULTS_FILE):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as f:
        f.write(f"\nTimestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 50 + "\n")
    print(f"Results saved to {file_path}")

# Function to save results to a CSV file
def save_results_csv(dataset_name, results, file_path=CSV_RESULTS_FILE):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = ["Dataset", "Timestamp", "Mean", "Std", "Class Distribution",
              "Unique Image Sizes", "Unique Color Channels", "Corrupted Files", "Pixel Value Range",
              "Duplicate Check", "Noise/Artifacts", "Brightness/Contrast", "File Format/Metadata",
              "Aspect Ratio", "Data Augmentation Feasibility", "Image Orientation", "Color Space",
              "Label Accuracy", "Image Quality Flags"]

    row = {
        "Dataset": dataset_name,
        "Timestamp": timestamp,
        "Mean": results.get("Mean"),
        "Std": results.get("Std"),
        "Class Distribution": results.get("Class Distribution"),
        "Unique Image Sizes": results.get("Unique Image Sizes"),
        "Unique Color Channels": results.get("Unique Color Channels"),
        "Corrupted Files": results.get("Corrupted Files"),
        "Pixel Value Range": results.get("Pixel Value Range"),
        "Duplicate Check": results.get("Duplicate Check"),
        "Noise/Artifacts": results.get("Noise/Artifacts"),
        "Brightness/Contrast": results.get("Brightness/Contrast"),
        "File Format/Metadata": results.get("File Format/Metadata"),
        "Aspect Ratio": results.get("Aspect Ratio"),
        "Data Augmentation Feasibility": results.get("Data Augmentation Feasibility"),
        "Image Orientation": results.get("Image Orientation"),
        "Color Space": results.get("Color Space"),
        "Label Accuracy": results.get("Label Accuracy"),
        "Image Quality Flags": results.get("Image Quality Flags")
    }

    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to CSV file {file_path}")

# --------------------------------------------------------------------
# Dataset Directories and Processing
# --------------------------------------------------------------------

# Traditional classification datasets with class folders
classification_dataset_dirs = {
    # 'roct': 'dataset/rotc/train',
    # 'chest_xray': 'dataset/chest_xray/train',
    # 'ccts': 'dataset/ccts/train',
    # 'dermnet': 'dataset/dermnet/train',
    # 'kvasir': 'dataset/kvasir/train',
    # 'scisic': 'dataset/scisic/Train',
    # 'tbcr': 'dataset/tbcr',
    # 'seabed': 'dataset/seabed',
    # 'cattled': 'dataset/cattled/Training',
}

# Object detection datasets with separate image and annotation folders
object_detection_datasets = {
    'cattleface': {
        'root': 'dataset/cattleface',
        'images': 'CowfaceImage',  # subfolder with images
        'annotations': 'Annotation'  # subfolder with annotations
    },
    # Add more object detection datasets here following the same pattern
    # 'dataset_name': {
    #     'root': 'dataset/dataset_name',
    #     'images': 'images_folder_name',
    #     'annotations': 'annotations_folder_name'
    # },
}

# Combined dataset dirs for backward compatibility
dataset_dirs = {**classification_dataset_dirs}

if __name__ == "__main__":
    print("---- Dataset Analysis Results ----")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Process traditional classification datasets
    for dataset_name, relative_path in dataset_dirs.items():
        data_dir = os.path.join(project_root, relative_path)
        if os.path.exists(data_dir):
            analyze_dataset(dataset_name, data_dir)
        else:
            print(f"Skipping {dataset_name} - directory not found: {data_dir}")
    
    # Process object detection datasets
    for dataset_name, config in object_detection_datasets.items():
        data_dir = os.path.join(project_root, config['root'])
        if os.path.exists(data_dir):
            analyze_dataset(dataset_name, data_dir)
        else:
            print(f"Skipping {dataset_name} - directory not found: {data_dir}")

