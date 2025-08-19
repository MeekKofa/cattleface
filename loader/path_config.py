"""
Path configuration module for robust cross-platform dataset path resolution
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging


class DatasetPathConfig:
    """Handles dataset path configuration and resolution"""

    def __init__(self, config_file: str = "loader/config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.base_dataset_dir = Path(self.config['data']['data_dir'])
        self.processed_data_dir = Path(self.config['data']['processed_dir'])

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        config_path = Path(self.config_file)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_dataset_config(self, dataset_name: str) -> Optional[Dict]:
        """Get configuration for a specific dataset"""
        for dataset in self.config['data']['data_key']:
            if dataset['name'].lower() == dataset_name.lower():
                return dataset
        return None

    def resolve_dataset_path(self, dataset_name: str) -> Path:
        """Resolve the full path to a dataset"""
        dataset_config = self.get_dataset_config(dataset_name)

        # Check multiple possible locations, prioritizing processed_data
        possible_paths = [
            # PRIORITY 1: Processed data path (where your data actually is)
            self.processed_data_dir / dataset_name,
            # PRIORITY 2: Current working directory + processed_data
            Path.cwd() / "processed_data" / dataset_name,
            # PRIORITY 3: Config-based paths
            self.base_dataset_dir / dataset_name if dataset_config else None,
            self.base_dataset_dir /
            dataset_config['train_dir'] if dataset_config else None,
            # PRIORITY 4: Direct path fallback
            Path(dataset_name) if Path(dataset_name).exists() else None
        ]

        # Filter out None values
        possible_paths = [p for p in possible_paths if p is not None]

        # Return the first existing path
        for path in possible_paths:
            if path.exists():
                logging.info(f"Found dataset '{dataset_name}' at: {path}")
                return path.resolve()

        # If none exist, return the processed_data path for creation
        default_path = self.processed_data_dir / dataset_name
        logging.warning(
            f"Dataset path not found, defaulting to: {default_path}")
        return default_path.resolve()

    def get_dataset_structure_paths(self, dataset_name: str) -> Dict[str, Path]:
        """Get structured paths for train/val/test splits"""
        base_path = self.resolve_dataset_path(dataset_name)
        dataset_config = self.get_dataset_config(dataset_name)

        if not dataset_config:
            # Default structure
            return {
                'train': base_path / 'train',
                'val': base_path / 'val',
                'test': base_path / 'test'
            }

        structure = dataset_config.get('structure', {})
        splits = structure.get('splits', {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        })

        return {
            split_name: base_path / split_path
            for split_name, split_path in splits.items()
        }

    def get_object_detection_paths(self, dataset_name: str) -> Dict[str, Dict[str, Path]]:
        """Get structured paths for object detection datasets (images + labels)"""
        structure_paths = self.get_dataset_structure_paths(dataset_name)
        dataset_config = self.get_dataset_config(dataset_name)

        # Use the correct subdirectory names based on your actual structure
        if dataset_name.lower() == 'cattleface':
            # Your processed cattleface uses 'images' and 'annotations'
            images_dir = "images"
            labels_dir = "annotations"
        elif dataset_name.lower() == 'cattlebody':
            # Standard structure for cattlebody
            images_dir = "images"
            labels_dir = "annotations"  # This matches your processed_data structure
        else:
            # Default for other datasets
            if dataset_config:
                images_dir = dataset_config.get('images_dir', 'images')
                labels_dir = dataset_config.get(
                    'annotations_dir', 'annotations')
            else:
                images_dir = "images"
                labels_dir = "annotations"

        result = {}
        for split_name, split_path in structure_paths.items():
            result[split_name] = {
                'images': split_path / images_dir,
                'labels': split_path / labels_dir
            }

        return result

    def validate_dataset_structure(self, dataset_name: str) -> Tuple[bool, str]:
        """Validate that required dataset directories exist"""
        try:
            dataset_config = self.get_dataset_config(dataset_name)
            if not dataset_config:
                return False, f"Dataset '{dataset_name}' not found in configuration"

            structure_type = dataset_config.get(
                'structure', {}).get('type', 'standard')

            if structure_type == 'object_detection':
                paths = self.get_object_detection_paths(dataset_name)

                # Check each split
                for split_name, split_paths in paths.items():
                    images_path = split_paths['images']
                    labels_path = split_paths['labels']

                    if not images_path.exists():
                        return False, f"Images directory not found: {images_path}"

                    if not labels_path.exists():
                        return False, f"Labels directory not found: {labels_path}"

                    # Check for actual files
                    image_exts = ('.jpg', '.jpeg', '.png', '.ppm',
                                  '.bmp', '.pgm', '.tif', '.tiff', '.webp')
                    has_images = any(
                        f.suffix.lower() in image_exts
                        for f in images_path.glob('*') if f.is_file()
                    )

                    if not has_images and split_name == 'train':  # At least training should have images
                        return False, f"No valid image files found in: {images_path}"

            else:
                # Standard classification structure
                structure_paths = self.get_dataset_structure_paths(
                    dataset_name)
                for split_name, split_path in structure_paths.items():
                    if split_name == 'train' and not split_path.exists():
                        return False, f"Required directory not found: {split_path}"

            return True, "Dataset structure validation passed"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_yaml_data_path(self, dataset_name: str) -> Optional[Path]:
        """Get path to YAML data configuration file for YOLO datasets"""
        dataset_config = self.get_dataset_config(dataset_name)
        if not dataset_config:
            return None

        yaml_path = dataset_config.get('training', {}).get('data')
        if yaml_path:
            # Try relative to dataset directory first
            base_path = self.resolve_dataset_path(dataset_name)
            yaml_file = base_path / yaml_path
            if yaml_file.exists():
                return yaml_file

            # Try relative to current working directory
            yaml_file = Path.cwd() / yaml_path
            if yaml_file.exists():
                return yaml_file

        return None


# Global instance for easy access
_path_config = None


def get_path_config() -> DatasetPathConfig:
    """Get the global path configuration instance"""
    global _path_config
    if _path_config is None:
        _path_config = DatasetPathConfig()
    return _path_config


def resolve_dataset_path(dataset_name: str) -> Path:
    """Convenience function to resolve dataset path"""
    return get_path_config().resolve_dataset_path(dataset_name)


def get_dataset_paths(dataset_name: str) -> Dict[str, Dict[str, Path]]:
    """Convenience function to get object detection dataset paths"""
    return get_path_config().get_object_detection_paths(dataset_name)


def validate_dataset(dataset_name: str) -> Tuple[bool, str]:
    """Convenience function to validate dataset structure"""
    return get_path_config().validate_dataset_structure(dataset_name)
