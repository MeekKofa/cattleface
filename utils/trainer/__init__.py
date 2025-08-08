"""
Modular trainer components for better separation of concerns
"""

from .base_trainer import BaseTrainer
from .object_detection_trainer import ObjectDetectionTrainer
from .classification_trainer import ClassificationTrainer
from .validation_handler import ValidationHandler
from .training_manager import TrainingManager

__all__ = [
    'BaseTrainer',
    'ObjectDetectionTrainer',
    'ClassificationTrainer',
    'ValidationHandler',
    'TrainingManager'
]
