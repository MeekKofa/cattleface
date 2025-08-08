"""
Training Manager - orchestrates the training process with proper modular structure
"""
import torch
import torch.nn as nn
import logging
import random
import numpy as np

from .object_detection_trainer import ObjectDetectionTrainer
from .validation_handler import ValidationHandler


class TrainingManager:
    """Manages the training process with modular components"""

    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self._setup_random_seeds()

        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None

        logging.info(f"TrainingManager initialized with device: {self.device}")

    def _setup_device(self):
        """Setup compute device"""
        if torch.cuda.is_available() and hasattr(self.args, 'gpu_ids') and self.args.gpu_ids:
            device = torch.device(f"cuda:{self.args.gpu_ids[0]}")
            logging.info(f"Using GPU: {device}")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU")
        return device

    def _setup_random_seeds(self):
        """Setup random seeds for reproducibility"""
        seed = getattr(self.args, 'seed', 42)
        logging.info(f"Setting random seed: {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def initialize_model(self):
        """Initialize the model"""
        from model.vgg_yolov8 import get_vgg_yolov8
        from class_mapping import get_num_classes

        # Extract architecture and depth
        arch_key = self.args.arch[0] if isinstance(
            self.args.arch, list) else self.args.arch

        if isinstance(self.args.depth, dict):
            depth_val = self.args.depth.get(arch_key, [16])[0]
        elif isinstance(self.args.depth, list):
            depth_val = self.args.depth[0]
        else:
            depth_val = 16

        # Get number of classes
        num_classes = get_num_classes()

        # Create model
        self.model = get_vgg_yolov8(num_classes=num_classes, depth=depth_val)
        self.model.to(self.device)

        logging.info(
            f"Model initialized: {arch_key}_{depth_val} with {num_classes} classes")
        return self.model

    def initialize_data(self):
        """Initialize data loaders"""
        from loader.dataset_loader import DatasetLoader, object_detection_collate
        from torch.utils.data import DataLoader

        dataset_name = self.args.data
        logging.info(f"Initializing dataset: {dataset_name}")

        # Load data
        train_loader, val_loader, test_loader = DatasetLoader().load_data(
            dataset_name=dataset_name,
            batch_size={
                'train': self.args.train_batch,
                'val': self.args.train_batch,
                'test': self.args.train_batch
            },
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )

        # Create data loaders with proper collate function
        self.train_loader = DataLoader(
            train_loader.dataset,
            batch_size=self.args.train_batch,
            sampler=train_loader.sampler,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=object_detection_collate
        )

        self.val_loader = DataLoader(
            val_loader.dataset,
            batch_size=self.args.train_batch,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=object_detection_collate
        )

        self.test_loader = DataLoader(
            test_loader.dataset,
            batch_size=self.args.train_batch,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=object_detection_collate
        )

        logging.info(
            f"Data loaders initialized: {len(self.train_loader.dataset)} training samples")
        return self.train_loader, self.val_loader, self.test_loader

    def initialize_optimizer(self):
        """Initialize optimizer"""
        if self.model is None:
            raise ValueError("Model must be initialized before optimizer")

        optimizer_name = self.args.optimizer.lower()

        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=getattr(self.args, 'weight_decay', 1e-4)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=getattr(self.args, 'momentum', 0.9),
                weight_decay=getattr(self.args, 'weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        logging.info(f"Optimizer initialized: {optimizer_name}")
        return self.optimizer

    def initialize_criterion(self):
        """Initialize loss criterion"""
        # For object detection, we'll use the model's built-in loss
        if hasattr(self.model, 'compute_loss'):
            self.criterion = self.model.compute_loss
            logging.info("Using model's compute_loss for object detection")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logging.info("Using CrossEntropyLoss as fallback")

        return self.criterion

    def create_trainer(self):
        """Create appropriate trainer based on model type"""
        if self.model is None:
            raise ValueError(
                "Model must be initialized before creating trainer")

        # Extract architecture info
        arch_key = self.args.arch[0] if isinstance(
            self.args.arch, list) else self.args.arch

        if isinstance(self.args.depth, dict):
            depth_val = self.args.depth.get(arch_key, [16])[0]
        elif isinstance(self.args.depth, list):
            depth_val = self.args.depth[0]
        else:
            depth_val = 16

        # Create object detection trainer
        trainer = ObjectDetectionTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            model_name=arch_key,
            task_name=self.args.task_name,
            dataset_name=self.args.data,
            device=self.device,
            config=self.args,
            scheduler=None,
            model_depth=depth_val
        )

        logging.info(
            f"Created ObjectDetectionTrainer for {arch_key}_{depth_val}")
        return trainer

    def diagnose_model(self, num_batches=3):
        """Diagnose model outputs before training"""
        if self.model is None or self.val_loader is None:
            logging.error(
                "Model and validation loader must be initialized before diagnosis")
            return

        logging.info("Running model diagnosis...")

        validation_handler = ValidationHandler(
            self.model, self.device, self.args)
        validation_handler.diagnose_model_outputs(self.val_loader, num_batches)

    def setup_all(self):
        """Setup all components"""
        logging.info("Setting up all training components...")

        # Initialize in order
        self.initialize_model()
        self.initialize_data()
        self.initialize_optimizer()
        self.initialize_criterion()

        # Run diagnosis
        self.diagnose_model()

        # Create trainer
        trainer = self.create_trainer()

        logging.info("All components initialized successfully")
        return trainer

    def train_dataset(self, run_test=False):
        """Train the model on the dataset"""
        logging.info(f"Starting training on dataset: {self.args.data}")

        # Setup all components
        trainer = self.setup_all()

        # Start training
        patience = getattr(self.args, 'patience', 10)
        best_loss, best_acc = trainer.train(patience=patience)

        logging.info(
            f"Training completed. Best Loss: {best_loss:.4f}, Best Accuracy: {best_acc:.4f}")

        # Optionally run test
        if run_test:
            logging.info("Running test evaluation...")
            # TODO: Implement test evaluation

        # Cleanup
        trainer.cleanup()

        return best_loss, best_acc
