"""
Modular training system with properly separated concerns
"""
from utils.trainer import ObjectDetectionTrainer, ClassificationTrainer
from utils.trainer import TrainingManager as ModularTrainingManager
import os
import logging
import warnings
import torch
import numpy as np
import random
from datetime import datetime

# Set environment
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)

# Import the modular trainer components


class TrainingManager:
    """Main training manager that uses modular components"""

    def __init__(self, args):
        self.args = args
        self.setup_logging()

        # Extract key attributes for compatibility
        self.data = args.data
        self.arch = args.arch
        self.depth = args.depth
        self.train_batch = args.train_batch
        self.epochs = args.epochs
        self.lr = args.lr
        self.drop = args.drop
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.gpu_ids = args.gpu_ids
        self.task_name = args.task_name
        self.optimizer = args.optimizer
        self.seed = getattr(args, 'seed', 42)

        # Setup device and random seeds
        self.device = self._setup_device()
        self._setup_random_seeds()

        # Create the modular training manager
        self.modular_manager = ModularTrainingManager(args)

        logging.info(f"TrainingManager initialized with device: {self.device}")

    def setup_logging(self):
        """Setup basic logging"""
        # Don't override existing logging setup
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(levelname)s - %(message)s'
            )

    def _setup_device(self):
        """Setup compute device"""
        if torch.cuda.is_available() and self.gpu_ids:
            device = torch.device(f"cuda:{self.gpu_ids[0]}")
            logging.info(f"Using GPU: {device}")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU")
        return device

    def _setup_random_seeds(self):
        """Setup random seeds for reproducibility"""
        logging.info(f"Setting random seed: {self.seed}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_dataset(self, dataset_name, run_test=False):
        """Train on the specified dataset using modular components"""
        logging.info("=" * 80)
        logging.info(f"STARTING MODULAR TRAINING: {dataset_name}")
        logging.info("=" * 80)

        try:
            # Use the modular training manager
            best_loss, best_acc = self.modular_manager.train_dataset(
                run_test=run_test)

            logging.info("=" * 80)
            logging.info(f"TRAINING COMPLETED SUCCESSFULLY")
            logging.info(f"Best Loss: {best_loss:.4f}")
            logging.info(f"Best Accuracy: {best_acc:.4f}")
            logging.info("=" * 80)

            return best_loss, best_acc

        except Exception as e:
            logging.error("=" * 80)
            logging.error(f"TRAINING FAILED: {e}")
            logging.error("=" * 80)
            import traceback
            logging.error(traceback.format_exc())
            raise


class Trainer:
    """Legacy Trainer class that delegates to modular components"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, criterion, model_name, task_name, dataset_name,
                 device, config, scheduler=None, is_object_detection=False, model_depth=None):

        logging.info("Initializing modular Trainer...")

        # Store parameters for compatibility
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_name = model_name
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.is_object_detection = is_object_detection
        self.model_depth = model_depth

        # Create modular trainer based on type
        if is_object_detection:
            self.modular_trainer = ObjectDetectionTrainer(
                model, train_loader, val_loader, test_loader,
                optimizer, criterion, model_name, task_name, dataset_name,
                device, config, scheduler, model_depth
            )
        else:
            self.modular_trainer = ClassificationTrainer(
                model, train_loader, val_loader, test_loader,
                optimizer, criterion, model_name, task_name, dataset_name,
                device, config, scheduler, model_depth
            )

        # Initialize basic attributes for compatibility
        self.has_trained = False
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = self._init_history()

        logging.info(
            f"Modular Trainer initialized for {model_name} ({'object detection' if is_object_detection else 'classification'})")

    def _init_history(self):
        """Initialize training history"""
        return {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'duration': [],
            'true_labels': [],
            'predictions': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_predictions': [],
            'val_targets': [],
            'adv_loss': [],
            'adv_accuracy': [],
            'adv_predictions': [],
            'adv_targets': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'per_class_metrics': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_per_class_metrics': []
        }

    def train(self, patience):
        """Delegate training to modular trainer"""
        logging.info("Starting modular training...")

        try:
            best_loss, best_acc = self.modular_trainer.train(patience)

            # Update legacy attributes for compatibility
            self.has_trained = True
            self.best_val_loss = best_loss
            self.best_val_acc = best_acc
            self.best_metrics = {'loss': best_loss, 'accuracy': best_acc}

            # Copy history from modular trainer
            if hasattr(self.modular_trainer, 'history'):
                self.history = self.modular_trainer.history

            return best_loss, best_acc

        except Exception as e:
            logging.error(f"Error in modular trainer: {e}")
            raise

    def validate(self):
        """Delegate validation to modular trainer"""
        return self.modular_trainer.validate()

    def validate_with_metrics(self):
        """Delegate validation with metrics to modular trainer"""
        if hasattr(self.modular_trainer, 'validate_with_metrics'):
            return self.modular_trainer.validate_with_metrics()
        else:
            # Fallback to simple validation
            val_loss, val_acc = self.modular_trainer.validate()
            metrics = {
                'loss': val_loss,
                'accuracy': val_acc,
                'precision': val_acc,
                'recall': val_acc,
                'f1': val_acc
            }
            return val_loss, metrics

    def save_model(self, filename=None):
        """Delegate model saving to modular trainer"""
        return self.modular_trainer.save_model(filename)


# For backward compatibility, keep the _init_history function
def _init_history():
    return {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'duration': [],
        'true_labels': [],
        'predictions': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_predictions': [],
        'val_targets': [],
        'adv_loss': [],
        'adv_accuracy': [],
        'adv_predictions': [],
        'adv_targets': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_metrics': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_per_class_metrics': []
    }


# For backward compatibility, export the main classes
__all__ = ['Trainer', 'TrainingManager', '_init_history']


if __name__ == "__main__":
    logging.info("train.py executed directly - use main.py instead")
