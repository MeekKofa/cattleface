import warnings
import argparse
import logging
from argument_parser import parse_args
from utils.task_handler import TaskHandler
from utils.logger import setup_logger
import torch.multiprocessing as mp
import torchvision
import torch
import os
import utils.pandas_patch
from utils.matplotlib_config import configure_matplotlib_backend
from utils.file_handler import FileHandler

# Configure matplotlib backend before any other imports that might use matplotlib
configure_matplotlib_backend()

# Suppress ALL warnings right from the start
warnings.filterwarnings("ignore")

# Apply visualization patch
try:
    from utils.visual.visualization_patch import patch_visualization
    patch_visualization()
except:
    pass

# Suppress common warnings
FileHandler.suppress_warnings()


def setup_environment(args):
    print("Torch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)
    print("CUDA available: ", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device ID: {i}")
        print(f"  Name: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
        print(f"  Is Integrated: {props.is_integrated}")
        print(f"  Is Multi-GPU Board: {props.is_multi_gpu_board}")
        # List all available attributes
        for attr in dir(props):
            if not attr.startswith('_'):
                print(f"  {attr}: {getattr(props, attr)}")
        print()

    # Set up logging with model-specific path when available
    task_name = args.task_name
    # Fix: dataset_name can be str or list, handle both
    dataset_name = args.data[0] if hasattr(args, 'data') and isinstance(args.data, (list, tuple)) else args.data

    model_name = None
    if hasattr(args, 'arch') and args.arch and hasattr(args, 'depth'):
        arch = args.arch
        if isinstance(args.depth, dict) and arch in args.depth:
            depth_value = str(args.depth[arch][0]) if args.depth[arch] else ""
            model_name = f"{arch}_{depth_value}" if depth_value else arch

    setup_logger(task_name=task_name, dataset_name=dataset_name, model_name=model_name)
    logging.info("Main script started.")


def parse_args():
    parser = argparse.ArgumentParser(description='Cattlebody Training Script')
    parser.add_argument('--data', type=str,
                        default='cattlebody', help='dataset name')
    parser.add_argument('--arch', type=str,
                        default='vgg_yolov8', help='model architecture')
    parser.add_argument(
        '--depth', type=str, default='{"vgg_yolov8": [16]}', help='model depth configuration')
    parser.add_argument('--train_batch', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='pin memory for data loader')
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                        default=[0], dest='gpu_ids', help='GPU IDs to use')
    parser.add_argument('--task_name', type=str,
                        default='normal_training', help='task name')
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='optimizer type')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    # Add missing arguments
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='optimizer weight decay')
    parser.add_argument('--scheduler', type=str, default='none', help='learning rate scheduler type')
    parser.add_argument('--min_epochs', type=int, default=0, help='minimum number of epochs before early stopping')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--augment', action='store_true', help='enable advanced data augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for loss')
    return parser.parse_args()


def main():
    # Basic setup
    # Only set start method if not already set
    import multiprocessing as mp
    try:
        if os.name == 'nt':
            mp.set_start_method('spawn', force=True)
        else:
            mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        # Context already set, ignore
        pass

    args = parse_args()
    # Fix depth argument: convert string to dict if needed
    import ast
    if isinstance(args.depth, str):
        try:
            args.depth = ast.literal_eval(args.depth)
        except Exception:
            args.depth = {"vgg_yolov8": [16]}  # fallback default
    # Set device attribute if not provided
    if not hasattr(args, 'device') or args.device is None:
        args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available(
        ) and args.gpu_ids else "cpu"
    logging.info("Starting training with arguments: %s", args)
    setup_environment(args)
    torch.cuda.empty_cache()

    # Initialize TaskHandler with all tasks
    task_handler = TaskHandler(args)

    # Execute task based on task_name
    if args.task_name == 'normal_training':
        # Add quick_test_after_training flag to arguments
        run_test = getattr(args, 'quick_test_after_training', False)
        # Only call run_train once, not in a loop
        task_handler.run_train(run_test)
    elif args.task_name == 'attack':
        task_handler.run_attack()
    elif args.task_name == 'defense':
        task_handler.run_defense()
    else:
        logging.error(f"Unknown task: {args.task_name}. No task was executed.")


if __name__ == "__main__":
    main()
