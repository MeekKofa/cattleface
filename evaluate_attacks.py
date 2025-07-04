#!/usr/bin/env python3

import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.torch_utils import fix_multiprocessing_issues

# Import our multiprocessing fix utility


# Apply the fix at the beginning before importing other modules that might use multiprocessing
fix_multiprocessing_issues()

from loader.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from gan.attack.attack_loader import AttackHandler
from gan.defense.prune import Pruner
from utils.adv_metrics import compute_adv_success_rate
from utils.metrics import Metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _is_classification_model(arch: str) -> bool:
    """Check if the given architecture is a classification model"""
    classification_models = ['resnet', 'densenet', 'vgg', 'vgg_myccc', 'vgg_yolov8', 'meddef1']
    # Handle case where arch might be a list or string
    if isinstance(arch, list):
        arch = arch[0] if arch else ''
    arch_str = str(arch).lower()
    return any(model in arch_str for model in classification_models)


def _should_force_classification(dataset_loader, dataset_name: str, arch: str) -> bool:
    """Determine if we should force classification mode for object detection dataset"""
    is_obj_detection_dataset = dataset_loader._is_object_detection_dataset(dataset_name)
    is_classification_model = _is_classification_model(arch)
    return is_obj_detection_dataset and is_classification_model


class AttackEvaluator:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.data
        self.model_name = args.arch
        self.depth = args.depth
        self.device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu")
        
        # Load dataset
        self.dataset_loader = DatasetLoader()
        # Check if we need to force classification mode
        force_classification = _should_force_classification(self.dataset_loader, self.dataset_name, self.model_name)
        _, _, self.test_loader = self.dataset_loader.load_data(
            dataset_name=self.dataset_name,
            batch_size={
                'train': args.batch_size,
                'val': args.batch_size,
                'test': args.batch_size
            },
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            force_classification=force_classification
        )
        
        # Get number of classes
        dataset = self.test_loader.dataset
        if hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            self.num_classes = len(dataset.class_to_idx)
        else:
            raise AttributeError("Dataset does not contain class information")
        
        # Model loader
        self.model_loader = ModelLoader(
            self.device, 
            self.model_name,
            pretrained=False
        )
        
        # Create output directory
        self.output_dir = os.path.join(
            "out", 
            "attack_evaluation", 
            self.dataset_name, 
            f"{self.model_name}_{self.depth}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = []
    
    def count_parameters(self, model):
        """
        Count the total and non-zero parameters in a model.
        For pruned models, this counts actual non-zero weights correctly.
        """
        total_params = 0
        nonzero_params = 0
        
        # Collect pruned parameter statistics
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask') and module.weight_mask is not None:
                # This is a pruned module
                weight = module.weight
                mask = module.weight_mask
                param_total = weight.numel()
                nonzero_count = torch.sum(mask).item()
                
                total_params += param_total
                nonzero_params += nonzero_count
                logging.info(f"Pruned module {name}: {nonzero_count}/{param_total} non-zero weights "
                            f"({nonzero_count/param_total*100:.2f}%)")
        
        # If we didn't find any pruned layers, do a normal parameter count
        if total_params == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_total = param.numel()
                    total_params += param_total
                    nonzero_count = torch.count_nonzero(param).item()
                    nonzero_params += nonzero_count
            
            logging.info(f"No pruning masks found. Total parameters: {total_params}, "
                        f"Non-zero parameters: {nonzero_params}")
        
        return total_params, nonzero_params
    
    def load_model(self, model_path):
        """Load model from path"""
        try:
            models_and_names = self.model_loader.get_model(
                model_name=self.model_name,
                depth=float(self.depth),
                input_channels=3,
                num_classes=self.num_classes
            )
            
            if not models_and_names:
                logging.error("No models returned from model loader")
                return None
            
            model, _ = models_and_names[0]
            model = model.to(self.device)
            
            # Load weights
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logging.info(f"Loaded model from {model_path}")
            else:
                logging.error(f"Model path {model_path} does not exist")
                return None
            
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    
    def apply_pruning(self, model, prune_rate):
        """Apply pruning to the model"""
        pruner = Pruner(model, prune_rate)
        pruned_model = pruner.unstructured_prune()
        return pruned_model
    
    def evaluate_clean(self, model, num_samples=None):
        """Evaluate model on clean test data"""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating on clean data"):
                if num_samples and total >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Get predictions
                probs = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                # Store results
                all_preds.append(predicted.cpu())
                all_targets.append(target.cpu())
                all_probs.append(probs.cpu())
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        # Concatenate all batches
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # Calculate metrics
        metrics = Metrics.calculate_metrics(all_targets, all_preds, all_probs)
        metrics['accuracy'] = correct / total
        
        return metrics, all_targets, all_preds, all_probs
    
    def evaluate_adversarial(self, model, attack_type, attack_params, num_samples=None):
        """Evaluate model on adversarial examples"""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_adv_preds = []
        all_adv_probs = []
        
        # Track numbers for better ASR calculation
        successful_attacks = 0
        valid_attacks = 0
        
        # Get all possible class indices from the dataset
        # This helps ensure we handle all classes correctly even if they're not in the current evaluation batch
        if hasattr(self.test_loader.dataset, 'classes'):
            num_classes = len(self.test_loader.dataset.classes)
        elif hasattr(self.test_loader.dataset, 'class_to_idx'):
            num_classes = len(self.test_loader.dataset.class_to_idx)
        else:
            num_classes = self.num_classes  # Use the class variable as fallback
        
        # Configure attack
        attack_config = argparse.Namespace(
            attack_type=attack_type,
            **attack_params
        )
        attack_handler = AttackHandler(model, attack_type, attack_config)
        
        for data, target in tqdm(self.test_loader, desc=f"Evaluating against {attack_type}"):
            if num_samples and total >= num_samples:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Get clean predictions first
            with torch.no_grad():
                clean_output = model(data)
                _, clean_predicted = torch.max(clean_output.data, 1)
            
            # Generate adversarial examples
            batch_results = attack_handler.generate_adversarial_samples_batch(data, target)
            adv_data = batch_results['adversarial'].to(self.device)
            
            # Get predictions on adversarial examples
            with torch.no_grad():
                adv_output = model(adv_data)
                adv_probs = torch.nn.functional.softmax(adv_output, dim=1)
                _, adv_predicted = torch.max(adv_output.data, 1)
            
            # Store results
            all_preds.append(clean_predicted.cpu())
            all_adv_preds.append(adv_predicted.cpu())
            all_targets.append(target.cpu())
            all_adv_probs.append(adv_probs.cpu())
            
            # Count for ASR calculation:
            # For each sample, if originally correct and now wrong after attack, it's a successful attack
            correct_before = clean_predicted == target
            correct_after = adv_predicted == target
            
            # Successful attack = was correct before, wrong after
            batch_successful = torch.logical_and(correct_before, torch.logical_not(correct_after)).sum().item()
            # Valid attack = was correct before (can only attack correct predictions)
            batch_valid = correct_before.sum().item()
            
            successful_attacks += batch_successful
            valid_attacks += batch_valid
            
            total += target.size(0)
            correct += (adv_predicted == target).sum().item()
            
        # Concatenate all batches
        all_preds = torch.cat(all_preds).numpy()
        all_adv_preds = torch.cat(all_adv_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_adv_probs = torch.cat(all_adv_probs).numpy()
        
        # Calculate metrics with comprehensive error handling
        try:
            adv_metrics = Metrics.calculate_metrics(all_targets, all_adv_preds, all_adv_probs)
        except ValueError as e:
            error_msg = str(e)
            # Handle case where all predictions are for a single class
            if "Only one class present in y_true" in error_msg:
                logging.warning(f"Attack {attack_type} led to single class predictions. Setting ROC AUC and AP metrics to 0.0")
                adv_metrics = self._create_fallback_metrics()
            # Handle dimension mismatch between labels and probabilities
            elif "Number of classes in y_true not equal to the number of columns in 'y_score'" in error_msg:
                logging.warning(f"Attack {attack_type} caused class dimension mismatch. Using basic metrics only.")
                adv_metrics = self._create_fallback_metrics()
                # Try to calculate basic metrics that don't depend on probability scores
                try:
                    adv_metrics['precision'] = precision_score(all_targets, all_adv_preds, average='macro', zero_division=0)
                    adv_metrics['recall'] = recall_score(all_targets, all_adv_preds, average='macro', zero_division=0)
                    adv_metrics['f1'] = f1_score(all_targets, all_adv_preds, average='macro', zero_division=0)
                except:
                    logging.warning(f"Could not calculate basic metrics for attack {attack_type}")
            else:
                # Re-raise if it's a different error
                logging.error(f"Unexpected error in metrics calculation: {error_msg}")
                adv_metrics = self._create_fallback_metrics()
        
        adv_metrics['accuracy'] = correct / total
        
        # Calculate corrected attack success rate
        if valid_attacks > 0:
            # ASR = (number of successfully attacked samples) / (number of correctly classified samples before attack)
            attack_success_rate = 100.0 * successful_attacks / valid_attacks
        else:
            # If no valid attacks (no samples correctly classified before attack),
            # then attack success rate is technically undefined/0
            attack_success_rate = 0.0
            
        adv_metrics['attack_success_rate'] = attack_success_rate
        
        # Also add original attack success calculation for comparison
        original_asr = compute_adv_success_rate(all_targets, all_preds, all_adv_preds)
        adv_metrics['original_attack_success_rate'] = original_asr
        
        logging.info(f"Attack success metrics for {attack_type}:")
        logging.info(f"  - Total samples: {total}")
        logging.info(f"  - Correct before attack: {valid_attacks}")
        logging.info(f"  - Successful attacks (correct→incorrect): {successful_attacks}")
        logging.info(f"  - Improved ASR: {attack_success_rate:.2f}%")
        logging.info(f"  - Original ASR calculation: {original_asr:.2f}%")
        
        return adv_metrics, all_targets, all_adv_preds, all_adv_probs

    def _create_fallback_metrics(self):
        """Create a basic metrics dict with defaults for failure cases"""
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'precision_micro': 0.0,
            'precision_weighted': 0.0,
            'recall_micro': 0.0,
            'recall_weighted': 0.0,
            'f1_micro': 0.0,
            'f1_weighted': 0.0,
            'specificity': 0.0,
            'balanced_accuracy': 0.0,
            'mcc': 0.0,
            'cohen_kappa': 0.0,
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'log_loss': 15.0,  # High log loss indicating poor performance
            'brier_score': 1.0,  # Worst possible Brier score
            'ece': 1.0          # Worst expected calibration error
        }
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        # Load base model
        model = self.load_model(self.args.model_path)
        if model is None:
            return
        
        # Count parameters of original model
        total_params, nonzero_params = self.count_parameters(model)
        logging.info(f"Original model - Total parameters: {total_params/1e6:.2f}M, Non-zero: {nonzero_params/1e6:.2f}M ({nonzero_params/total_params*100:.2f}%)")
        
        # Full model name including depth
        full_model_name = f"{self.model_name}_{self.depth}"
        
        # Evaluate on clean data
        logging.info("Evaluating original model on clean test data")
        clean_metrics, clean_targets, clean_preds, clean_probs = self.evaluate_clean(model)
        
        # Add results to tracking
        self.results.append({
            'dataset_name': self.dataset_name,
            'model_name': full_model_name,
            'model_type': 'original',
            'prune_rate': 0.0,
            'attack_type': 'none',
            'accuracy': clean_metrics['accuracy'],
            'attack_success_rate': 0.0,
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'params_ratio': nonzero_params/total_params*100,
            **{k: v for k, v in clean_metrics.items() if k not in ['confusion_matrix', 'tp', 'tn', 'fp', 'fn']}
        })
        
        # Evaluate against different attacks
        attack_types = self.args.attack_types or ['fgsm', 'pgd']
        attack_params = {
            # Fast attacks - keep as is
            'fgsm': {'attack_eps': self.args.attack_eps},
            'pgd': {
                'attack_eps': self.args.attack_eps,
                'attack_alpha': self.args.attack_eps / 10,
                'attack_steps': 20
            },
            'bim': {
                'attack_eps': self.args.attack_eps,
                'attack_alpha': self.args.attack_eps / 10,
                'attack_steps': 10
            },
            'jsma': {'attack_eps': self.args.attack_eps},
            
            # Optimized parameters for slower attacks
            'cw': {
                'attack_eps': self.args.attack_eps, 
                'attack_c': 1.0,
                'attack_iterations': 10,     # Reduced from 40 to 10
                'attack_lr': 0.05,           # Increased for faster convergence
                'attack_binary_steps': 3,    # Reduced from 5 to 3
                'attack_confidence': 0
            },
            'zoo': {
                'attack_eps': self.args.attack_eps,
                'attack_iterations': 10,     # Reduced from 40 to 10
                'attack_h': 0.001,
                'attack_binary_steps': 3     # Reduced from 5 to 3
            },
            'boundary': {
                'attack_eps': self.args.attack_eps,
                'attack_steps': 20,          # Reduced from 50 to 20
                'attack_spherical_step': 0.05, # Increased for faster movement
                'attack_source_step': 0.05,    # Increased for faster movement
                'attack_step_adaptation': 1.5,
                'attack_max_directions': 10   # Reduced from 25 to 10
            },
            'elasticnet': {
                'attack_eps': self.args.attack_eps,
                'attack_alpha': 0.05,        # Increased for faster convergence
                'attack_iterations': 10,     # Reduced from 40 to 10
                'attack_beta': 1.0
            },
            'onepixel': {
                'attack_eps': self.args.attack_eps,
                'attack_pixel_count': 1,
                'attack_max_iter': 20,       # Reduced from 40 to 20
                'attack_popsize': 5         # Reduced from 10 to 5
            }
        }

        # Add a maximum time limit for slower attacks
        time_limits = {
            'cw': 60,        # 1 minute max per batch 
            'zoo': 60,
            'boundary': 60,
            'elasticnet': 60,
            'onepixel': 60
        }
        
        # Add support for limiting evaluation to a subset of samples for slow attacks
        max_samples_for_slow_attacks = {
            'cw': 100,       # Only evaluate on 100 samples
            'zoo': 100,
            'boundary': 100,
            'elasticnet': 100,
            'onepixel': 100
        }
        
        for attack_type in attack_types:
            logging.info(f"Evaluating original model against {attack_type} attack")
            
            # For slow attacks, limit number of samples
            num_samples = None
            if attack_type in max_samples_for_slow_attacks:
                num_samples = max_samples_for_slow_attacks[attack_type]
                logging.info(f"Limiting {attack_type} attack to {num_samples} samples for speed")
            
            adv_metrics, _, _, _ = self.evaluate_adversarial(
                model, 
                attack_type, 
                attack_params[attack_type],
                num_samples=num_samples
            )
            
            # Add results to tracking
            self.results.append({
                'dataset_name': self.dataset_name,
                'model_name': full_model_name,
                'model_type': 'original',
                'prune_rate': 0.0,
                'attack_type': attack_type,
                'accuracy': adv_metrics['accuracy'],
                'attack_success_rate': adv_metrics['attack_success_rate'],
                'total_params': total_params,
                'nonzero_params': nonzero_params,
                'params_ratio': nonzero_params/total_params*100,
                **{k: v for k, v in adv_metrics.items() if k not in ['confusion_matrix', 'tp', 'tn', 'fp', 'fn']}
            })
        
        # Evaluate pruned models
        prune_rates = self.args.prune_rates or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        for prune_rate in prune_rates:
            logging.info(f"Applying pruning with rate {prune_rate}")
            pruned_model = self.apply_pruning(model, prune_rate)
            
            # Count parameters after pruning
            total_params, nonzero_params = self.count_parameters(pruned_model)
            logging.info(f"Pruned model (rate={prune_rate}) - Total parameters: {total_params/1e6:.2f}M, Non-zero: {nonzero_params/1e6:.2f}M ({nonzero_params/total_params*100:.2f}%)")
            
            # Evaluate pruned model on clean data
            logging.info(f"Evaluating pruned model (rate={prune_rate}) on clean test data")
            pruned_clean_metrics, pruned_clean_targets, pruned_clean_preds, pruned_clean_probs = self.evaluate_clean(pruned_model)
            
            # Add results to tracking
            self.results.append({
                'dataset_name': self.dataset_name,
                'model_name': full_model_name,
                'model_type': 'pruned',
                'prune_rate': prune_rate,
                'attack_type': 'none',
                'accuracy': pruned_clean_metrics['accuracy'],
                'attack_success_rate': 0.0,
                'total_params': total_params,
                'nonzero_params': nonzero_params,
                'params_ratio': nonzero_params/total_params*100,
                **{k: v for k, v in pruned_clean_metrics.items() if k not in ['confusion_matrix', 'tp', 'tn', 'fp', 'fn']}
            })
            
            # Evaluate pruned model against different attacks
            for attack_type in attack_types:
                logging.info(f"Evaluating pruned model (rate={prune_rate}) against {attack_type} attack")
                
                # For slow attacks, limit number of samples
                num_samples = None
                if attack_type in max_samples_for_slow_attacks:
                    num_samples = max_samples_for_slow_attacks[attack_type]
                
                pruned_adv_metrics, _, _, _ = self.evaluate_adversarial(
                    pruned_model, 
                    attack_type, 
                    attack_params[attack_type],
                    num_samples=num_samples
                )
                
                # Add results to tracking
                self.results.append({
                    'dataset_name': self.dataset_name,
                    'model_name': full_model_name,
                    'model_type': 'pruned',
                    'prune_rate': prune_rate,
                    'attack_type': attack_type,
                    'accuracy': pruned_adv_metrics['accuracy'],
                    'attack_success_rate': pruned_adv_metrics['attack_success_rate'],
                    'total_params': total_params,
                    'nonzero_params': nonzero_params,
                    'params_ratio': nonzero_params/total_params*100,
                    **{k: v for k, v in pruned_adv_metrics.items() if k not in ['confusion_matrix', 'tp', 'tn', 'fp', 'fn']}
                })
            
            # Save pruned model if requested
            if self.args.save_pruned_models:
                pruned_path = os.path.join(self.output_dir, f"pruned_rate{prune_rate}.pth")
                torch.save(pruned_model.state_dict(), pruned_path)
                logging.info(f"Saved pruned model to {pruned_path}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save evaluation results"""
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, "attack_evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved results to {csv_path}")
        
        # Generate parameter plots
        self.generate_parameter_plots()
        
        # Generate plots
        self.generate_plots()
        
        # Generate summary
        self.generate_summary()

    def generate_plots(self):
        """Generate visualization plots for the results"""
        df = pd.DataFrame(self.results)
        
        # Plot 1: Clean accuracy vs prune rate
        plt.figure(figsize=(10, 6))
        clean_data = df[df['attack_type'] == 'none']
        # Convert pandas Series to numpy arrays before plotting
        prune_rates = clean_data['prune_rate'].to_numpy()
        accuracies = clean_data['accuracy'].to_numpy()
        plt.plot(prune_rates, accuracies, 'o-', label='Clean Accuracy')
        plt.xlabel('Pruning Rate')
        plt.ylabel('Accuracy')
        plt.title('Clean Accuracy vs Pruning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'clean_accuracy_vs_pruning.png'))
        plt.close()
        
        # Plot 2: Attack success rate vs prune rate for different attacks
        plt.figure(figsize=(10, 6))
        attack_types = df['attack_type'].unique()
        attack_types = [at for at in attack_types if at != 'none']
        
        for attack_type in attack_types:
            attack_data = df[df['attack_type'] == attack_type]
            # Convert pandas Series to numpy arrays before plotting
            prune_rates = attack_data['prune_rate'].to_numpy()
            asr = attack_data['attack_success_rate'].to_numpy()
            plt.plot(prune_rates, asr, 'o-', label=f'{attack_type}')
        
        plt.xlabel('Pruning Rate')
        plt.ylabel('Attack Success Rate (%)')
        plt.title('Attack Success Rate vs Pruning Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'attack_success_vs_pruning.png'))
        plt.close()
        
        # Plot 3: Robustness (1-ASR) vs prune rate
        plt.figure(figsize=(10, 6))
        
        for attack_type in attack_types:
            attack_data = df[df['attack_type'] == attack_type]
            # Convert pandas Series to numpy arrays before plotting
            prune_rates = attack_data['prune_rate'].to_numpy()
            robustness = 100 - attack_data['attack_success_rate'].to_numpy()
            plt.plot(prune_rates, robustness, 'o-', label=f'{attack_type}')
        
        plt.xlabel('Pruning Rate')
        plt.ylabel('Robustness (100 - ASR%)')
        plt.title('Adversarial Robustness vs Pruning Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'robustness_vs_pruning.png'))
        plt.close()
        
        # Plot 4: Combined plot showing trade-off between clean accuracy and robustness
        plt.figure(figsize=(12, 8))
        
        for attack_type in attack_types:
            attack_data = df[df['attack_type'] == attack_type]
            accuracy_data = df[(df['attack_type'] == 'none') & df['prune_rate'].isin(attack_data['prune_rate'])]
            
            # Convert pandas Series to numpy arrays before plotting
            accuracies = accuracy_data['accuracy'].to_numpy()
            robustness = 100 - attack_data['attack_success_rate'].to_numpy()
            prune_rates = attack_data['prune_rate'].to_numpy()
            
            plt.scatter(accuracies, robustness, label=f'{attack_type}', s=100)
            
            # Add annotations for prune rates
            for i, (acc, rob, pr) in enumerate(zip(accuracies, robustness, prune_rates)):
                plt.annotate(f"{pr:.1f}", (acc, rob), 
                             fontsize=9, ha='center', va='bottom')
        
        plt.xlabel('Clean Accuracy')
        plt.ylabel('Robustness (100 - ASR%)')
        plt.title('Accuracy-Robustness Trade-off for Different Pruning Rates')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'accuracy_robustness_tradeoff.png'))
        plt.close()

    def generate_summary(self):
        """Generate a summary text file with key findings"""
        df = pd.DataFrame(self.results)
        
        # Set up the summary file path
        summary_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"=== MedDef Attack Evaluation Summary ===\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Model: {self.model_name}_{self.depth}\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Model parameter summary
            f.write("=== Model Parameter Summary ===\n")
            param_data = df.drop_duplicates(['model_type', 'prune_rate'])[['model_type', 'prune_rate', 'total_params', 'nonzero_params', 'params_ratio']]
            for _, row in param_data.sort_values(['model_type', 'prune_rate']).iterrows():
                model_type = "Original" if row['model_type'] == 'original' else "Pruned"
                f.write(f"{model_type} (Prune Rate {row['prune_rate']:.1f}): "
                        f"Total params: {row['total_params']/1e6:.2f}M, "
                        f"Non-zero: {row['nonzero_params']/1e6:.2f}M "
                        f"({row['params_ratio']:.2f}%)\n")
            
            f.write("\n")
            
            # Clean Accuracy Summary
            f.write("=== Clean Accuracy Summary ===\n")
            clean_data = df[df['attack_type'] == 'none']
            for _, row in clean_data.iterrows():
                f.write(f"Prune Rate {row['prune_rate']:.1f}: "
                        f"Accuracy = {row['accuracy']:.4f}, "
                        f"Params = {row['nonzero_params']/1e6:.2f}M ({row['params_ratio']:.2f}%)\n")
            
            f.write("\n")
            
            # Attack Success Rate Summary with clarification
            f.write("=== Attack Success Rate Summary ===\n")
            for attack_type in sorted(df['attack_type'].unique()):
                if attack_type == 'none':
                    continue
                    
                f.write(f"\n{attack_type.upper()} Attack:\n")
                attack_data = df[df['attack_type'] == attack_type]
                for _, row in attack_data.iterrows():
                    model_acc = df[(df['attack_type'] == 'none') & (df['prune_rate'] == row['prune_rate'])]['accuracy'].values[0]
                    attack_acc = row['accuracy']
                    asr = row['attack_success_rate']
                    
                    # Add a note if the ASR might be misleading
                    asr_note = ""
                    if model_acc <= 0.3 and asr == 0.0:
                        asr_note = " (Note: Low baseline accuracy may affect ASR)"
                    elif abs(model_acc - attack_acc) < 0.01 and asr == 0.0:
                        asr_note = " (Attack ineffective, model accuracy unchanged)"
                    
                    f.write(f"  Prune Rate {row['prune_rate']:.1f}: "
                            f"ASR = {row['attack_success_rate']:.2f}%{asr_note}, "
                            f"Accuracy = {row['accuracy']:.4f}, "
                            f"Params = {row['nonzero_params']/1e6:.2f}M ({row['params_ratio']:.2f}%)\n")
            
            f.write("\n")
            
            # Best pruning rate for each attack
            f.write("=== Best Pruning Rate for Robustness ===\n")
            for attack_type in sorted(df['attack_type'].unique()):
                if attack_type == 'none':
                    continue
                    
                attack_data = df[df['attack_type'] == attack_type]
                max_robust_idx = (100 - attack_data['attack_success_rate']).idxmax()
                max_robust_row = attack_data.loc[max_robust_idx]
                
                f.write(f"{attack_type.upper()}: Prune Rate {max_robust_row['prune_rate']:.1f} " +
                       f"(ASR = {max_robust_row['attack_success_rate']:.2f}%, " +
                       f"Accuracy = {max_robust_row['accuracy']:.4f}, " +
                       f"Params = {max_robust_row['nonzero_params']/1e6:.2f}M ({max_robust_row['params_ratio']:.2f}%))\n")
            
            f.write("\n")
            
            # Accuracy-Robustness Trade-off
            f.write("=== Accuracy vs Robustness Trade-off ===\n")
            # Find the best trade-off (this is subjective, but we'll use a simple metric)
            # We'll use accuracy * (1 - ASR/100) as a simple trade-off metric
            tradeoff_data = []
            
            for attack_type in sorted(df['attack_type'].unique()):
                if attack_type == 'none':
                    continue
                    
                attack_data = df[df['attack_type'] == attack_type]
                clean_data = df[df['attack_type'] == 'none']
                
                for _, row in attack_data.iterrows():
                    prune_rate = row['prune_rate']
                    clean_row = clean_data.loc[clean_data['prune_rate'] == prune_rate].iloc[0]
                    clean_acc = clean_row['accuracy'] 
                    robustness = 1 - row['attack_success_rate']/100
                    trade_off_score = clean_acc * robustness
                    nonzero_params = row['nonzero_params']
                    params_ratio = row['params_ratio']
                    
                    tradeoff_data.append({
                        'attack_type': attack_type,
                        'prune_rate': prune_rate,
                        'clean_acc': clean_acc,
                        'robustness': robustness * 100,  # Convert back to percentage
                        'trade_off_score': trade_off_score,
                        'nonzero_params': nonzero_params,
                        'params_ratio': params_ratio
                    })
            
            tradeoff_df = pd.DataFrame(tradeoff_data)
            for attack_type in sorted(tradeoff_df['attack_type'].unique()):
                attack_data = tradeoff_df[tradeoff_df['attack_type'] == attack_type]
                best_idx = attack_data['trade_off_score'].idxmax()
                best_row = attack_data.loc[best_idx]
                
                f.write(f"{attack_type.upper()}: Best trade-off at Prune Rate {best_row['prune_rate']:.1f} " +
                       f"(Clean Acc = {best_row['clean_acc']:.4f}, " +
                       f"Robustness = {best_row['robustness']:.2f}%, " +
                       f"Params = {best_row['nonzero_params']/1e6:.2f}M ({best_row['params_ratio']:.2f}%))\n")
            
            f.write("\n")
            
            # Add recommendations
            f.write("=== Recommendations ===\n")
            if len(tradeoff_df) > 0:
                # Get the overall best trade-off across all attacks
                best_overall_idx = tradeoff_df['trade_off_score'].idxmax()
                best_overall = tradeoff_df.loc[best_overall_idx]
                
                f.write(f"Best overall pruning rate: {best_overall['prune_rate']:.1f} " +
                       f"(against {best_overall['attack_type'].upper()} attack)\n")
                f.write(f"This provides a clean accuracy of {best_overall['clean_acc']:.4f} " +
                       f"with robustness of {best_overall['robustness']:.2f}% " +
                       f"using {best_overall['nonzero_params']/1e6:.2f}M parameters ({best_overall['params_ratio']:.2f}% of original)\n")
            else:
                f.write("Not enough data to provide recommendations.\n")
        
        logging.info(f"Generated summary at {summary_path}")
        return summary_path

    # Add a new method to generate parameter visualization
    def generate_parameter_plots(self):
        """Generate parameter-related visualization plots"""
        df = pd.DataFrame(self.results)
        
        # Get unique combinations of model_type and prune_rate
        param_data = df.drop_duplicates(['model_type', 'prune_rate'])[['model_type', 'prune_rate', 
                                                                 'total_params', 'nonzero_params', 'params_ratio']]
        
        # Sort by prune rate
        param_data = param_data.sort_values('prune_rate')
        
        # Plot: Parameter count vs prune rate - Convert to numpy arrays first to avoid pandas indexing issues
        plt.figure(figsize=(10, 6))
        prune_rates = param_data['prune_rate'].to_numpy()
        nonzero_params = param_data['nonzero_params'].to_numpy() / 1e6
        total_params = param_data['total_params'].to_numpy() / 1e6
        
        plt.plot(prune_rates, nonzero_params, 'o-', 
                color='blue', label='Non-zero parameters (M)')
        plt.plot(prune_rates, total_params, '--', 
                color='gray', label='Total parameters (M)')
        plt.xlabel('Pruning Rate')
        plt.ylabel('Parameters (millions)')
        plt.title('Model Size vs Pruning Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'parameters_vs_pruning.png'))
        plt.close()
        
        # Add a parameter efficiency plot - Convert to numpy arrays first
        plt.figure(figsize=(10, 6))
        
        clean_data = df[df['attack_type'] == 'none'].copy()
        clean_data['acc_per_param'] = clean_data['accuracy'] / (clean_data['nonzero_params']/1e6)
        
        prune_rates = clean_data['prune_rate'].to_numpy()
        acc_per_param = clean_data['acc_per_param'].to_numpy()
        
        plt.plot(prune_rates, acc_per_param, 'o-', 
                color='green', label='Accuracy per million parameters')
        plt.xlabel('Pruning Rate')
        plt.ylabel('Clean Accuracy per Million Parameters')
        plt.title('Parameter Efficiency vs Pruning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'parameter_efficiency.png'))
        plt.close()
        
        # Plot: 3D visualization of accuracy, robustness, and parameter count
        attack_types = [at for at in df['attack_type'].unique() if at != 'none']
        
        if attack_types:  # If we have attack data
            try:
                # Choose first attack for visualization
                attack_type = attack_types[0]
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                attack_data = df[df['attack_type'] == attack_type]
                clean_data = df[df['attack_type'] == 'none']
                
                # Prepare data points - Convert all to numpy arrays
                prune_rates = attack_data['prune_rate'].values
                # Match clean data to attack data by prune rate
                accuracies = []
                for pr in prune_rates:
                    matching_clean = clean_data[clean_data['prune_rate'] == pr]
                    if not matching_clean.empty:
                        accuracies.append(matching_clean['accuracy'].values[0])
                    else:
                        accuracies.append(np.nan)  # Handle missing data
                
                accuracies = np.array(accuracies)
                robustness = 100 - attack_data['attack_success_rate'].values
                param_ratios = attack_data['params_ratio'].values
                
                # Remove any NaN values
                valid_indices = ~np.isnan(accuracies)
                if np.any(valid_indices):
                    prune_rates = prune_rates[valid_indices]
                    accuracies = accuracies[valid_indices]
                    robustness = robustness[valid_indices]
                    param_ratios = param_ratios[valid_indices]
                    
                    # Create scatter plot
                    scatter = ax.scatter(accuracies, robustness, param_ratios,
                                       c=param_ratios, cmap='viridis', s=100)
                    
                    # Add labels for points
                    for i, rate in enumerate(prune_rates):
                        ax.text(accuracies[i], robustness[i], param_ratios[i], 
                               f"{rate:.1f}", fontsize=9)
                    
                    ax.set_xlabel('Clean Accuracy')
                    ax.set_ylabel('Robustness (100 - ASR%)')
                    ax.set_zlabel('Parameters Remaining (%)')
                    ax.set_title(f'3D Trade-off: Accuracy, Robustness ({attack_type}), and Model Size')
                    
                    # Add color bar
                    cbar = fig.colorbar(scatter, ax=ax, label='Parameters Remaining (%)')
                    
                    plt.savefig(os.path.join(self.output_dir, 'tradeoff_3d.png'))
                plt.close()
            except Exception as e:
                logging.error(f"Error generating 3D plot: {e}")
                # Continue with other plots even if 3D plot fails


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model robustness against adversarial attacks')
    
    # Model parameters
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--arch', type=str, required=True, help='Model architecture')
    parser.add_argument('--depth', type=float, required=True, help='Model depth/variant')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    
    # Attack parameters
    parser.add_argument('--attack_types', nargs='+', default=['fgsm', 'pgd'], 
                        help='Attack types to evaluate (default: fgsm pgd)')
    parser.add_argument('--attack_eps', type=float, default=0.05, 
                        help='Attack epsilon/strength parameter (default: 0.1)')
    
    # Pruning parameters
    parser.add_argument('--prune_rates', nargs='+', type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9], 
                        help='Pruning rates to evaluate (default: 0.0 0.1 0.3 0.5 0.7 0.9)')
    parser.add_argument('--save_pruned_models', action='store_true', 
                        help='Save the pruned model variants')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (default: 4)')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin_memory')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0], help='GPU IDs to use')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluator = AttackEvaluator(args)
    evaluator.run_evaluation()
