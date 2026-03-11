"""
Evaluation script for CryTransformer models

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --test_list audio_list/test.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config
from dataset.dataset import CryDataset
from model import create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Input: List of (features, label) tuples where features is [T, F] numpy array
    Output: (features_tensor, label_indices) where features_tensor is [B, T, F]
    """
    features_list, labels = zip(*batch)

    # Stack features directly (they should all have same shape [T, F])
    features = torch.from_numpy(np.stack(features_list)).float()  # [B, T, F]

    label_to_idx = {'cry': 1, 'other': 0}
    label_indices = torch.tensor([label_to_idx.get(l, 0) for l in labels], dtype=torch.long)

    return features, label_indices


class Evaluator:
    """Model evaluator"""

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on dataset"""
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []

        for features, targets in tqdm(dataloader, desc="Evaluating"):
            # Features are already in [B, T, F] format from dataset
            features = features.to(self.device)

            # Measure inference time
            start_time = time.time()
            outputs = self.model(features)
            inference_times.append((time.time() - start_time) / features.size(0))

            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Cry probability

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='binary'),
            'recall': recall_score(all_targets, all_preds, average='binary'),
            'f1': f1_score(all_targets, all_preds, average='binary'),
            'auc': roc_auc_score(all_targets, all_probs),
            'avg_inference_time': np.mean(inference_times) * 1000,  # ms
        }

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        metrics['confusion_matrix'] = cm
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

        # Specificity
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])

        return metrics, all_targets, all_preds, all_probs

    def benchmark(self, input_shape: Tuple[int, int, int] = (1, 157, 64), num_runs: int = 100) -> Dict:
        """Benchmark model inference speed"""
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start = time.time()
            _ = self.model(dummy_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)

        times = np.array(times) * 1000  # Convert to ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'fps': 1000 / np.mean(times),
        }


def print_results(metrics: Dict):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"AUC-ROC:      {metrics['auc']:.4f}")
    print("-" * 60)
    print(f"Avg Inference: {metrics['avg_inference_time']:.2f} ms")
    print("=" * 60)
    print("\nConfusion Matrix:")
    print(f"  TN={metrics['tn']}, FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}, TP={metrics['tp']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CryTransformer model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_list', type=str, required=True, help='Path to test data list JSON')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    parser.add_argument('--save_report', type=str, default=None, help='Save report to file')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")

    # Load checkpoint
    LOGGER.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    # Load test data
    LOGGER.info(f"Loading test data: {args.test_list}")
    with open(args.test_list, 'r') as f:
        test_dict = json.load(f)

    # Create dataset with feature extraction
    test_dataset = CryDataset(
        test_dict,
        config.dataset,
        aug_config=None,
        feat_config=config.feature
    )
    # Evaluation should use all samples, not balanced sampling
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Calculate input feature dimension (with deltas)
    in_channels = config.feature.n_mels * config.feature.num_channels

    # Create model
    model = create_model(
        config=config.model,
        in_channels=in_channels,
        num_classes=config.model.num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    evaluator = Evaluator(model, config, device)
    metrics, targets, preds, probs = evaluator.evaluate(test_loader)

    print_results(metrics)

    # Benchmark
    if args.benchmark:
        LOGGER.info("Running speed benchmark...")
        bench_results = evaluator.benchmark()
        print("\n" + "=" * 60)
        print("Speed Benchmark")
        print("=" * 60)
        print(f"Mean:  {bench_results['mean_ms']:.2f} ms")
        print(f"Std:   {bench_results['std_ms']:.2f} ms")
        print(f"P95:   {bench_results['p95_ms']:.2f} ms")
        print(f"P99:   {bench_results['p99_ms']:.2f} ms")
        print(f"FPS:   {bench_results['fps']:.1f}")
        print("=" * 60)

    # Save report
    if args.save_report:
        import json as json_mod
        report = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()},
        }
        with open(args.save_report, 'w') as f:
            json_mod.dump(report, f, indent=2)
        LOGGER.info(f"Report saved to: {args.save_report}")


if __name__ == '__main__':
    main()
