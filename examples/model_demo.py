"""
Model demonstration script

Shows how to:
1. Create models with different architectures (controlled by layer parameters)
2. Run inference on dummy data
3. Benchmark model performance
4. Export models

Usage:
    # Create medium-sized model using layer parameters
    python model_demo.py --d_model 256 --n_layers 6

    # Create custom-sized model
    python model_demo.py --d_model 192 --n_layers 4 --n_heads 4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ModelConfig
from model import create_model, list_models, get_model_info, print_model_summary, create_model_from_variant, MODEL_CONFIGS


def demo_model_creation():
    """Demonstrate creating different model variants"""
    print("\n" + "=" * 70)
    print("Model Creation Demo")
    print("=" * 70)

    # List available preset configs
    print("\nAvailable preset configs (for quick prototyping):")
    for name, cfg in list_models().items():
        print(f"  - {name}: d_model={cfg['d_model']}, layers={cfg['n_layers']}")

    # Create models using different methods
    print("\n--- Method 1: Custom config (recommended) ---")
    config = ModelConfig(d_model=256, n_layers=6, n_heads=4, d_ff=1024)
    model = create_model(config=config, in_channels=64, num_classes=2)
    print_model_summary(model)

    print("\n--- Method 2: Using presets ---")
    config = ModelConfig(**MODEL_CONFIGS['tiny'])
    model = create_model(config=config, in_channels=64, num_classes=2)
    print_model_summary(model)

    print("\n--- Method 3: Quick prototyping (legacy) ---")
    model = create_model_from_variant('medium', in_channels=64, num_classes=2)
    print_model_summary(model)


def demo_inference(d_model=256, n_layers=6, n_heads=4):
    """Demonstrate inference on dummy data"""
    print("\n" + "=" * 70)
    print(f"Inference Demo (d_model={d_model}, n_layers={n_layers})")
    print("=" * 70)

    # Create model
    config = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    model = create_model(config=config, in_channels=64, num_classes=2)
    model.eval()

    # Create dummy input (batch_size=4, time_frames=157, feature_dim=64)
    # Format: [B, T, F] where T=time frames, F=feature dimension (mel bins)
    dummy_input = torch.randn(4, 157, 64)
    print(f"\nInput shape: {dummy_input.shape}")

    # Run inference
    with torch.no_grad():
        start = time.time()
        output = model(dummy_input)
        inference_time = (time.time() - start) * 1000  # ms

    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")

    # Softmax to get probabilities
    probs = torch.softmax(output, dim=1)
    print(f"\nProbabilities:")
    print(f"  Non-cry: {probs[:, 0]}")
    print(f"  Cry:     {probs[:, 1]}")

    print(f"\nInference time: {inference_time:.2f} ms")


def demo_benchmark(d_model=256, n_layers=6, n_heads=4, num_runs=100):
    """Benchmark model performance"""
    print("\n" + "=" * 70)
    print(f"Benchmark Demo (d_model={d_model}, n_layers={n_layers})")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    config = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    model = create_model(config=config, in_channels=64, num_classes=2)
    model = model.to(device)
    model.eval()

    # Create dummy input on device
    # Format: [B, T, F] where T=time frames, F=feature dimension
    dummy_input = torch.randn(1, 157, 64).to(device)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    times = np.array(times) * 1000  # Convert to ms

    print("\nResults:")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")
    print(f"  P95:  {np.percentile(times, 95):.2f} ms")
    print(f"  FPS:  {1000/np.mean(times):.1f}")


def demo_compare_models():
    """Compare different model architectures"""
    print("\n" + "=" * 70)
    print("Model Comparison (controlled by layer parameters)")
    print("=" * 70)

    print("\n{:12s} {:>10s} {:>10s} {:>12s} {:>15s} {:>12s}".format(
        "Category", "d_model", "n_layers", "Params (M)", "Size (MB)", "MACs (M)"
    ))
    print("-" * 75)

    # Define models by layer parameters
    models = [
        ('Large', 512, 12, 8, 2048),
        ('Medium', 256, 6, 4, 1024),
        ('Small', 192, 4, 4, 768),
        ('Tiny', 128, 3, 2, 256),
        ('Nano', 64, 2, 2, 128),
    ]

    for name, d_model, n_layers, n_heads, d_ff in models:
        config = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
        model = create_model(config=config, in_channels=64, num_classes=2)
        info = get_model_info(model)

        params_m = info['total_params'] / 1e6
        size_mb = info['model_size_mb']
        macs_m = info['estimated_macs_m']

        print("{:12s} {:>10d} {:>10d} {:>12.2f} {:>15.2f} {:>12.1f}".format(
            name, d_model, n_layers, params_m, size_mb, macs_m
        ))


def main():
    parser = argparse.ArgumentParser(
        description='CryTransformer Model Demo - Architecture controlled by layer parameters'
    )
    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model hidden dimension (default: 256)')
    parser.add_argument('--n_layers', type=int, default=6,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='Feed-forward dimension (default: 1024)')

    parser.add_argument('--mode', type=str, default='all',
                       choices=['creation', 'inference', 'benchmark', 'compare', 'all'],
                       help='Demo mode')
    args = parser.parse_args()

    if args.mode in ['creation', 'all']:
        demo_model_creation()

    if args.mode in ['inference', 'all']:
        demo_inference(args.d_model, args.n_layers, args.n_heads)

    if args.mode in ['benchmark', 'all']:
        demo_benchmark(args.d_model, args.n_layers, args.n_heads)

    if args.mode in ['compare', 'all']:
        demo_compare_models()

    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
