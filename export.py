"""
Model export script for CryTransformer

Supports exporting to multiple formats for different deployment scenarios:
- PyTorch (.pt): Native PyTorch format
- ONNX (.onnx): Cross-platform deployment
- TorchScript (.ptl): Mobile/edge deployment
- CoreML (.mlmodel): Apple devices
- TensorFlow Lite (.tflite): Mobile/embedded

Usage:
    # Export to ONNX
    python export.py --checkpoint checkpoints/best_model.pt --format onnx

    # Export to TorchScript
    python export.py --checkpoint checkpoints/best_model.pt --format torchscript

    # Export with quantization
    python export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize int8
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from model import create_model, get_model_info, print_model_summary
from model.transformer import CryTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


class FeatureExtractorWrapper(nn.Module):
    """Wrapper that includes feature extraction in the model"""

    def __init__(self, model: CryTransformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be waveform or features
        # If 1D, assume waveform and we'd need feature extraction
        # For now, assume features are passed
        return self.model(x)


class QuantizedModel(nn.Module):
    """Dynamic quantized model for CPU inference"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    def forward(self, x):
        return self.quantized_model(x)


def export_pytorch(checkpoint_path: str, output_path: str):
    """Export to native PyTorch format"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    torch.save(checkpoint, output_path)
    LOGGER.info(f"PyTorch model saved: {output_path}")


def export_torchscript(
    model: CryTransformer,
    output_path: str,
    example_input: torch.Tensor,
    quantize: bool = False
):
    """Export to TorchScript format for mobile/edge deployment"""
    model.eval()

    try:
        # Try tracing first
        traced_model = torch.jit.trace(model, example_input)

        if quantize:
            # Dynamic quantization
            traced_model = torch.quantization.quantize_dynamic(
                traced_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        traced_model.save(output_path)
        LOGGER.info(f"TorchScript model saved: {output_path}")

        # Verify
        loaded = torch.jit.load(output_path)
        with torch.no_grad():
            original_output = model(example_input)
            loaded_output = loaded(example_input)
            diff = torch.abs(original_output - loaded_output).max().item()
            LOGGER.info(f"Verification - Max difference: {diff:.6f}")

        return True

    except Exception as e:
        LOGGER.warning(f"Tracing failed: {e}, trying scripting...")
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            LOGGER.info(f"TorchScript model saved (scripted): {output_path}")
            return True
        except Exception as e2:
            LOGGER.error(f"Scripting also failed: {e2}")
            return False


def export_onnx(
    model: CryTransformer,
    output_path: str,
    example_input: torch.Tensor,
    opset_version: int = 13,
    quantize: str = None
):
    """Export to ONNX format"""
    model.eval()

    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        LOGGER.info(f"ONNX model saved: {output_path}")

        # Verify with onnxruntime
        try:
            import onnxruntime as ort
            import onnx

            # Check model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            LOGGER.info("ONNX model validation passed")

            # Inference check
            ort_session = ort.InferenceSession(output_path)

            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare
            with torch.no_grad():
                torch_output = model(example_input).numpy()

            diff = np.abs(torch_output - ort_outputs[0]).max()
            LOGGER.info(f"ONNX verification - Max difference: {diff:.6f}")

        except ImportError:
            LOGGER.warning("onnx/onnxruntime not installed, skipping verification")

        # Quantization
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType

                quant_path = output_path.replace('.onnx', f'_{quantize}.onnx')
                quantize_dynamic(
                    model_input=output_path,
                    model_output=quant_path,
                    weight_type=QuantType.QInt8 if quantize == 'int8' else QuantType.QUInt8
                )
                LOGGER.info(f"Quantized ONNX model saved: {quant_path}")

                # Compare sizes
                import os
                orig_size = os.path.getsize(output_path) / (1024 * 1024)
                quant_size = os.path.getsize(quant_path) / (1024 * 1024)
                LOGGER.info(f"Model sizes - Original: {orig_size:.2f}MB, Quantized: {quant_size:.2f}MB")

            except ImportError:
                LOGGER.warning("onnxruntime quantization not available")

        return True

    except Exception as e:
        LOGGER.error(f"ONNX export failed: {e}")
        return False


def export_coreml(
    model: CryTransformer,
    output_path: str,
    example_input: torch.Tensor
):
    """Export to CoreML format for Apple devices"""
    try:
        import coremltools as ct

        # Trace model
        traced_model = torch.jit.trace(model, example_input)

        # Convert to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            classifier_config=None,
            compute_units=ct.ComputeUnit.ALL
        )

        # Add metadata
        mlmodel.author = "CryTransformer"
        mlmodel.license = "MIT"
        mlmodel.short_description = "Baby cry detection model"

        mlmodel.save(output_path)
        LOGGER.info(f"CoreML model saved: {output_path}")
        return True

    except ImportError:
        LOGGER.error("coremltools not installed. Install with: pip install coremltools")
        return False
    except Exception as e:
        LOGGER.error(f"CoreML export failed: {e}")
        return False


def export_tflite(
    model: CryTransformer,
    output_path: str,
    example_input: torch.Tensor,
    quantize: bool = False
):
    """Export to TensorFlow Lite format"""
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        import onnx

        # First export to ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')

        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )

        # Convert ONNX to TensorFlow
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_path = output_path.replace('.tflite', '_tf')
        tf_rep.export_graph(tf_path)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        LOGGER.info(f"TensorFlow Lite model saved: {output_path}")
        return True

    except ImportError as e:
        LOGGER.error(f"Required packages not installed: {e}")
        return False
    except Exception as e:
        LOGGER.error(f"TFLite export failed: {e}")
        return False


def export_size_report(checkpoint_path: str):
    """Print model size report"""
    import os

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    model = create_model(
        config=config.model,
        in_channels=config.feature.feature_dim,
        num_classes=config.model.num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    info = get_model_info(model)

    print("\n" + "=" * 60)
    print("Model Size Report")
    print("=" * 60)
    print(f"Variant:          {info['variant']}")
    print(f"Total Params:     {info['total_params']:,}")
    print(f"Trainable Params: {info['trainable_params']:,}")
    print(f"Model Size (fp32): {info['model_size_mb']:.2f} MB")
    print(f"Model Size (fp16): {info['model_size_mb'] / 2:.2f} MB")
    print(f"Model Size (int8): {info['model_size_mb'] / 4:.2f} MB (estimated)")
    print(f"MACs:             {info['estimated_macs_m']:.1f} M")
    print("=" * 60)

    # Checkpoint size
    ckpt_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Checkpoint size:  {ckpt_size:.2f} MB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Export CryTransformer model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--format', type=str, required=True,
                       choices=['pytorch', 'torchscript', 'onnx', 'coreml', 'tflite', 'all'],
                       help='Export format')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: auto-generated)')
    parser.add_argument('--quantize', type=str, default=None,
                       choices=['int8', 'uint8'],
                       help='Quantization type')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--seq_len', type=int, default=157,
                       help='Sequence length for export')
    parser.add_argument('--size_report', action='store_true',
                       help='Print model size report')
    args = parser.parse_args()

    # Load model
    LOGGER.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    model = create_model(
        config=config.model,
        in_channels=config.feature.feature_dim,
        num_classes=config.model.num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print_model_summary(model)

    if args.size_report:
        export_size_report(args.checkpoint)

    # Create example input - shape [B, T, F] (batch, time_frames, feature_dim)
    example_input = torch.randn(args.batch_size, args.seq_len, config.feature.feature_dim)
    LOGGER.info(f"Example input shape: {example_input.shape}")

    # Generate output path
    base_path = args.checkpoint.replace('.pt', '')

    # Export based on format
    formats = ['pytorch', 'torchscript', 'onnx', 'coreml', 'tflite'] if args.format == 'all' else [args.format]

    results = {}

    for fmt in formats:
        LOGGER.info(f"\nExporting to {fmt}...")

        if fmt == 'pytorch':
            output = args.output or f"{base_path}_pytorch.pt"
            export_pytorch(args.checkpoint, output)
            results[fmt] = output

        elif fmt == 'torchscript':
            output = args.output or f"{base_path}_torchscript.ptl"
            success = export_torchscript(model, output, example_input,
                                        quantize=args.quantize is not None)
            if success:
                results[fmt] = output

        elif fmt == 'onnx':
            output = args.output or f"{base_path}.onnx"
            success = export_onnx(model, output, example_input,
                                 quantize=args.quantize)
            if success:
                results[fmt] = output

        elif fmt == 'coreml':
            output = args.output or f"{base_path}.mlmodel"
            success = export_coreml(model, output, example_input)
            if success:
                results[fmt] = output

        elif fmt == 'tflite':
            output = args.output or f"{base_path}.tflite"
            success = export_tflite(model, output, example_input,
                                   quantize=args.quantize is not None)
            if success:
                results[fmt] = output

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for fmt, path in results.items():
        print(f"  {fmt:12s}: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
