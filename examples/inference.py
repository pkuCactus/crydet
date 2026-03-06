"""
Example: Inference script for baby cry detection
Uses soundfile for audio loading (portable to edge devices)
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import soundfile as sf
from crydet import FeatureExtractor, CryTransformer, FeatureConfig, ModelConfig


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio file"""
    waveform, sample_rate = sf.read(file_path)

    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample if needed
    if sample_rate != target_sr:
        duration = len(waveform) / sample_rate
        new_length = int(duration * target_sr)
        orig_indices = np.arange(len(waveform))
        new_indices = np.linspace(0, len(waveform) - 1, new_length)
        waveform = np.interp(new_indices, orig_indices, waveform)

    return waveform.astype(np.float32)


def load_model(checkpoint_path, feature_dim, num_channels, device='cuda'):
    """Load trained Transformer model"""
    config = ModelConfig()
    model = CryTransformer.from_config(
        config,
        feature_dim=feature_dim,
        num_channels=num_channels
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(audio_path, model, feature_extractor, max_frames=1000, device='cuda'):
    """Predict if audio contains baby cry"""
    # Load audio using soundfile
    waveform = load_audio(audio_path, target_sr=feature_extractor.sample_rate)

    # Extract features (returns shape: num_channels, feature_dim, time)
    features = feature_extractor.extract(waveform)

    # Pad/truncate to fixed length
    num_channels, feature_dim, num_frames = features.shape
    if num_frames < max_frames:
        padding = np.zeros((num_channels, feature_dim, max_frames - num_frames), dtype=features.dtype)
        features = np.concatenate([features, padding], axis=2)
    elif num_frames > max_frames:
        features = features[:, :, :max_frames]

    # Convert to tensor and add batch dimension
    features = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()

    return pred_class, confidence


if __name__ == '__main__':
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup feature extractor with delta features
    feature_config = FeatureConfig(
        feature_type='mfcc',
        n_mfcc=40,
        sample_rate=16000,
        use_delta=True,
        use_freq_delta=True
    )

    feature_extractor = FeatureExtractor(feature_config)

    # Load model
    model = load_model(
        'checkpoints/best.pt',
        feature_dim=feature_config.feature_dim,
        num_channels=feature_config.num_channels,
        device=device
    )

    # Predict
    audio_file = 'test_audio.wav'
    pred_class, confidence = predict(audio_file, model, feature_extractor, device=device)

    label = "Cry" if pred_class == 1 else "No Cry"
    print(f"Prediction: {label} (confidence: {confidence:.2%})")
