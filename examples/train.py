"""
Example: Training script for baby cry detection
"""
import sys
sys.path.insert(0, '..')

import torch
from crydet import (
    Config, CryTransformer, CryTransformerLite,
    create_dataloaders
)


def train_with_config(config_path: str = 'configs/default.yaml'):
    """Train using config file"""
    # Load config
    config = Config.from_yaml(config_path)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_config=config.data,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        use_augmentation=config.train.use_augmentation
    )

    # Create model
    model = CryTransformer.from_config(
        config.model,
        feature_dim=config.data.feature.feature_dim,
        num_channels=config.data.feature.num_channels
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )

    # Training loop
    for epoch in range(config.train.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%')

    # Save model
    torch.save(model.state_dict(), 'checkpoints/model.pt')
    print("Training complete!")


if __name__ == '__main__':
    train_with_config()
