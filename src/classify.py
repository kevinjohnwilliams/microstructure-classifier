"""
Phase classification model for steel microstructures.

Phase 2 of the project: fine-tune a CNN to classify individual phases
(ferrite, pearlite, martensite, bainite, austenite, cementite).

This module provides:
- A transfer-learning classifier built on ResNet50
- Training loop with standard augmentation
- Inference for single images

NOTE: This requires labeled training data organized as:
    data/processed/train/
        ferrite/
        pearlite/
        martensite/
        ...
    data/processed/val/
        ferrite/
        ...
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import get_transform


class PhaseClassifier(nn.Module):
    """
    Transfer-learning phase classifier.
    
    Replaces ResNet50's final FC layer with a new head
    for microstructure phase classification.
    """
    
    def __init__(self, num_classes: int = None, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes or config.NUM_CLASSES
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Optionally freeze the backbone (train only the new head)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the classification head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def predict(self, image_path: str | Path, device: str = None) -> dict:
        """
        Predict phase for a single image.
        
        Returns:
            dict with 'predicted_phase', 'confidence', and 'probabilities'
        """
        from src.utils import load_and_preprocess
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        tensor = load_and_preprocess(image_path).to(device)
        
        with torch.no_grad():
            logits = self(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
        
        predicted_idx = probs.argmax().item()
        
        return {
            "predicted_phase": config.PHASE_CLASSES[predicted_idx],
            "confidence": probs[predicted_idx].item(),
            "probabilities": {
                phase: probs[i].item()
                for i, phase in enumerate(config.PHASE_CLASSES)
            },
        }


def train_classifier(
    train_dir: str | Path,
    val_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str | Path = None,
):
    """
    Train the phase classifier on labeled data.
    
    Expects data organized in ImageFolder format:
        train_dir/class_name/image.png
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    # Datasets
    train_dataset = ImageFolder(str(train_dir), transform=get_transform(augment=True))
    val_dataset = ImageFolder(str(val_dir), transform=get_transform(augment=False))
    
    print(f"Training classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Model
    model = PhaseClassifier(num_classes=len(train_dataset.classes))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        scheduler.step(val_loss)
        
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_to = save_path or config.DATA_DIR / "models" / "phase_classifier.pt"
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_to)
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")
    
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    return model
