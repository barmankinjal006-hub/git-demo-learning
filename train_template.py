```python
"""
EcoSort Challenge — Model Training & Export
============================================
Usage:
    python train_template.py --epochs 30 --batch_size 32 --lr 0.001
"""

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

MAX_PARAMETERS = 350_000
MAX_MODEL_SIZE_MB = 1.50

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# 2. MODEL BUILDING BLOCKS
# ============================================================

class ConvBNAct(nn.Sequential):
    """Convolution + BatchNorm + optional ReLU."""

    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, groups=1,
        activation=True
    ):
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        ]

        if activation:
            layers.append(nn.ReLU6(inplace=True))

        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """
    Lightweight MobileNet-style block.

    Uses pointwise expansion, depthwise convolution,
    pointwise projection and an optional residual connection.
    """

    def __init__(
        self, in_channels, out_channels,
        stride=1, expansion=2
    ):
        super().__init__()

        hidden_channels = in_channels * expansion

        layers = []

        # Expand channels
        if expansion != 1:
            layers.append(
                ConvBNAct(
                    in_channels,
                    hidden_channels,
                    kernel_size=1
                )
            )

        # Depthwise convolution
        layers.append(
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels
            )
        )

        # Project to output channels
        layers.append(
            ConvBNAct(
                hidden_channels,
                out_channels,
                kernel_size=1,
                activation=False
            )
        )

        self.block = nn.Sequential(*layers)

        self.use_residual = (
            stride == 1 and in_channels == out_channels
        )

        if self.use_residual:
            self.residual_activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            out = out + x
            out = self.residual_activation(out)

        return out


# ============================================================
# 3. LIGHTWEIGHT CNN CLASSIFIER
# ============================================================

class ToyClassifier(nn.Module):
    """
    Lightweight replacement for the original toy network.

    Input:  (B, 3, 96, 96)
    Output: (B, 6)
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        self.features = nn.Sequential(
            # Initial feature extraction
            ConvBNAct(3, 24, kernel_size=3, stride=1),

            # Stage 1
            InvertedResidual(24, 24, stride=1),
            InvertedResidual(24, 24, stride=1),

            # Stage 2
            InvertedResidual(24, 32, stride=2),
            InvertedResidual(32, 32, stride=1),

            # Stage 3
            InvertedResidual(32, 48, stride=2),
            InvertedResidual(48, 48, stride=1),

            # Stage 4
            InvertedResidual(48, 64, stride=2),
            InvertedResidual(64, 64, stride=1),

            # Stage 5
            InvertedResidual(64, 96, stride=2),
            InvertedResidual(96, 96, stride=1),

            # Stage 6
            InvertedResidual(96, 128, stride=2),
            InvertedResidual(128, 128, stride=1),

            # Final feature processing
            ConvBNAct(128, 128, kernel_size=1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# 4. COUNT TRAINABLE PARAMETERS
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# 5. LOAD DATASET
# ============================================================

def get_dataloaders(data_dir: str, batch_size: int = 32):

    data_dir = Path(data_dir).resolve()

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training dataset not found: {train_dir}"
        )

    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Validation dataset not found: {val_dir}"
        )

    # Augmentation for training images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            96,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Deterministic preprocessing for validation
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transform=val_transform
    )

    if train_dataset.classes != CLASSES:
        raise ValueError(
            f"Expected classes: {CLASSES}\n"
            f"Found classes: {train_dataset.classes}"
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Training and validation class mappings differ."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    return train_loader, val_loader


# ============================================================
# 6. TRAIN AND VALIDATE
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    patience=10
):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01
    )

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    model_path = PROJECT_DIR / "best_model.pth"

    for epoch in range(1, epochs + 1):

        # ---------------- TRAINING ----------------
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = 100.0 * train_correct / train_total

        # ---------------- VALIDATION ----------------
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save the best validation checkpoint
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0

            best_state = copy.deepcopy(model.state_dict())

            torch.save(best_state, model_path)

            print(f"Best checkpoint saved: {model_path}")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    return model


# ============================================================
# 7. EXPORT TO ONNX
# ============================================================

def export_to_onnx(
    model: nn.Module,
    output_path: str = "model.onnx"
):
    model = model.cpu().eval()

    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 96, 96)

    print(f"Exporting model to: {output_path}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            dynamo=False
        )

    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"ONNX file created: {output_path}")
    print(f"File size: {size_mb:.3f} MiB")

    if size_mb > MAX_MODEL_SIZE_MB:
        print("WARNING: Model exceeds the file-size limit.")
    else:
        print("File-size limit satisfied.")

    # Verify the exported graph if ONNX Runtime is installed
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"]
        )

        result = session.run(
            ["output"],
            {"input": np.zeros((1, 3, 96, 96), dtype=np.float32)}
        )[0]

        print(f"ONNX test output shape: {result.shape}")

        if result.shape != (1, len(CLASSES)):
            raise RuntimeError(
                f"Unexpected output shape: {result.shape}"
            )

        print("ONNX inference test passed.")

    except ImportError:
        print(
            "ONNX Runtime is not installed; "
            "skipping the inference test."
        )


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="EcoSort Lightweight CNN Training"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data" / "public"),
        help="Directory containing train and val folders"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output ONNX filename"
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch_size and lr must be positive.")

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("EcoSort Edge-AI Training")
    print("=" * 60)

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Dataset directory: {Path(args.data_dir).resolve()}")
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )

    model = ToyClassifier(
        num_classes=len(CLASSES)
    ).to(device)

    total_params = count_parameters(model)

    print(f"Trainable parameters: {total_params:,}")
    print(f"Maximum allowed: {MAX_PARAMETERS:,}")

    if total_params > MAX_PARAMETERS:
        raise RuntimeError(
            "Model exceeds the parameter limit. "
            "Reduce the model size before training."
        )

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr
    )

    # Export the best validation checkpoint
    export_to_onnx(
        model,
        output_path=args.output
    )

    print("\nTraining and export complete.")
    print("Next, run the challenge verification tool:")
    print(
        f"python verify_submission.py "
        f"--model {args.output} "
        f"--val_dir {args.data_dir}/val"
    )


if __name__ == "__main__":
    main()
```
