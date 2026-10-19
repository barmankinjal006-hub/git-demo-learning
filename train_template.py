```python
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

MAX_PARAMETERS = 350_000
MAX_MODEL_SIZE_MB = 1.50

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------
# Convolution Block
# --------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        activation=True,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
            if activation
            else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------
# Lightweight Inverted Residual Block
# --------------------------------------------------

class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=2,
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
                    kernel_size=1,
                )
            )

        # Depthwise convolution
        layers.append(
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            )
        )

        # Project channels
        layers.append(
            ConvBNAct(
                hidden_channels,
                out_channels,
                kernel_size=1,
                activation=False,
            )
        )

        self.block = nn.Sequential(*layers)

        self.use_residual = (
            stride == 1 and in_channels == out_channels
        )

    def forward(self, x):
        result = self.block(x)

        if self.use_residual:
            result = result + x

        return result


# --------------------------------------------------
# EcoSort CNN Model
# --------------------------------------------------

class ToyClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNAct(3, 24, stride=2),

            InvertedResidual(24, 24),
            InvertedResidual(24, 24),

            InvertedResidual(24, 32, stride=2),
            InvertedResidual(32, 32),

            InvertedResidual(32, 48, stride=2),
            InvertedResidual(48, 48),

            InvertedResidual(48, 64, stride=2),
            InvertedResidual(64, 64),

            InvertedResidual(64, 96, stride=2),
            InvertedResidual(96, 96),

            InvertedResidual(96, 128, stride=2),
            InvertedResidual(128, 128),

            ConvBNAct(128, 128, kernel_size=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# --------------------------------------------------
# Count Parameters
# --------------------------------------------------

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def get_dataloaders(data_dir, batch_size=32):
    data_dir = Path(data_dir).expanduser().resolve()

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}"
        )

    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Validation folder not found: {val_dir}"
        )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            96,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform=train_transforms,
    )

    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transform=val_transforms,
    )

    if train_dataset.classes != CLASSES:
        raise ValueError(
            f"Unexpected training classes: {train_dataset.classes}\n"
            f"Expected classes: {CLASSES}"
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Training and validation class mappings do not match."
        )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, val_loader


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return average_loss, accuracy


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    lr=0.001,
):
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=lr * 0.01,
    )

    best_accuracy = -1.0
    patience = 10
    epochs_without_improvement = 0

    best_model_path = PROJECT_DIR / "best_model.pth"

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_accuracy = 100.0 * correct / max(total, 1)

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation accuracy has not improved.")
            break

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    )

    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    return model


# --------------------------------------------------
# Export Model to ONNX
# --------------------------------------------------

def export_to_onnx(model, output_path):
    output_path = Path(output_path).expanduser()

    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path

    output_path = output_path.resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 96, 96)

    export_args = {
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        "opset_version": 17,
    }

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            dynamo=False,
            **export_args,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            **export_args,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"ONNX model saved to: {output_path}")
    print(f"ONNX model size: {size_mb:.3f} MB")

    if size_mb > MAX_MODEL_SIZE_MB:
        print("WARNING: ONNX model exceeds the 1.50 MB limit.")

    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )

        result = session.run(
            ["output"],
            {"input": dummy_input.numpy()},
        )[0]

        print(f"ONNX output shape: {result.shape}")

        if result.shape != (1, len(CLASSES)):
            raise RuntimeError(
                f"Unexpected ONNX output shape: {result.shape}"
            )

    except ImportError:
        print(
            "ONNX validation skipped. Install onnx and "
            "onnxruntime to validate the exported model."
        )


# --------------------------------------------------
# Main Function
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and export the EcoSort CNN model."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_DIR / "data" / "public"),
        help="Path containing train/ and val/ folders.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output ONNX file path.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    if args.lr <= 0:
        parser.error("--lr must be greater than 0.")

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        args.batch_size,
    )

    model = ToyClassifier(
        num_classes=len(CLASSES)
    ).to(device)

    total_parameters = count_parameters(model)

    print(f"Model parameters: {total_parameters:,}")

    if total_parameters > MAX_PARAMETERS:
        raise RuntimeError(
            f"Model has {total_parameters:,} parameters; "
            f"the limit is {MAX_PARAMETERS:,}."
        )

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
    )

    export_to_onnx(
        model,
        args.output,
    )

    print("\nTraining and export completed.")
    print("Next, verify the model using verify_submission.py.")


if __name__ == "__main__":
    main()
```
