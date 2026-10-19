# EcoSort Edge-AI — Waste Classification

## 1. Project Overview
EcoSort is an AI-based waste classification project that classifies waste images into six categories using a lightweight CNN model.

**Classes:**
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## 2. Project Structure

```text
EcoSort/
├── starter_kit/
│   └── download_data.py
├── data/
│   └── public/
│       ├── train/
│       └── val/
├── train_template.py
├── verify_submission.py
├── best_model.pth
├── model.onnx
└── README.md
```

## 3. Requirements

Install the required libraries:

```bash
pip install torch torchvision numpy pillow onnx onnxruntime
```

## 4. Download the Dataset

Run the following command from the project root:

```bash
python starter_kit/download_data.py
```

Make sure the training and validation folders are available inside `data/public/`.

## 5. Train the Model

Run:

```bash
python train_template.py --epochs 30 --batch_size 32 --lr 0.001
```

The model trains on the dataset, checks validation accuracy, and saves the best model as `best_model.pth`. It then exports the trained model to `model.onnx`.

## 6. Verify the Model

Run:

```bash
python verify_submission.py --model model.onnx --val_dir data/public/val
```

This checks the model’s size, parameter count, input/output format, inference speed, and validation accuracy when the required dependencies and dataset are available.

## 7. Challenge Requirements

- **Input:** `(Batch, 3, 96, 96)`
- **Output:** `(Batch, 6)`
- **Maximum parameters:** 350,000
- **Maximum ONNX size:** 1.50 MB
- **Target validation accuracy:** At least 75%
- **Target CPU latency:** At most 35 ms per image

## 8. Expected Output

The main output file for submission is:

```text
model.onnx
```

Check the verification results before submitting. Meeting the accuracy and performance targets depends on training and must be confirmed by testing.
