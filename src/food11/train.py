"""
Train a ResNet18 model on the Food-11 dataset, with MLflow tracking.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""

import argparse

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["mini", "processed"],
    default="mini",
    help="Which dataset folder to use: 'mini' (small, fast) or 'processed' (full size).",
)
parser.add_argument("--epochs", type=int, default=5, help="How many full passes over the training data.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
parser.add_argument("--batch-size", type=int, default=32, help="How many images per training step.")
args = parser.parse_args()

dataset_folder = "food11_processed_mini" if args.dataset == "mini" else "food11_processed"
data_root = f"./data/{dataset_folder}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.ImageFolder(f"{data_root}/training", transform=transform)
val_dataset = datasets.ImageFolder(f"{data_root}/validation", transform=transform)
test_dataset = datasets.ImageFolder(f"{data_root}/evaluation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Classes ({num_classes}): {train_dataset.classes}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")

with mlflow.start_run():
    mlflow.log_params(
        {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
        }
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_total = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_total += images.size(0)

        train_loss = running_loss / running_total
        val_loss, val_accuracy = evaluate(val_loader)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

    test_loss, test_accuracy = evaluate(test_loader)
    print(f"Final test_accuracy: {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)

    example_images, _ = next(iter(train_loader))
    input_example = example_images[:1].numpy()
    mlflow.pytorch.log_model(model, "model", input_example=input_example)

print("Training complete. Check the MLflow UI at http://127.0.0.1:5000")
