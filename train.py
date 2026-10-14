import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
OUTPUT_DIR = "outputs"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # 26x26 -> 13x13
            nn.Conv2d(32, 64, kernel_size=3),  # 13x13 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # 11x11 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    transform = transforms.ToTensor()
    full_train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MNISTCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

    test_loss, test_accuracy = run_epoch(model, test_loader, criterion)
    results = {
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mnist_cnn.pth"))

    epochs = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_accuracy"], label="Training")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Training")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

    model.eval()
    sample_images, sample_labels = next(iter(DataLoader(test_dataset, batch_size=10, shuffle=False)))
    with torch.no_grad():
        sample_logits = model(sample_images.to(DEVICE))
        probabilities = torch.softmax(sample_logits, dim=1).cpu()
        predictions = sample_logits.argmax(dim=1).cpu()

    sample_rows = []
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        image = sample_images[i].squeeze(0).numpy()
        actual = int(sample_labels[i])
        predicted = int(predictions[i])
        confidence = float(probabilities[i].max())
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Actual: {actual} | Pred: {predicted}")
        ax.axis("off")
        sample_rows.append({
            "sample": i,
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence,
        })
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUTPUT_DIR, "sample_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(sample_rows, f, indent=2)

    print("\nTest results")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Saved model, metrics, plots, and predictions in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
