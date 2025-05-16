import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    confusion_matrix,
    auc
)
from data_loader import load_flood_data


def calculate_pos_weight(y_train):
    """
    Compute positive class weight for imbalanced data.

    Args:
        y_train (torch.Tensor): Training labels of shape (n_samples, 1)

    Returns:
        torch.Tensor: Weight tensor for positive class
    """
    # Count positive (flood) examples
    positives = y_train.sum().item()
    # Remaining examples are negative (no flood)
    negatives = y_train.numel() - positives
    # Avoid division by zero and return ratio as float tensor
    return torch.tensor([negatives / max(positives, 1)], dtype=torch.float)


class FloodNN(nn.Module):
    """
    Feedforward neural network for binary flood prediction.
    Architecture: Input -> Dense(32) -> BatchNorm -> ReLU -> Dropout -> Dense(16) -> BatchNorm -> ReLU -> Dropout -> Output(logit)
    """
    def __init__(self, input_dim: int = 7):
        super(FloodNN, self).__init__()
        # Define sequential layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),    # First hidden layer
            nn.BatchNorm1d(32),          # Normalize activations
            nn.ReLU(),                   # Activation
            nn.Dropout(0.2),             # Regularization
            nn.Linear(32, 16),           # Second hidden layer
            nn.BatchNorm1d(16),          # Normalize
            nn.ReLU(),                   # Activation
            nn.Dropout(0.2),             # Regularization
            nn.Linear(16, 1)             # Output layer: raw logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input features, shape (n_samples, input_dim)

        Returns:
            torch.Tensor: Raw logits, shape (n_samples,)
        """
        # Squeeze to remove last dimension after linear
        return self.model(x).squeeze(dim=-1)


def train_model():
    """
    Train and evaluate FloodNN model with early stopping and evaluation metrics.
    """
    # Load data arrays from custom loader
    X_train, y_train, X_test, y_test = load_flood_data()

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move tensors to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test   = X_test.to(device),  y_test.to(device)

    # Initialize model and move to device
    model = FloodNN(input_dim=X_train.shape[1]).to(device)
    # Compute positive class weight for loss
    pos_weight = calculate_pos_weight(y_train).to(device)
    # Use BCEWithLogitsLoss which combines sigmoid + weighted BCE
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Adam optimizer with default parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    epochs = 100
    best_loss = float('inf')
    patience = 10  # epochs to wait before early stopping
    trials = 0
    losses = []    # track training loss per epoch

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: compute logits and loss
        logits = model(X_train)
        # Loss expects logits and float labels
        loss = criterion(logits, y_train.float().squeeze())
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Record loss and implement early stopping
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            trials = 0
            # Save model checkpoint
            torch.save(model.state_dict(), 'best_floodnn.pth')
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Log every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    # Load the best model parameters
    model.load_state_dict(torch.load('best_floodnn.pth'))

    # Evaluate model performance
    model.eval()
    with torch.no_grad():
        # Compute logits on test set
        test_logits = model(X_test)
        # Convert logits to probabilities
        probs = torch.sigmoid(test_logits).cpu().numpy()
        # Apply decision threshold for binary predictions
        threshold = 0.65
        preds = (probs >= threshold).astype(int)
        y_true = y_test.cpu().numpy().astype(int)

    # Print accuracy
    acc = accuracy_score(y_true, preds)
    print(f"\nTest Accuracy: {acc:.2f}\n")

    # Detailed classification metrics
    print("Classification Report:")
    print(classification_report(y_true, preds, target_names=['No Flood', 'Flood']))

    # Confusion matrix computation and plotting
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Flood', 'Flood'])
    ax.set_yticklabels(['No Flood', 'Flood'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.title('Confusion Matrix')
    # Annotate counts
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    plt.colorbar(im)
    plt.savefig("confusion_matrix.png")  # Save figure

    # Precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC={pr_auc:.2f})')
    plt.savefig("precision_recall_curve.png")


if __name__ == "__main__":
    train_model()
