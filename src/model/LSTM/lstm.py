import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from scipy.special import expit  # for applying sigmoid to logits
from data_loader import get_dataloaders


class FloodLSTM(nn.Module):
    """
    LSTM-based neural network for flood prediction.
    Processes sequential input data and outputs a raw logit for binary classification.
    """
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        # LSTM layer: takes input sequences of shape (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Classifier: two-layer feedforward network to map final hidden state to a single logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),  # reduce dimension
            nn.ReLU(),                   # non-linear activation
            nn.Linear(32, 1)             # output raw logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: torch.Tensor of shape (batch, seq_len, input_dim)

        Returns:
            torch.Tensor of shape (batch,), raw logits for each example
        """
        # Pass input through LSTM; ignore hidden state outputs
        output_seq, _ = self.lstm(x)
        # Extract the last time-step output: shape (batch, hidden_dim)
        last_step = output_seq[:, -1, :]
        # Pass through classifier to get raw logits, then squeeze to (batch,)
        logits = self.classifier(last_step).squeeze()
        return logits


def calculate_pos_weight(train_loader, device) -> torch.Tensor:
    """
    Compute the positive class weight for balancing the BCEWithLogitsLoss.

    Args:
        train_loader: DataLoader yielding (inputs, labels) for training
        device: torch.device where the tensor should be allocated

    Returns:
        torch.Tensor containing the ratio (negatives / positives)
    """
    total_samples = 0
    positive_count = 0
    # Count total and positive labels across batches
    for _, labels in train_loader:
        total_samples += labels.numel()
        positive_count += labels.sum().item()
    negative_count = total_samples - positive_count
    # Avoid division by zero
    pos_weight = negative_count / max(positive_count, 1)
    return torch.tensor([pos_weight], device=device)


def train_and_evaluate(
    csv_path: str = "data/final/combined-data.csv",
    seq_len: int = 12,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    device=None
):
    """
    Train the FloodLSTM model and evaluate its performance with plots.

    Args:
        csv_path: Path to the CSV data file
        seq_len: Number of time steps in each input sequence
        batch_size: Number of samples per training batch
        epochs: Number of training epochs
        lr: Learning rate for the Adam optimizer
        device: torch device ('cpu' or 'cuda'); auto-detected if None
    """
    # Determine computing device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and testing DataLoaders
    train_loader, test_loader = get_dataloaders(csv_path, seq_len, batch_size)

    # Instantiate model and move to device
    model = FloodLSTM().to(device)

    # Compute class weight and define loss with BCEWithLogits
    pos_weight = calculate_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Adam optimizer for model parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device).squeeze().float()

            # Zero gradients, forward pass, compute loss, backpropagate, and update
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Compute average loss for the epoch
        avg_loss = running_loss / len(train_loader.dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    # Switch to evaluation mode
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            batch_logits = model(inputs).cpu().numpy()
            all_logits.append(batch_logits)
            all_labels.append(labels.numpy())

    # Combine batch-wise results
    logits = np.concatenate(all_logits).flatten()
    labels = np.concatenate(all_labels).flatten().astype(int)

    # Convert logits to probabilities
    probabilities = expit(logits)

    # Define decision threshold for classification
    threshold = 0.6
    predictions = (probabilities >= threshold).astype(int)

    # Calculate basic metrics
    accuracy = accuracy_score(labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.3f}\n")

    # Display class distribution
    unique_vals, counts = np.unique(labels, return_counts=True)
    print("Test class distribution:", dict(zip(unique_vals, counts)))

    # Detailed classification report
    print(classification_report(labels, predictions, target_names=["No Flood", "Flood"]))

    # Precision-Recall AUC
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(labels, probabilities)
    pr_auc_value = auc(recall_vals, precision_vals)
    print(f"PR AUC: {pr_auc_value:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Flood", "Flood"])
    ax.set_yticklabels(["No Flood", "Flood"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    plt.title("Confusion Matrix")
    plt.colorbar(im)
    plt.savefig("confusion.png")  # save plot to file

    # Plot Recall vs. Threshold curve
    fig, ax = plt.subplots()
    ax.plot(pr_thresholds, recall_vals[:-1], marker='.')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Recall")
    plt.title("Recall vs. Threshold")
    plt.savefig("recall_vs_threshold.png")  # save plot to file


if __name__ == "__main__":
    train_and_evaluate()
