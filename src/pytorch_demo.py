"""Minimal PyTorch model used for discovery testing."""

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

# Run one inference so PyTorch is actually imported and used.
sample = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
prediction = model(sample)

print("PyTorch:", torch.__version__)
print("Prediction:", prediction.detach().numpy())
