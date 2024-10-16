import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# Define the network


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        # Embedding layer: 3 possible values, embedding size 100
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=100)
        # Example first layer after embedding
        # Flattening the 3x3 grid (which after embedding will be 3x3x4) to a vector of size 3*3*4 = 36
        self.fc1 = nn.Linear(3 * 3 * 100, 75)
        self.fc2 = nn.Linear(75, 75)  # Fully connected layer
        self.fc3 = nn.Linear(75, 9)

    def forward(self, x):
        # Assuming x is a 3x3 tensor with values 0, 1, or 2
        x = self.embedding(x)  # Shape: (3, 3, 4)
        x = x.view(-1, 3 * 3 * 100)  # Flatten to (1, 36) if batch size is 1
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Set up the networks on the device
policy_net = ValueNet().to(device)
target_net = ValueNet().to(device)

#model_state = torch.load("crosser_trained")

# Loads the model
#policy_net.load_state_dict(model_state["model_state_dict"])
# Copy the weights of the policy_net to the target_net
#target_net.load_state_dict(policy_net.state_dict())


def save_checkpoint(model, optimizer, save_path):
    """function to save checkpoints on the model weights, the optimiser state and epoch"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
