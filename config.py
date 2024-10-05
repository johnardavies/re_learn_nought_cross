import torch

# The parameters

# The discount factor
GAMMA = 0.9
# The batch size for the replay memory
BATCH_SIZE = 100
# The weight for the soft update of the target network
TAU = 0.05
# The learning rate for the optimizer
LR = 1e-4
# The EPS parameters which specify the transition from random play to using the network
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# Reward values
reward_scores = {
    "three_score": 10000,
    "two_score": 2000,
    "one_score": 10,
    "loss": -4000,
    "illegal_move_loss": -8000
}

# Specifies the device to use
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
