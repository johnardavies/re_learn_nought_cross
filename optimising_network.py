import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import logging

from config import *

# Importing the network used to play the game
from game_network import policy_net, target_net, device, save_checkpoint

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Sets up Transition that will be used to save the previous values
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Sets up a list to save the loss scores
loss_scores = []


def optimize_model(memory, BATCH_SIZE, target_net, policy_net, GAMMA, LR):
    """For the batch of returns, actions and values compute an estimated value function and see if it converges"""
    if len(memory) < BATCH_SIZE:
        logging.info(len(memory))
        return
    # Sample a batch_size of the memory
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # concatenate the batched state, action and rew
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_next_states = non_final_next_states.long()

    ## Compute the estimated value function results (at time t) for the state and corresponding actions in the batch (using policy_net) 
  
    # Transfers the action_batch information to the device where state_batch is
    action_batch = action_batch.to(state_batch.device)

    # Identify which action was taken in the batch
    action_batch = action_batch.max(1, keepdim=True)[1]

    # Get the estimated value function results from policy_net for the state_batch and corresponding action_batch actions
    state_action_values = policy_net(state_batch).gather(1, action_batch.long())

    ## Compute the reward and estimated value function results (at time t+1) using the reward_batch, the next_state and corresponding value function maximising actions (using target_net) 

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )

    next_state_values = next_state_values.unsqueeze(1)
   
    # Add the reward and the discounted next_state_values together to get the expected state_action_values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)

    # Compute Huber loss between the next_state_values and the expected_state_action_values
    criterion = nn.SmoothL1Loss()
  
    # Seeing how far apart the two state_action_values and the expected_state_action_values are
    loss = criterion(state_action_values, expected_state_action_values)

    # Append the loss to the loss_scores list
    loss_scores.append(loss)

    # Optimize the model minimising the loss between the state_action values and the expected state action values
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
