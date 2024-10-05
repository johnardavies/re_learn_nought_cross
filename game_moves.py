import numpy as np
import random
import torch
import logging
import math

from game_network import policy_net, device, ValueNet

import config as config


steps_done = 0


def select_random_zero_coordinate(array):
    """Function that selects a random zero coordinate from a 3x3 array"""
    if isinstance(array, torch.Tensor):
        array = array.squeeze(0)  # Unsqueeze to remove the extra empty dimension
        array = array.cpu().numpy()
    else:
        assert array.shape == (3, 3)  # "Input must be a 3x3 array"
    zero_coordinates = list(zip(*np.where(array == 0)))
    if zero_coordinates:
        #  return zero_coordinates
        return random.choice(zero_coordinates)
    else:
        return None


def generate_random_tuple():
    """Function that generates a random move represented as a tuple"""
    x = random.randint(0, 2)
    y = random.randint(0, 2)
    return (x, y)


def map_tensor_to_index(tensor):
    """Function that maps the output of the network back to a tuple"""
    tensor = tensor.cpu().squeeze(0)
    #   assert tensor.dim() == 1 and 0 <= tensor.item() <= 8, "Input must be a 1-dimensional tensor with values between 0 and 8"
    index = tensor.item()
    return index // 3, index % 3


def create_tensor_with_value_at_index(x):
    """Function that maps the tuples corresponding to a move into a 9 dimensional tensor"""
    index = x[0] * 3 + x[1]
    tensor_action = torch.zeros(9)
    tensor_action[index] = 1
    return tensor_action


def select_action(state, config):
    """Function that selects an action based on the state. Initially randomly, but later based on the maximum value from policy net"""
    global steps_done
    sample = random.random()
    # The eps threshold for using the network declines over time
    eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(
        -1.0 * steps_done / config.EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        # play a network move
        logging.info("using network")
#        tensor = torch.from_numpy(state)
        if (state == 0).all():
            state = state.to(device)
        else:
            state = state.long()

        # Convert the tensor to long integers
        with torch.no_grad():
            # Pick the action with the larger expected reward.
            return map_tensor_to_index(policy_net(state).max(1).indices.view(1, 1))
    else:  # Play a random move
        logging.info("using random")
        return generate_random_tuple()



def select_action_model(state):
    """Function that selects an action based on the state. It uses a trained model to make the move and makes a random legal move if the model makes an illegal proposal"""

    state_pos = (torch.tensor(state.board, dtype=torch.int8, device=device).unsqueeze(0).int())

    # Set up the network and load the past state
    model_play = ValueNet().to(device)

    model_state = torch.load("crosser_trained_2")

    # Loads the model
    model_play.load_state_dict(model_state["model_state_dict"])

    # Sets the model state to evaluate
    model_play.eval()

    #  tensor = torch.from_numpy(state)
    if (state_pos == 0).all():
        state_pos = state_pos.to(device)
    else:
        state_pos = state_pos.long()
    
    with torch.no_grad():

        proposed_move=map_tensor_to_index(model_play(state_pos).max(1).indices.view(1, 1))
        if proposed_move not in state.all_locs:
            return proposed_move
        else:
            print("illegal moves")
            return select_random_zero_coordinate(state.board)
