import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

import config as config

# Importing the game board and functions to calculate rewardss
from game import GameBoard, PersonScores, calculate_rewards, contains_three

# Importing the network used to play the game
from game_network import policy_net, target_net, device, save_checkpoint

# Importing the functions used to create game moves
from game_moves import (
    select_random_zero_coordinate,
    select_action,
    create_tensor_with_value_at_index,
    select_action_model,
)

# Importing the network optimisation functions
from optimising_network import optimize_model, ReplayMemory, loss_scores, optimizer

# Importing charting functions
from chart_funcs import plot_wins, plot_wins_stacked, plot_errors, cumulative_wins


file_handler = logging.FileHandler(filename="re_learn4.log")

# Sets up a logger to record the games
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler],
)

# Sets up the replay memory
memory = ReplayMemory(20000)


# If cuda is available use more games for training
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_games = 5000
else:
    num_games = 50


# Sets up a list to register the game outcome which has entries:
# 1. If Player 1 wins, 2. If Player 1 makes an illegal move and loses
# 3. If the game is a draw, 4. If Player 2 wins
game_outcome = []


def main(config):
    # Start the training loop
    for i_game in range(num_games):
        if i_game % 100 == 0:
            print(i_game)
        # Create a new game
        game = GameBoard()

        logging.info("Game starts")
        for i in range(1, 10):

            logging.info(game.board)
            if i % 2 != 0:
                # Player 1's turn
                state = game.board
                state = (
                    torch.tensor(game.board, dtype=torch.int8, device=device)
                    .unsqueeze(0)
                    .int()
                )
                logging.info("Player 1 moves")

                # Get the Player 1 action given the state
                player_1_action = select_action_model(game)

                #        select_action(state, config)

                logging.info(player_1_action)

                # Converts the Player 1 action to a tensor so that it can be fed into the network
                player_1_action_tensor = torch.tensor(
                    create_tensor_with_value_at_index(player_1_action),
                    dtype=torch.int8,
                    device=device,
                ).unsqueeze(0)

                # If player 1 makes an illegal move end the game
                if player_1_action in game.all_locs:

                    logging.info("Player 1 makes an illegal move")
                    reward = torch.tensor(
                        [config.reward_scores["illegal_move_loss"]], device=device
                    ).unsqueeze(0)
                    logging.info(f" Reward {reward}")
                    next_state = None
                    game_outcome.append(2)

                    memory.push(state, player_1_action_tensor, next_state, reward)
                    # End the game. There is no next state and the reward is the losing reward
                    break
                else:
                    # Player 1 makes the move and the game updates
                    game.update(player_1_action, "player_1")

                    # If after Player 1 has moved there is a three, Player 1 wins and game ends
                    if contains_three(PersonScores(game.board).all_scores) == True:
                        logging.info("Player 1 wins")

                        # Append a 1 to the game_outcome list indicating a Player 1 win
                        game_outcome.append(1)
                        logging.info(game.board)

                        # Games ends so the next state is the board at the end of the game
                        next_state = (
                            torch.tensor(game.board, dtype=torch.int8, device=device)
                            .unsqueeze(0)
                            .int()
                        )
                        memory.push(state, player_1_action_tensor, next_state, reward)
                        logging.info("Game ends")
                        break
                    # If 9 moves have been played and still no winnr, the game is a draw
                    elif (
                        i == 9
                        and contains_three(PersonScores(game.board).all_scores) == False
                    ):
                        # Append a 3 to the game_outcone list indicating a draw
                        game_outcome.append(3)
                        reward = calculate_rewards(
                            PersonScores(game.board).all_scores, "player_1"
                        )
                        next_state = (
                            torch.tensor(game.board, dtype=torch.int8, device=device)
                            .unsqueeze(0)
                            .int()
                        )
                        reward = torch.tensor([reward], device=device).unsqueeze(0)
                        logging.info(f"Reward {reward}")
                        memory.push(state, player_1_action_tensor, next_state, reward)
                        logging.info("Last move - game is drawn")
                        break

            elif i % 2 == 0:

                # Player 1 did not win the last move and so it is Player 2's turn
                logging.info("Player 2 moves")

                # Player chooses a random move
                player_2_action = select_random_zero_coordinate(game.board)
                # select_action_model(game)

                # Update the game's board
                game.update(player_2_action, "player_2")

                # Convert the board to a tensor to feed it into the network
                next_state = (
                    torch.tensor(game.board, dtype=torch.int8, device=device)
                    .unsqueeze(0)
                    .int()
                )
                # Checks if Player 2 won after the move
                if contains_three(PersonScores(game.board).all_scores) == True:
                    logging.info("Player 2 moves and wins")
                    # Append a 4 to the game_outcome list indicating a Player 2 win
                    game_outcome.append(4)
                    reward = torch.tensor(
                        [config.reward_scores["loss"]], device=device
                    ).unsqueeze(0)
                    logging.info(f"Reward {reward}")
                    logging.info(game.board)
                    memory.push(state, player_1_action_tensor, next_state, reward)
                    break

                # Player 2 did not win the last move and we calculate Player 1's payoff
                elif contains_three(PersonScores(game.board).all_scores) == False:
                    # Player 1 made a legal move and the game is still in play. This should be the most common scenario
                    logging.info(
                        "Player 1 made legal move, player 2 has moved, but not won"
                    )
                    reward = calculate_rewards(
                        PersonScores(game.board).all_scores, "player_1"
                    )
                    reward = torch.tensor([reward], device=device).unsqueeze(0)
                    logging.info(f" Reward {reward}")
                    memory.push(state, player_1_action_tensor, next_state, reward)

            # Perform one step of the optimization (on the policy network) after player 2 moves
            optimize_model(
                memory,
                config.BATCH_SIZE,
                target_net,
                policy_net,
                config.GAMMA,
                config.LR,
            )

            # Soft update of the target network's weights. New weights are mostly taken from  target_net_state_dict
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * config.TAU + target_net_state_dict[key] * (1 - config.TAU)
                target_net.load_state_dict(target_net_state_dict)

    # Save the model at the end of the training
    save_checkpoint(target_net, optimizer, "crosser_trained_new2")

    # Generate the plot for the number of errors
    plot_errors(loss_scores)

    # Generate the plot for the number of wins
    plot_wins(cumulative_wins(game_outcome))

    # Plot a stacked bar chart of how the games are going
    plot_wins_stacked(cumulative_wins(game_outcome))

    print("Training complete")


if __name__ == "__main__":
    main(config)
