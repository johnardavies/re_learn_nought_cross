import numpy as np
import config as config

class GameBoard:
    """Class that keeps track of board positions"""

    def __init__(self):
        self.board = np.zeros((3, 3))
        self.person_1_locs = []
        self.person_2_locs = []
        self.all_locs = []

    def update(self, x, player):
        assert (x[0] < 3) and (x[1] < 3)
        if player == "player_1":
            self.board[x[0], x[1]] = 1
        if player == "player_2":
            self.board[x[0], x[1]] = 2
        self.person_1_locs = list(
            zip(np.where(self.board == 1)[0], np.where(self.board == 1)[1])
        )
        self.person_2_locs = list(
            zip(np.where(self.board == 2)[0], np.where(self.board == 2)[1])
        )
        self.all_locs = self.person_1_locs + self.person_2_locs


class PersonScores:
    """Class that keeps track of the scores of the players"""

    def __init__(self, game_board):

        self.vertical_scores = list(
            zip(
                np.count_nonzero(game_board == 1, axis=0),
                np.count_nonzero(game_board == 2, axis=0),
            )
        )
        self.horizontal_scores = list(
            zip(
                np.count_nonzero(game_board == 1, axis=1),
                np.count_nonzero(game_board == 2, axis=1),
            )
        )
        self.diagonal_scores1 = [
            (
                np.count_nonzero(np.diag(game_board) == 1),
                np.count_nonzero(np.diag(game_board) == 2),
            )
        ]
        self.diagonal_scores2 = [
            (
                np.count_nonzero(np.diag(np.fliplr(game_board)) == 1),
                np.count_nonzero(np.diag(np.fliplr(game_board)) == 2),
            )
        ]
        self.all_scores = (
            self.vertical_scores
            + self.horizontal_scores
            + self.diagonal_scores1
            + self.diagonal_scores2
        )


def assign_scores(values, config):
    """Function that assigns rewards to the players based on the highest score"""
    if 3 in values:
        score = config.reward_scores["three_score"]
    elif 2 in values:
        score = config.reward_scores["two_score"]
    elif 1 in values:
        score = config.reward_scores["one_score"]
    return score

def calculate_scores(scores_metrics, player):
    """Function that calculates the rewards for the players given a proposed move and the state of the board"""

    # Get the scores for player 1 and player 2
    player_1_scores = [t[0] for t in scores_metrics]
    player_2_scores = [t[1] for t in scores_metrics]

    if player == "player_1":
        reward = assign_scores(player_1_scores, config)
    if player == "player_2":
        reward = assign_scores(player_2_scores, config)

    return reward


def contains_three(tuples):
    """Function that checks if a tuple contains a 3"""
    return any(3 in t for t in tuples)


