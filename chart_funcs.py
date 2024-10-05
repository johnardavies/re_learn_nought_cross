import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns



print(matplotlib.style.available)

def cumulative_wins(outcomes_list):
    """Function to calculate the cumulative wins/losses/draws over time"""
    wins = [0]
    losses = [0]
    illegal_moves_loss = [0]
    draws = [0]
    benchmark = [0]
    for i, outcome in enumerate(outcomes_list):
        benchmark.append(i * 0.5)
        if outcome == 1:
            # Player 1 plays and wins
            wins.append(wins[-1] + 1)
            losses.append(losses[-1])
            illegal_moves_loss.append(illegal_moves_loss[-1])
            draws.append(draws[-1])
        elif outcome == 2:
            # Player 1 makes and illegal moves and losses
            wins.append(wins[-1])
            losses.append(losses[-1])
            illegal_moves_loss.append(illegal_moves_loss[-1] + 1)
            draws.append(draws[-1])
        elif outcome == 3:
            # Player 1 plays and draws
            wins.append(wins[-1])
            losses.append(losses[-1])
            illegal_moves_loss.append(illegal_moves_loss[-1])
            draws.append(draws[-1] + 1)
        elif outcome == 4:
            # Player 2 plays and wins (Player 1 loses)
            wins.append(wins[-1])
            losses.append(losses[-1] + 1)
            illegal_moves_loss.append(illegal_moves_loss[-1])
            draws.append(draws[-1])

    return wins, losses, illegal_moves_loss, draws, benchmark


def plot_wins(results):
    """Function to plot the cumulative wins/losses/draws over time"""
    plt.figure(figsize=(12, 8))
    plt.style.use("seaborn-muted")

    plt.ylim(0, 4000)
    plt.xlim([0, 4000])
    wins, losses, illegal_moves, draws, benchmark = results
    plt.plot(wins, label="Player 1 Wins", color="gold")
    plt.plot(losses, label="Player 2 Wins", color="blue")
    plt.plot(illegal_moves, label="Player 1 Illegal Move loss", color="red")
    plt.plot(draws, label="Draws", color="brown")
    plt.plot(benchmark, label="Benchmark of winning 50%", color="black")

    # Insert a line where we start using the batched information
    plt.vlines(x=128, ymin=0, ymax=4000, linestyle="dashed")

    plt.xlabel("Game Number", fontsize=15)
    plt.ylabel("Count of how the game ends", fontsize=15)
    plt.title("How the game ends over time", fontsize=18, weight="bold")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.legend()
    plt.savefig("winning_count_40002.png")


def plot_errors(losses):
    plt.figure(figsize=(12, 8))
    plt.style.use("seaborn-muted")
    losses = torch.tensor(losses, dtype=torch.float)
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.plot(losses.numpy())
    plt.title("Error level by training iteration", fontsize=18, weight="bold")
  
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Take 100 episode averages and plot them too
    if len(losses) >= 100:
        means = losses.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig("errors_40002.png")


def sum_chunks(lst, interval):
    result = []
    for i in range(0, len(lst), interval):
        chunk_sum = 100 * (sum(lst[i : i + interval]) / interval)
        result.append(chunk_sum)
    return result


def plot_wins_stacked(results):
    wins, losses, illegal_moves_loss, draws, bench = results

    width = 200

    # Create bins every 200 games
    bins = list(np.arange(0, len(wins), width))

    # Difference the cumulative over time to get which game the outcomes are happening in
    wins_per_period = np.diff(np.array(wins))
    losses_per_period = np.diff(np.array(losses))
    illegal_moves_per_period = np.diff(np.array(illegal_moves_loss))
    draws_per_period = np.diff(np.array(draws))

    # Sum with the custom bins
    binned_wins = sum_chunks(wins_per_period, width)
    binned_losses = sum_chunks(losses_per_period, width)
    binned_illegal_moves = sum_chunks(illegal_moves_per_period, width)
    binned_draws = sum_chunks(draws_per_period, width)

    # Create a dataframe as easier to do the charts from this
    df = pd.DataFrame(
        {
            "Wins": binned_wins,
            "Losses": binned_losses,
            "Losses due to illegal move": binned_illegal_moves,
            "Draws": binned_draws,
        }
    )

    colors = ["gold", "blue", "red", "brown"]

    # Create a stacked bar chart
    plt.figure(figsize=(12, 8))
    plt.style.use("seaborn-muted")
    ax = df.plot.bar(stacked=True, color=colors, width=0.8)

    ax.set_xticklabels(bins[1:])

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.xlabel("Game Number", fontsize=12)
    plt.ylabel("% outcomes split every 200 games", fontsize=12)
    plt.title("Percentage Player 1 outcomes over time", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.legend(loc="best", fancybox=True, fontsize=7, shadow=True, ncol=2)
    plt.savefig(
        "winning_count_stacked2.png"
    )
