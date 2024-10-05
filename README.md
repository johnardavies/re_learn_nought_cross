# Reinforcement learning example

This is repo which implements a reinforcement learning approach to Noughts and crosses. It uses the PyTorch framework.

The code is talked through in more detail in this post [here](https://johnardavies.github.io/technical/reinforcement/). 

  - **config.py** holds key parameters
  - **game.py** records the state of the game (the position of pieces on the board) and how the players are doing
  - **game_moves.py** generates the player moves
  - **game_network.py** specifies the network that estimates the value function
  - **optimising_network.py** adjusts the network weights to satisfy value function condition 
  - **chart_funcs.py** tracks game progress and charts it
  - **learn_game.py** imports all the scripts and runs the training loop that plays the game and trains the network

### To run the example:

### 1.  Clone the repo
```
$ git clone https://github.com/johnardavies/re_learn_nought_cross
```
### 2.  Install the project dependencies
Create and activate virtual environment, here called pytorch_env and install the requirements:
```
$ python -m  venv relearn_env && source relearn_env/bin/activate &&  pip install -r requirements.txt
```
### 3.  Train the network to play the game
cd into the project directory and run the below: 
```
(relearn_env) $ python game_learn.py
```
This will output the metrics on how the network is playing the game.
