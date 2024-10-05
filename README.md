# Reinforcement learning example

This is repo which implements a reinforcement learning approach to Noughts and crosses. It uses the PyTorch framework.

The code is talked through in more detail in this post [here](https://johnardavies.github.io/technical/reinforcement/). .

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
