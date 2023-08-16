# SnakeGame

## Background and Game details
Here Pygame is used to create the famous Snake Game. Some background is added on the screen. 
The game is played on 1 800x800 window with a block size of 80-100.
The snake's head is a yellow block and the body is green.
There is support for the snake to be played by a human but that is not the main point of this game.

## Relevant Info

This environment is created to train a Reinforcement Learning AI. The learning method used is TD learning.
Value iteration is used with TD learning which achieved a stable 5-6 length snake AI within 150k episodes.

## State Encoding

The state encoding is very novel here. Rather than using the whole X,Y of the snake and the fruit, a relative coordinate system is used.
The snake's head is taken as the (0,0) of this coordinate system. Then the snake's distance from the front and right wall are calculated.
The snake's direction of movement is irrelevant because the coordinate system is such that the snake always moves forward wrt to it.
The fruit's location is calculated wrt the snake's head. These 4 inputs are used the snake's state.

There are too many states to use Bellman Value Iteration on this. Therefore TD learning is chosen.

## Actions
#### Turn left
#### Turn right
#### Do nothing (keep moving in the current direction)



## Hyper parameters:

#### num_episode=1_000_000
#### max_step_per_episode=1000
#### max_snake_length=4
#### min_lr=0.01
#### max_lr=0.1
#### epsilon=0.1
#### gamma=0.9
#### decay_rate=inf
