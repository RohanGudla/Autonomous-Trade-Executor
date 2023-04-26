
# Stock Trading Bot with Deep Reinforcement Learning

This project presents a Stock Trading Bot developed using Deep Reinforcement Learning, specifically Deep Q-learning. The implementation aims to adhere closely to the algorithm discussed in the research paper, with simplicity to aid in learning.

## Introduction

Reinforcement Learning belongs to a class of machine learning techniques enabling the creation of intelligent agents that learn from interactions with the environment. These agents learn optimal policies through trial and error. This approach is particularly valuable in real-world scenarios where supervised learning may not be suitable due to factors like the task's nature or the absence of labeled data.

The key concept here is that Reinforcement Learning can be applied to various real-world tasks described as Markovian processes.

## Approach

This project employs a Model-free Reinforcement Learning technique called Deep Q-Learning, which is a neural variant of Q-Learning. In each episode, the agent observes its current state (represented as an n-day window of stock prices), selects and executes an action (buy/sell/hold), observes a subsequent state, receives a reward signal (reflecting changes in portfolio position), and updates its parameters based on the loss gradient.

Several enhancements to the Q-learning algorithm have been incorporated into this project, including:

- Vanilla DQN
- DQN with fixed target distribution
- Double DQN
- Prioritized Experience Replay (pending)
- Dueling Network Architectures (pending)

## Results

The agent was trained on `GOOG` stock data from 2010 to 2017 and tested on data from 2019, resulting in a profit of $1141.45 (validated on 2018 data with a profit of $863.41):

![Google Stock Trading Episode](./extra/visualization.png)

You can generate similar visualizations for your model evaluations using the provided [notebook](./visualize.ipynb).

## Some Considerations

- The agent can only choose to buy/sell one stock at a time in any given state to simplify the problem of portfolio redistribution.
- The n-day window feature representation comprises a vector of consecutive differences in the Adjusted Closing price of the traded stock, followed by a sigmoid operation to normalize the values to the range [0, 1].
- Training is preferably conducted on a CPU due to its sequential nature. After each trading episode, experience is replayed (1 epoch over a small minibatch), and model parameters are updated.

## Data

You can download historical financial data for training from [Yahoo! Finance](https://ca.finance.yahoo.com/) or use sample datasets located under `data/`.

## Getting Started

To use this project, first install the required Python packages:

```bash
pip3 install -r requirements.txt
```

Then, initiate training by running the following command:

```bash
python3 train.py data/GOOG.csv data/GOOG_2018.csv --strategy t-dqn
```

Once training is complete, execute the evaluation script to let the agent make trading decisions:

```bash
python3 eval.py data/GOOG_2019.csv --model-name model_GOOG_50 --debug
```

Now, you're ready to go!

## Acknowledgments

- [@keon](https://github.com/keon) for the [deep-q-learning](https://github.com/keon/deep-q-learning) repository
- [@edwardhdlu](https://github.com/edwardhdlu) for the [q-trader](https://github.com/edwardhdlu/q-trader) repository

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

The original implementation and data credits go to their respective authors. This information is presented here for personal understanding and educational purposes.