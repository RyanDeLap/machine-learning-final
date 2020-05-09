# ryanml

## Quickstart
```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Running
```bash
python cartpole.py
```

## RL TL;DR
### Parameters
An agent is in an environment

The environment has a state `s`

The agent can observe `o` the environment state

The agent can act `π(s)` by taking an action `a` in the environment

The environment moves to another state based on a probability distribution `P`

The environment gives the agent a reward `r`

Rewards are positive if good, negative if bad

A Markov state is where all future states can be predicted based on the current state

Reinforcement learning assumes the environment is Markovian (has Markov states)

The history of all states can be included in a state representation, making it Markovian

### How do we make the agent perform well?

Each state can be assigned a value `V(s)`

... Or we can record the quality of taking an action in a state `Q(s,a)`

To act, simply take an action to the highest valued state at each step

### How can these values be found?

Bellman Equation: `V(s)=R(s,a)+γ*sum(P(s'|s,a)*V(s'))`

Where:
* `V(s)` is the value of a state
* `R(s,a)` is the reward of taking an action in the state
* `γ` is the discount factor, a hyperparameter, how far into the future should we care about? Typically 0.9-0.99
* `sum(P(s'|s,a)*V(s'))` is the sum of values for each immediate future state weighted by their probability of being transitioned to
