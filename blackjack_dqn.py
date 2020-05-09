import abc

import numpy as np

import blackjack
from dqn import DQN


class BlackjackDQNAgent(abc.ABC):
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.previous_observation = None
        self.previous_action = None
        self.multiple_steps = False
        self.dqn = DQN(observation_space, action_space)
        self.skip_train_steps = 1000
        self.skipped = 0

    @abc.abstractmethod
    def convert_transition(self, hand, score, flipped_card_score):
        pass

    def act(self, hand, score, flipped_card_score):
        return self.dqn.act_greedily(self.convert_transition(hand, score, flipped_card_score))

    def memorise(self, hand, score, flipped_card_score, action):
        observation = self.convert_transition(hand, score, flipped_card_score)

        if self.previous_observation is None:
            self.previous_observation = observation
            self.previous_action = action
            return

        self.dqn.memorise((self.previous_observation, self.previous_action, 0, observation, 0))
        self.previous_observation = observation
        self.previous_action = action
        self.multiple_steps = True

    def train(self, reward):
        index = self.dqn.replay_buffer.index - 1
        if index < 0:
            index = self.dqn.replay_buffer.max_size - 1

        if self.multiple_steps:
            self.dqn.replay_buffer.rewards[index] = reward
            self.dqn.replay_buffer.done[index] = 1
        else:
            self.dqn.memorise((self.previous_observation, self.previous_action, reward, self.previous_observation, 1))

        self.previous_observation = None
        self.previous_action = None

        if self.skipped % self.skip_train_steps == 0:
            self.dqn.train()


class BlackjackPlayerDQNAgent(BlackjackDQNAgent):
    def __init__(self, action_space):
        super().__init__(3, action_space)

    def convert_transition(self, hand, score, flipped_card_score):
        return np.array((int('Ace' in hand), score / 21, flipped_card_score / 11))


class BlackjackDealerDQNAgent(BlackjackDQNAgent):
    def __init__(self, action_space):
        super().__init__(2, action_space)

    def convert_transition(self, hand, score, flipped_card_score):
        return np.array((int('Ace' in hand), score / 21))


def main():
    player = BlackjackPlayerDQNAgent(2)
    dealer = BlackjackDealerDQNAgent(2)
    blackjack.train(1000, player, dealer)
    while blackjack.play_round(player, dealer):
        continue


if __name__ == '__main__':
    main()
