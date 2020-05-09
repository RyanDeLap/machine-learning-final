import abc
import copy

import blackjack


class MonteCarloAgent(abc.ABC):
    def __init__(self, action_space):
        self.action_space = action_space
        # learned parameters for each given state in the environment
        self.default_state_parameters = {
            'q': [0 for _ in range(action_space)],  # action quality
            'n': 0,  # state visitation count
        }
        self.traceback = []

        # learning horizon - how far the agent looks into the future to value their current state
        self.gamma = 0.9

    @abc.abstractmethod
    def state_properties(self, ace_in_hand, score, flipped_card_score):
        pass

    def act(self, hand, score, flipped_card_score):
        state_properties = self.state_properties('Ace' in hand, score, flipped_card_score)
        action = max(range(self.action_space), key=lambda i: state_properties['q'][i])
        return action

    def memorise(self, hand, score, flipped_card_score, action):
        state_properties = self.state_properties('Ace' in hand, score, flipped_card_score)
        self.traceback.append((state_properties, action))

    def train(self, reward):
        """Updates agent parameters found within the traceback with the end of episode reward."""
        next_state = None
        for current_state, action in reversed(self.traceback):
            n = current_state['n'] = current_state['n'] + 1
            if next_state is None:
                current_state['q'][action] += (1 / n) * (reward - current_state['q'][action])
            else:
                current_state['q'][action] += (1 / n) * (self.gamma * max(next_state['q']) - current_state['q'][action])
            next_state = current_state
        self.traceback = []


class MonteCarloDealer(MonteCarloAgent):
    def __init__(self, action_space):
        # dealer observes: whether dealer has usable ace, dealer's score
        super().__init__(action_space)
        self.table = [[copy.deepcopy(self.default_state_parameters) for _ in range(21)] for _ in range(2)]

    def state_properties(self, ace_in_hand, score, flipped_card_score):
        return self.table[ace_in_hand][score - 1]


class MonteCarloPlayer(MonteCarloAgent):
    def __init__(self, action_space):
        # player observes: whether player has usable ace, player's score, dealer's flipped card
        super().__init__(action_space)
        self.table = [[[copy.deepcopy(self.default_state_parameters) for _ in range(11)] for _ in range(21)]
                      for _ in range(2)]

    def state_properties(self, ace_in_hand, score, flipped_card_score):
        return self.table[ace_in_hand][score - 1][flipped_card_score - 1]


def main():
    player = MonteCarloPlayer(2)
    dealer = MonteCarloDealer(2)
    blackjack.train(10000, player, dealer)
    while blackjack.play_round(player, dealer):
        continue


if __name__ == '__main__':
    main()
