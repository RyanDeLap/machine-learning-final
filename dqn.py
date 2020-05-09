import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# Q(s,a) = Q(s,a) + (reward + discount_factor * max_a'(Q(s',a')) - Q(s,a))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class ReplayBuffer:
    def __init__(self, observation_space, max_size=8192):
        self.observations = np.empty((max_size, observation_space))
        self.actions = np.empty((max_size,))
        self.rewards = np.empty((max_size,))
        self.next_observations = np.empty((max_size, observation_space))
        self.done = np.empty((max_size,))
        self.index = 0
        self.size = 0
        self.max_size = max_size

    def write(self, transition):
        observation, action, reward, next_observation, done = transition
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_observations[self.index] = next_observation
        self.done[self.index] = done
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def read(self, batch_size=128):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.observations[indices]).float().to(device),
            torch.from_numpy(self.actions[indices]).long().view(-1, 1).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.observations[indices]).float().to(device),
            torch.from_numpy(self.done[indices]).long(),
        )


class QNetwork(nn.Module):
    def __init__(self, observation_space, discrete_actions):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, 4)
        self.fc2 = nn.Linear(4, discrete_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, observation_space, discrete_actions, gamma=0.99, update_frequency=100):
        super().__init__()

        self.observation_space = observation_space
        self.discrete_actions = discrete_actions
        self.gamma = gamma  # discount factor
        self.update_frequency = update_frequency

        self.replay_buffer = ReplayBuffer(observation_space)
        self.base = QNetwork(observation_space, discrete_actions).to(device)
        self.target = QNetwork(observation_space, discrete_actions).to(device)
        self.update_targets()

        self.optimizer = optim.Adam(self.base.parameters())
        self.criterion = nn.MSELoss()

        self.gamma = gamma

        self.training_steps = 0

    def update_targets(self):
        self.target.load_state_dict(self.base.state_dict())
        self.target.eval()

    def memorise(self, transition):
        self.replay_buffer.write(transition)

    def train(self):
        if self.training_steps % self.update_frequency == 0:
            self.update_targets()

        observation, action, reward, next_action, done = self.replay_buffer.read()

        with torch.no_grad():
            future_q = self.target(observation)
            future_q = future_q.gather(1, action)
            future_q[done] = 0
            targets = reward + self.gamma * future_q

        self.optimizer.zero_grad()
        outputs = self.base(observation)
        targets = outputs.scatter(1, action, targets.float())
        loss = self.criterion(outputs, targets)
        loss.backward()
        for param in self.base.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def act_greedily(self, observation):
        return int(self.base(torch.from_numpy(observation).float().to(device)).argmax())
