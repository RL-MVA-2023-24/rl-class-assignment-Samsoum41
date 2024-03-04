from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from joblib import dump, load

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, nb_hidden_layers = 4, hid_dim = 512):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(n_observations, hid_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(nb_hidden_layers)])
        self.output_layer = nn.Linear(hid_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
          x = F.relu(hidden_layer(x))
        return self.output_layer(x)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        config = {
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 1000000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 10000,
            'epsilon_delay_decay': 300,
            'batch_size': 850,
            'tau': 0.001}

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = 4
        self.nb_observations = 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.nb_observations, self.nb_actions).to(device)
        self.target_model = DQN(self.nb_observations, self.nb_actions).to(device)
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.tau = config['tau']

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            # Double DQN, we use the target the model in the loss function
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 200 # CHANGE TO 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        # max_episode = 200
        current_episode_length = 0
        best_return = 0
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # Slowly copy the model 
            target_dict = self.target_model.state_dict()
            model_dict = self.model.state_dict()
            for key in model_dict:
                target_dict[key] = self.tau*model_dict[key] + (1-self.tau)*target_dict[key]
            self.target_model.load_state_dict(target_dict)


            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode),
                      # ", episode_length ", current_episode_length,
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return
    
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()

        return greedy_action(self.model, observation)

    def save(self, path):
        # model_path = "dqn_maxEpLen200_200ep_24neurons"
        save_path = os.path.join("./model", f'{path}.pt')
        torch.save({
            # 'episodes_return': returns,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)

    def load(self):
        model_path = "doubleDQN_randomisation_true_300ep"
        save_path = os.path.join("./model", f'{model_path}.pt')
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])