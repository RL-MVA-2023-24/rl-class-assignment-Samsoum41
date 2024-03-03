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


def greedy_action(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.Q = None
        self.load()
        
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()

        return greedy_action(self.Q, observation, env.action_space.n)

    def save(self, path):
        # model_path = "dqn_maxEpLen200_200ep_24neurons"
        save_path = os.path.join('../model', f'{path}.joblib')
        dump(self.Q, save_path)
    def load(self):
        model_path = "Q_value_function_FQI_paper"
        save_path = os.path.join('../model', f'{model_path}.joblib')
        self.Q = load(save_path)