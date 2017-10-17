from torch.utils.data import Dataset
import gym
import numpy as np
from torch.autograd import Variable
import torch
import random


class GameDataset(object):
    def __init__(self, q_function, batch_size=60):
        self.n = 0
        self.batch_game_play_count = 10
        self.batch_size = batch_size
        self.env = gym.make('FrozenLake-v0')
        self.env.reset()
        self.game_done = False
        self.MAX_STEPS = 1000000
        self.epsilon = 0.5
        self.discount_factor = 0.9
        self.actions = [0, 1, 2, 3]
        self.q_function = q_function
        self.play_memory = []

    @staticmethod
    def state_action_vector(observation, action):
        observation_vector = np.zeros(16)
        action_vector = np.zeros(4)
        observation_vector[observation] = 1
        action_vector[action] = 1
        return np.concatenate((observation_vector, action_vector), 0).astype(dtype=np.float32)

    def play_game(self):
        play_memory = []
        # Play game
        for i in range(self.batch_game_play_count):
            observation = self.env.reset()
            for j in range(self.MAX_STEPS):
                experience = list()
                experience.append(observation)
                if np.random.random() < self.epsilon:
                    current_action = self.env.action_space.sample()
                    observation, reward, is_done, info = self.env.step(current_action)
                else:
                    _, current_action = self.get_max_q_function_value_and_action(observation)
                    # print('current action : {0}'.format(current_action))
                    observation, reward, is_done, info = self.env.step(current_action)

                experience += [current_action, reward, observation, is_done]
                # print(experience)
                play_memory.append(
                    tuple(experience)
                )
                if is_done:
                    break
            if len(play_memory) >= self.batch_size:
                break

        random.shuffle(play_memory)
        self.play_memory = play_memory[:self.batch_size]

    def get_max_q_function_value_and_action(self, observation):
        q_value = -9999
        q_var = None
        current_action = None
        for action in self.actions:
            target_q_value = self.q_function(Variable(torch.from_numpy(self.state_action_vector(observation, action))))
            # print('target q value : {0}'.format(target_q_value.data.numpy()[0][0]))
            # print(type(target_q_value.data.numpy()))
            if target_q_value.data.numpy()[0][0] > q_value:
                q_value = target_q_value.data.numpy()[0][0]
                q_var = target_q_value
                current_action = action
        return q_var, current_action

    def __iter__(self):
        return GameDataIterator(self)


class GameDataIterator(object):
    def __init__(self, game_dataset):
        self.game_dataset = game_dataset
        self.n = 0

    def __next__(self):
        if self.n > 0:
            raise StopIteration
        self.game_dataset.play_game()
        self.n += 1
        return self.game_dataset.play_memory

    next = __next__
