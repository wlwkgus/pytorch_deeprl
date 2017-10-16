from models import Model, Q
from bases import Base, Trainer
from config import Config
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet as tnt
import torch
import gym
import numpy as np
import random


class ConcreteTrainer(Base, Trainer):
    learning_rate = 0.001
    momentum = 0.5
    batch_size = 60
    batch_game_play_count = 10
    MAX_STEPS = 1000000

    def __init__(self):
        q_function, criterion, optim = self.set_up()
        self.q_function = q_function
        config = Config()
        Base.__init__(self, config)
        Trainer.__init__(self, q_function, criterion, optim)

        self.meters = [tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()]
        self.batch_size = 60
        self.batch_workers = 2
        self.actions = []
        self.env = gym.make('FrozenLake-v0')
        self.env.reset()
        self.epsilon = 0.2
        self.discount_factor = 0.9

    def set_up(self):
        # set up model, criterion, optimizer, dataset here.
        q = Q()
        # Check env.action_space : Discrete(4)
        self.actions = [0, 1, 2, 3]

        criterion = F.mse_loss
        optimizer = optim.SGD(q.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return q, criterion, optimizer

    @staticmethod
    def state_action_vector(observation, action):
        observation_vector = np.zeros(16)
        action_vector = np.zeros(4)
        observation_vector[observation] = 1
        action_vector[action] = 1
        return np.concatenate((observation_vector, action_vector), 0)

    def get_iterator(self, is_train):
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
                    observation, reward, is_done, info = self.env.step(current_action)

                experience += [current_action, reward, observation, is_done]
                play_memory.append(
                    tuple(experience)
                )
                if is_done:
                    break

        random.shuffle(play_memory)
        for i in range(int(len(play_memory) / self.batch_size)):
            yield play_memory[i*self.batch_size: (i+1)*self.batch_size]

    def get_max_q_function_value_and_action(self, observation):
        q_value = -9999
        current_action = None
        for action in self.actions:
            target_q_value = self.q_function(self.state_action_vector(observation, action))
            if target_q_value > q_value:
                q_value = target_q_value
                current_action = action
        return q_value, current_action

    def get_loss_and_output(self, sample):
        # sample : Batch of (observation, action, reward, next observation, is_done)
        inputs = []
        targets = []
        rewards = []
        for history in sample:
            observation = history[0]
            action = history[1]
            reward = history[2]
            next_observation = history[3]
            is_done = history[4]
            input_value = self.q_function(self.state_action_vector(observation, action))
            if is_done:
                target_value = reward
            else:
                max_q_value, _ = self.get_max_q_function_value_and_action(next_observation)
                target_value = reward + self.discount_factor * max_q_value
            inputs.append(input_value)
            targets.append(target_value)
            rewards.append(reward)

        # Set up loss & output
        inputs = Variable(inputs)
        targets = Variable(targets)
        loss = self.criterion(inputs, targets)
        return loss, rewards

    def _print_information(self, prefix):
        print('Training loss: %.4f, mean rewards: %.2f%%' % (self.meters[0].value()[0], self.meters[1].value()[0]))

    def on_end_epoch(self, state):
        # print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        self._print_information('')
        # do validation at the end of each epoch
        self.reset_meters()
        self.engine.test(self.get_loss_and_output, self.get_iterator(False))
        self._print_information('')

    def on_forward(self, state):
        # output is rewards.
        self.meters[0].add(state['loss'].data)
        self.meters[1].add(state['output'].data)
