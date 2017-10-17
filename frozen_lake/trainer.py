from torch.utils.data import DataLoader

from game_dataset import GameDataset
from models import Model, Q
from bases import Base, Trainer
from config import Config
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from torch.autograd import Variable
import torchnet as tnt
import torch
import gym
import numpy as np
import random


class ConcreteTrainer(Base, Trainer):
    learning_rate = 0.0005
    momentum = 0.5
    batch_size = 60
    batch_game_play_count = 100
    MAX_STEPS = 1000000

    def __init__(self):
        self.actions = []
        q_function, criterion, optim = self.set_up()
        self.q_function = q_function
        config = Config()
        Base.__init__(self, config)
        Trainer.__init__(self, q_function, criterion, optim)

        self.meters = [tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()]
        self.batch_size = 60
        self.epsilon = 0.6
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
        # print(observation)
        observation_vector[observation] = 1
        action_vector[action] = 1
        return np.concatenate((observation_vector, action_vector), 0).astype(dtype=np.float32)

    def on_sample(self, state):
        # print(state['sample'])
        pass

    def get_iterator(self, is_train):
        dataset = GameDataset(self.q_function)
        return dataset

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

    def get_loss_and_output(self, sample):
        # sample : Batch of (observation, action, reward, next observation, is_done)
        inputs = None
        targets = None
        rewards = []
        # print(sample)
        for history in sample:
            # print('this is history {0}'.format(history))
            observation = history[0]
            action = history[1]
            reward = history[2]
            next_observation = history[3]
            is_done = history[4]
            input_value = self.q_function(Variable(torch.from_numpy(self.state_action_vector(observation, action))))
            if is_done:
                # print('reward : {0}'.format(reward))
                target_value = Variable(torch.FloatTensor([reward]))
            else:
                max_q_value, _ = self.get_max_q_function_value_and_action(next_observation)
                # print(type(max_q_value))
                target_value = reward + self.discount_factor * max_q_value
            if inputs is None:
                inputs = input_value
            else:
                inputs = torch.cat((inputs, input_value), 0)

            if targets is None:
                targets = target_value
            else:
                targets = torch.cat((targets, target_value), 0)
            # print(reward)
            # print(target_value)
            # targets.append(target_value)
            rewards.append(reward)

        # Set up loss & output
        # print(inputs)
        # print(targets)
        loss = self.criterion(inputs, targets)
        # print('this is loss')
        # print(rewards)
        return loss, np.mean(np.asarray(rewards, dtype=np.float32))

    def _print_information(self, prefix):
        print('Training loss: %.4f, mean rewards: %.6f' % (self.meters[0].value()[0], self.meters[1].value()[0]))

    def on_end_epoch(self, state):
        # print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        self._print_information('')
        # do validation at the end of each epoch
        self.reset_meters()
        # self.engine.test(self.get_loss_and_output, self.get_iterator(False))
        # self._print_information('')

    def on_forward(self, state):
        # output is rewards.
        np_loss = state['loss'].data.numpy()[0]
        # print(np_loss)
        np_output = state['output']
        # print(np_output)
        self.meters[0].add(np_loss)
        self.meters[1].add(np_output)
