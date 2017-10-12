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


class ConcreteTrainer(Base, Trainer):
    learning_rate = 0.001
    momentum = 0.5
    batch_size = 60
    batch_game_play_count = 10

    def __init__(self):
        q_function, criterion, optim = self.set_up()
        self.q_function = q_function
        config = Config()
        Base.__init__(self, config)
        Trainer.__init__(self, q_function, criterion, optim)

        self.meters = [tnt.meter.AverageValueMeter(), tnt.meter.ClassErrorMeter(accuracy=True)]
        self.batch_size = 60
        self.batch_workers = 2
        self.actions = []
        self.env = gym.make('FrozenLake-v0')
        self.env.reset()
        self.epsilon = 0.2
        self.reduce_factor = 0.9

    def set_up(self):
        # TODO : set up model, criterion, optimizer, dataset here.
        q = Q()
        # TODO : set up action spaces
        # Check env.action_space : Discrete(4)
        self.actions = [0, 1, 2, 3]

        criterion = F.nll_loss
        optimizer = optim.SGD(q.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return q, criterion, optimizer

    def get_iterator(self, is_train):
        # TODO : implement game iterator.
        for i in range(self.batch_game_play_count):
            # TODO : play game here
            self.env.reset()
            game_board = self.env.render()
            is_done = False
            while not is_done:
                if np.random.random() < self.epsilon:
                    observation, reward, is_done, info = self.env.step(self.env.action_space.sample())
                else:
                    # TODO : get maxarg q value
                    q_value = -9999
                    maxarg_action = None
                    for action in self.actions:
                        if self.q_function()
                    observation, reward, is_done, info = self.env.step(maxarg_action)
            # TODO : format game memory
            sample = ()
            yield sample

    def get_loss_and_output(self, sample):
        inputs = Variable(sample[0].float())
        targets = Variable(torch.LongTensor(sample[1]))
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        return loss, output

    def _print_information(self, prefix):
        print('Training loss: %.4f, accuracy: %.2f%%' % (self.meters[0].value()[0], self.meters[1].value()[0]))

    def on_end_epoch(self, state):
        # print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        self._print_information('')
        # do validation at the end of each epoch
        self.reset_meters()
        self.engine.test(self.get_loss_and_output, self.get_iterator(False))
        self._print_information('')

    def on_forward(self, state):
        self.meters[1].add(state['output'].data, torch.LongTensor(state['sample'][1]))
        self.meters[0].add(state['loss'].data[0])
