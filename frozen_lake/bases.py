import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import init
from torchnet.engine import Engine
import torchnet as tnt
from tqdm import tqdm


class Base(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    @property
    def model_param_dir(self):
        return self.config.get_model_dir() + self.__name__ + '.pkl'

    def save_model(self):
        print("[ Saving Model... ]")
        torch.save(self.model.state_dict(), self.model_param_dir)
        print("[ Saving Model Success. ]")

    def load_model(self):
        print("[ Loading Model... ]")
        try:
            self.model.load_state_dict(torch.load(self.model_param_dir))
            print("[ Loading Model Success. ]")
        except FileNotFoundError:
            print("[ Loading Model Failed. ]")


class Trainer(object):
    def __init__(self, model=None, criterion=None, optim=None):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.engine = Engine()

        # These parameters should be defined in subclass.
        self.meters = []
        self.batch_size = NotImplemented
        self.batch_workers = NotImplemented

    def set_up(self):
        raise NotImplemented

    def _print_information(self, prefix):
        raise NotImplemented

    def get_loss_and_output(self, sample):
        raise NotImplemented

    def reset_meters(self):
        for meter in self.meters:
            meter.reset()

    def get_iterator(self, is_train):
        raise NotImplemented

    @staticmethod
    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(self, state):
        raise NotImplemented

    def on_start_epoch(self, state):
        self.reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state):
        raise NotImplemented

    def on_update(self, state):
        raise NotImplemented

    def run(self, epochs):
        self.engine.hooks['on_sample'] = self.on_sample
        self.engine.hooks['on_forward'] = self.on_forward
        self.engine.hooks['on_start_epoch'] = self.on_start_epoch
        self.engine.hooks['on_end_epoch'] = self.on_end_epoch
        self.engine.train(self.get_loss_and_output, self.get_iterator(True), maxepoch=epochs, optimizer=self.optim)
