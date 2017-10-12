from models import Model
from bases import Base, Trainer
from config import Config
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet as tnt
import torch


class ConcreteTrainer(Base, Trainer):
    learning_rate = 0.001
    momentum = 0.5
    batch_size = 60

    def __init__(self):
        model, criterion, optim = self.set_up()
        config = Config()
        Base.__init__(self, config)
        Trainer.__init__(self, model, criterion, optim)

        self.meters = [tnt.meter.AverageValueMeter(), tnt.meter.ClassErrorMeter(accuracy=True)]
        self.batch_size = 60
        self.batch_workers = 2

    def set_up(self):
        # TODO : set up model, criterion, optimizer, dataset here.
        model = Model()
        criterion = F.nll_loss
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return model, criterion, optimizer

    def get_iterator(self, is_train):
        dataset = datasets.MNIST(
            './data', train=is_train, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        data = getattr(dataset, 'train_data' if is_train else 'test_data')
        labels = getattr(dataset, 'train_labels' if is_train else 'test_labels')
        tensor_datasets = tnt.dataset.TensorDataset([data, labels])
        return tensor_datasets.parallel(batch_size=self.batch_size, num_workers=self.batch_workers, shuffle=is_train)

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
