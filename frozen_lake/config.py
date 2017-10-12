class Config(object):
    def __init__(self):
        self.model_dir = './ckpt/'

    def get_model_dir(self):
        if self.model_dir[-1] != '/':
            return self.model_dir + '/'
        return self.model_dir
