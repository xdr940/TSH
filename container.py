
from dataset.dataloader import AerDataset

class Container:
    def __init__(self,config):
        self.config = config
        aer_dataset = AerDataset(config['data_prep'])
        # aer_dataset.data_prep()
        aer_dataset.load()

        #

        pass
    def __call__(self):
        pass