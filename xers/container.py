
from dataset.dataloader import AerDataset
from xers.drawer import Drawer
import matplotlib.pyplot as plt
import numpy as np
class Container:
    def __init__(self,config):
        self.config = config
        self.data = AerDataset(config['data_prep'])

        #

    def __call__(self):
        # step1 prep
        # self.data.data_prep()

        # step2. load
        self.data.load()


        # step3. draw
        # self.draw()

        # step4. random access
        self.data.data_recons(config =self.config['algorithm'] )


    def random_access(self):
        pass



    def draw(self):
        drawer = Drawer()
        # drawer.run(self.data.df)
        for idx, access in enumerate(self.data.access):
            ret = self.data[access][self.config['data_show']['lines']]  # get nparr
            drawer(idx, access, ret)

        drawer.run()