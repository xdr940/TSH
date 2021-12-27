
from utils.yaml_wrapper import YamlHandler
import argparse

from dataset.dataloader import AerDataset
from components.drawer import Drawer
from components.accessor import Accessor
import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd

def main(args):
    yml = YamlHandler(args.settings)
    config = yml.read_yaml()


    # load data
    data = AerDataset(config)

    #split data
    data.data_prep(config)

    # split data reload and process
    data.load()

    drawer = Drawer()
    data.fig = drawer.drawAer(data, config=config)


    data.data_align(config)
    data.pre_alg()


    accessor = Accessor(data)
    accessor.build_graph()
    fig = drawer.drawGraph(accessor.G,position = accessor.position)
    plt.show()

    yml.save_log(data.out_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    parser.add_argument("--settings", default='../configs/config.yaml')
    args = parser.parse_args()
    main(args)