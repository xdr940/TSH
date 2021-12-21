
from utils.yaml_wrapper import YamlHandler
import argparse

from dataset.dataloader import AerDataset
from components.drawer import Drawer
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
    drawer(data, config=config)
    data.fig = drawer.get_handle()


    data.data_align(config)
    data.alg()




    yml.save_log(data.out_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    parser.add_argument("--settings", default='../configs/config.yaml')
    args = parser.parse_args()
    main(args)