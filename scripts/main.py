
from utils.yaml_wrapper import YamlHandler
import argparse

from dataset.dataloader import AerDataset
from components.drawer import Drawer
from components.solver import DpSolver
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




    data.data_align(config)
    data.pre_alg()


    solver = DpSolver(data)
    solver.build_graph()
    solver.run()
    solver.result_stat()



    drawer = Drawer()
    data.fig = drawer.drawAer(data, config=config,position=solver.position,soulution=solver.solution)
    fig = drawer.drawGraph(solver.G,position = solver.position,soulution=solver.solution)
    plt.show()

    yml.save_log(data.out_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    parser.add_argument("--settings", default='../configs/config.yaml')
    args = parser.parse_args()
    main(args)