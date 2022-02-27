
from utils.yaml_wrapper import YamlHandler
import argparse

from components.dataloader import AerDataset
from components.drawer import Drawer
from components.stator import Stator
from components.solver import DpSolver
import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd
import os

from utils.tool import time_stat

def main(args):
    yml = YamlHandler(args.settings)
    config = yml.read_yaml()

    print("\n=====DATA=======")
    # load data
    data = AerDataset(config)
    #split data
    has_solution = data.data_prep()

    # split data reload and process
    data.load_align()

    stator = Stator(data)
    stator.data_stat()

    data.data_parse()



    print("\n=====PROBLEM=======")



    drawer = Drawer()
    #
    solver = DpSolver(data)
    solver.build_graph()
    solver.run()
    solver.result_stat()
    # #
    #
    fig1 = drawer.drawAer(data, config=config,position=data.position)
    fig2 = drawer.drawAerSolution(data=data,
                                  config=config,
                                  position=data.position,
                                  final_solution=solver.final_solution,
                                  inter_tk_dict=solver.inter_tk_dict,
                                  data_processed=solver.data)
    fig4 = drawer.drawAccessSolution(data=data,
                                  config=config,
                                  position=data.position,
                                  final_solution=solver.final_solution,
                                  inter_tk_dict=solver.inter_tk_dict,
                                  data_processed=solver.data
                             )
    fig3 = drawer.drawGraph(solver.G,position = data.position,final_solution=solver.final_solution)

    #
    #
    #

    plt.show()

    yml.save_log(data.out_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/measure_config.yaml'):
        parser.add_argument("--settings", default='../configs/measure_config.yaml')
    else:
        parser.add_argument("--settings", default='./configs/measure_config.yaml')

    args = parser.parse_args()
    main(args)