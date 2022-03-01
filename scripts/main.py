
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
    solver.build_graph(weights='tk')


    final_solution = solver.dp_run()
    # final_solution = solver.rss_run()
    # final_solution = solver.mst_run()
    #
    inter_tk_dict = solver.get_inter_tks(final_solution)

    final_value = solver.get_selected_alg_base(inter_tk_dict,final_solution)
    solver.result_stat(final_solution,final_value)

    #
    #
    fig1 = drawer.drawAer(data, config=config,position=data.position)
    # plt.plot(final_value,'r')
    fig2 = drawer.drawAerSolution(data=data,
                                  config=config,
                                  position=data.position,
                                  final_solution=final_solution,
                                  inter_tk_dict=inter_tk_dict,
                                  )
    fig4 = drawer.drawAccessSolution(data=data,
                                  config=config,
                                  position=data.position,
                                  final_solution=final_solution,
                                  inter_tk_dict=inter_tk_dict,
                             )
    fig3 = drawer.drawGraph(solver.G,position = data.position,final_solution=final_solution)
    figs = drawer.drawGraph(solver.G,position = data.position)
    #
    #
    #

    plt.show()
    #
    # yml.save_log(data.out_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/config.yaml')
    else:
        parser.add_argument("--settings", default='./configs/config.yaml')

    args = parser.parse_args()
    main(args)