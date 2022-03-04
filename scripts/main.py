
from utils.yaml_wrapper import YamlHandler
import argparse
import datetime
from components.dataloader import AerDataset
from components.drawer import Drawer
from components.stator import Stator
from components.solver import Solver
import matplotlib.pyplot as plt
import os
from path import Path
from utils.tool import json2dict,dict2json,to_csv,read_csv

def main(args):
    yml = YamlHandler(args.settings)
    config = yml.read_yaml()
    instance_save_path = Path("/home/roit/models/sn_instances")/datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    instance_save_path.mkdir_p()

    print("\n===========DATA==============")
    # load data
    data = AerDataset(config)
    #split data
    data.data_prep()

    # split data reload and process
    if config['random_seed']:
        data.load_align(config['random_seed'])
    else:
        data.load_align()

    stator = Stator(data)
    stator.data_stat()

    data.data_parse()



    print("\n=============PROBLEM=============")
    solver = Solver(data)
    solver.build_graph(weights='tk')

    if config['algorithm']=='dp':

        final_solution = solver.dp_run()
    elif config['algorithm']=='mea':

        final_solution = solver.mea_run()
    else:
        final_solution = solver.mst_run()


    inter_tk_dict = solver.get_inter_tks(final_solution)
    final_value = solver.get_selected_alg_base(inter_tk_dict,final_solution)
    # solver.result_stat(final_solution,inter_tk_dict,final_value)

    carrier = stator.solution_stat(final_solution,final_value,algorithm=config['algorithm'])


    print('\n============DRAW===============')

    drawer = Drawer(data,config)



    plt.figure(figsize=[18,8])
    ax1=plt.subplot(2,3,4)
    drawer.drawAer(ax1,position=data.position)

    ax2=plt.subplot(2,3,5)
    drawer.drawAerSolution(ax2,position=data.position,
                                  final_solution=final_solution,
                                  inter_tk_dict=inter_tk_dict
                                  )

    plt.subplot(2, 3,6)
    drawer.drawGraph(solver.G, position=data.position, final_solution=final_solution)
    # figs = drawer.drawGraph(solver.G,position = data.position)

    ax4=plt.subplot(2, 1, 1)
    drawer.drawAccessSolution(ax4,
                                  position=data.position,
                                  final_solution=final_solution,
                                  inter_tk_dict=inter_tk_dict,
                             )

    print('============SAVE===============')
    plt.savefig(instance_save_path/'solution.png')

    yml.save_log(instance_save_path/'settings.yaml')
    dict2json(instance_save_path/'stat_results.json',carrier)
    to_csv(instance_save_path/'final_value.csv',final_value)
    read_csv(instance_save_path/'final_value.csv')


    print('-> LOGFILE SAVED AS :{}'.format(instance_save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/config.yaml')
    else:
        parser.add_argument("--settings", default='./configs/config.yaml')

    args = parser.parse_args()
    main(args)