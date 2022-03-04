
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
    for seed in config['random_seed']:
        try:
            data.load_align()
            stator = Stator(data)

            # stator.data_stat()
        # drawer = Drawer(data,config)

            data.data_parse()



        # print("\n=============PROBLEM=============")
            solver = Solver(data)
            for alg in config['algorithm']:

                if alg=='dp':
                    solver.build_graph(weights='tk')

                    final_solution = solver.dp_run()
                elif alg=='mea':

                    final_solution = solver.mea_run()
                elif alg=='mst':
                    solver.build_graph(weights='1')

                    final_solution = solver.mst_run()
                else:
                    print('error in alg')
                    exit(-1)


                inter_tk_dict = solver.get_inter_tks(final_solution)
                final_value = solver.get_selected_alg_base(inter_tk_dict,final_solution)
                # solver.result_stat(final_solution,inter_tk_dict,final_value)

                carrier = stator.solution_stat(final_solution,final_value,algorithm=alg)



                dict2json(instance_save_path/'{}_{}_stat_results.json'.format(alg,seed),carrier)
                to_csv(instance_save_path/'{}_{}_final_value.csv'.format(alg,seed),final_value)
            del stator
        except:
            continue
        # del solver
    yml.save_log(instance_save_path/'settings.yaml')


    print('-> LOGFILE SAVED AS :{}'.format(instance_save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/batch_run_config.yaml')
    else:
        parser.add_argument("--settings", default='./configs/batch_run_config.yaml')

    args = parser.parse_args()
    main(args)