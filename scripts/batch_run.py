
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
from tqdm import tqdm
def main(args):
    yml = YamlHandler(args.settings)
    out_stem = args.out_stem
    config = yml.read_yaml()
    if not out_stem:
        instance_save_path = Path("/home/roit/models/sn_instances")/datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    else:
        instance_save_path = Path("/home/roit/models/sn_instances")/out_stem

    instance_save_path.mkdir_p()

    print("\n===========DATA==============")
    # load data
    data = AerDataset(config)
    #split data
    data.data_prep()

    # split data reload and process
    for seed in tqdm(config['random_seed']):
        try:
            data.load_align()
            stator = Stator(data)

            # stator.data_stat()
        # drawer = Drawer(data,config)

            data.data_parse()



        # print("\n=============PROBLEM=============")
            solver = Solver(data)
            solver.build_graph(weights='tk')
            final_solution=None
            for alg in config['algorithm']:
                if alg=='dp':

                    final_solution = solver.dp_run()
                elif alg=='mea':

                    final_solution = solver.mea_run()
                elif alg=='mst':
                    solver.build_graph(weights='1')

                    final_solution = solver.mst_run()
                elif alg =='gd':

                    final_solution = solver.greedy_run()
                else:
                    print('error in alg')
                    exit(-1)


                inter_tk_dict,inter_tk_list = solver.get_inter_tks(final_solution)
                final_value = solver.get_selected_alg_base(inter_tk_dict,final_solution)
                # solver.result_stat(final_solution,inter_tk_dict,final_value)

                carrier = stator.solution_stat(final_solution,inter_tk_list,final_value,algorithm=alg)


                to_csv(instance_save_path/'{:03d}_{}.csv'.format(seed,alg),final_value)# final value

                dict2json(instance_save_path/'{:03d}_{}.json'.format(seed,alg),carrier)# stat results
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

    parser.add_argument("--out_stem",default=None)

    args = parser.parse_args()
    main(args)