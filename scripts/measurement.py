'''
城市-仿真时长 --> 随机种子 --> 方法+链路
'''
from utils.yaml_wrapper import YamlHandler
import argparse

import matplotlib.pyplot as plt
import os
from path import Path
from dataset import ResultDataset,CallDataset
from components.drawer import StatDrawer
import numpy as np
import pandas as pd
import seaborn as sns
from components.measurement import Measurer

def main(args):

    yml = YamlHandler(args.settings)
    config = yml.read_yaml()
    data = ResultDataset(
        instances_path=Path(config['instances_path']),
        stems=config['stems'],
        ignore_dirs=config['ignore_dirs'],
        ignore_words=config['ignore_words']
    )
    data.load()
    sim_duration = config['stems'][0][-4:]

    procedure_container = Measurer(data.total_precedures[sim_duration])
    start_end_list = procedure_container.get_start_end()
    call_container = CallDataset(start_end_list=start_end_list)

    total_dfs = pd.DataFrame(columns=['start', 'end'], index=None)
    for batch_procedure, batch_call in zip(
            procedure_container.get_batch_procedure(),
            call_container.get_batch_calls()
    ):
        value_vec = []#对么个batch 的procedure 的值记录
        algs_name = [procedure.name[4:] for procedure in batch_procedure]
        names_arry = []
        batch_df = pd.DataFrame(columns=total_dfs.columns)
        for procedure, name in zip(batch_procedure, algs_name):
            vec, handover = procedure.inject(batch_call)
            names_arry += ([name] * len(vec))
            value_vec.append(vec)
        batch_call_resize = np.concatenate([batch_call[:-1]] * len(algs_name))
        batch_df[['start', 'end']] = batch_call_resize
        batch_df['duration'] = batch_df['end'] - batch_df['start']
        batch_df['value'] = np.concatenate(value_vec, axis=0)
        batch_df['alg'] = names_arry
        batch_df.index = range(len(total_dfs), len(total_dfs) + len(names_arry))

        total_dfs = pd.concat([total_dfs, batch_df], axis=0)

    sns.lineplot(data=total_dfs, x="duration", y="value", hue="alg", hue_order=['mst', 'mea', 'dp', 'gd'],
                 palette=['r', 'g', 'b', 'y'])
    plt.yscale("log")
    plt.grid()

    plt.ylabel("P_drop")
    plt.xlabel("Call Duration (Sec)")

    plt.yticks([0.1, 0.2, 0.5, 1])

    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/measurement.yaml')
    else:
        parser.add_argument("--settings", default='./configs/measurement.yaml')

    args = parser.parse_args()
    main(args)