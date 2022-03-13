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
from components.measurement import Measurer

def main(args):

    yml = YamlHandler(args.settings)
    config = yml.read_yaml()
    data = ResultDataset(
        instances_path = Path(config['instances_path']),
        stems=config['stems'],
        ignore_dirs=config['ignore_dirs'],
        ignore_words=config['ignore_words']
    )
    data.load()

    call_set = CallDataset()
    measurer = Measurer()
    measurer.P_drop(data.total_precedures['0500'])




    drawer = StatDrawer(data,config['palette'])

    user_demands = data.create_user_demands()
    # FIG1
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    drawer.draw_rssi()




    #FIG 2 handover times
    plt.subplot(1,3,2)

    drawer.draw_num_handovers()



    # FIG 3
    plt.subplot(1,3,3)
    drawer.draw_last_durations()

    plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/measurement.yaml')
    else:
        parser.add_argument("--settings", default='./configs/measurement.yaml')

    args = parser.parse_args()
    main(args)