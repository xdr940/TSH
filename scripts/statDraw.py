'''
城市-仿真时长 --> 随机种子 --> 方法+链路
'''
from utils.yaml_wrapper import YamlHandler
import argparse

import matplotlib.pyplot as plt
import os
from path import Path
from dataset import ResultDataset
from components.drawer import StatDrawer

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

    drawer = StatDrawer(data, config_palette=config['palette'])
    drawer.draw_rssi()
    plt.figure()
    drawer.draw_num_handovers()
    plt.figure()
    drawer.draw_last_durations()
    plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/statDraw.yaml'):
        parser.add_argument("--settings", default='../configs/statDraw.yaml')
    else:
        parser.add_argument("--settings", default='./configs/statDraw.yaml')

    args = parser.parse_args()
    main(args)