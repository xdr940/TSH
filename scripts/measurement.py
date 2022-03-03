
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
def get_np_value(results,name):
    arr=[]
    for item in results:
        arr.append(item[name])
    return arr

def main(args):
    dp_results=[]
    mst_results=[]
    mea_results=[]

    dp_values =[]
    mst_values=[]
    mea_values =[]
    instances_path = Path("/home/roit/models/sn_instances")
    for dir in instances_path.dirs():
        jsons_files = dir.files('*.json')
        csv_files = dir.files('*.csv')
        jsons_files.sort()
        csv_files.sort()
        for item in jsons_files:
            if 'dp' in item.stem:
                dp_results.append(json2dict(item))
            if 'mst' in item.stem:
                mst_results.append(json2dict(item))
            if 'mea' in item.stem:
                mea_results.append(json2dict(item))
        for item in csv_files:
            if 'dp' in item.stem:
                dp_values.append(read_csv(item))
            if 'mst' in item.stem:
                mst_values.append(read_csv(item))
            if 'mea' in item.stem:
                mea_values.append(read_csv(item))

        dp_arr = get_np_value(dp_results,'handover times')
        mea_arr = get_np_value(mea_results,'handover times')
        mst_arr = get_np_value(mst_results,'handover times')
        plt.plot(dp_arr,'r')
        plt.plot(mea_arr,'g')
        plt.plot(mst_arr,'b')
        plt.figure()
        dp_arr = get_np_value(dp_results, 'avg signal')
        mea_arr = get_np_value(mea_results, 'avg signal')
        mst_arr = get_np_value(mst_results, 'avg signal')
        plt.plot(dp_arr, 'r')
        plt.plot(mea_arr, 'g')
        plt.plot(mst_arr, 'b')

        plt.show()

    print('ok')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/config.yaml')
    else:
        parser.add_argument("--settings", default='./configs/config.yaml')

    args = parser.parse_args()
    main(args)