
from utils.yaml_wrapper import YamlHandler
import argparse
import datetime
from components.dataloader import AerDataset
from components.drawer import Drawer
from components.stator import Stator
from components.solver import Solver
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from path import Path
from utils.tool import json2dict,dict2json,to_csv,read_csv
import seaborn as sns
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
    greedy_values=[]
    results=[]

    yml = YamlHandler(args.settings)
    config = yml.read_yaml()
    instances_path = Path(config['instances_path'])
    stems = config['stems']
    if len(stems)>0:
        dirs =[ instances_path/stem for stem in stems]
    else:
        dirs = instances_path.dirs()


    results_df = pd.DataFrame(columns=['algorithm','handover times','total time','avg signal'])
    for dir in dirs:# 每个dir 是一个时间的sim
        if dir.stem in config['ignore']:
            continue
        instance_df = pd.DataFrame()
        if not dir.exists():
            os.error('wrong in {}'.format(dir))
            continue
        try:


            jsons_files = dir.files('*.json')
            csv_files = dir.files('*.csv')
            jsons_files.sort()
            csv_files.sort()
            for item in jsons_files:
                results.append(json2dict(item))
            for item in csv_files:
                if 'dp' in item.stem:
                    dp_values.append(read_csv(item))
                if 'mst' in item.stem:
                    mst_values.append(read_csv(item))
                if 'mea' in item.stem:
                    mea_values.append(read_csv(item))
                if 'greedy' in item.stem:
                    greedy_values.append(read_csv(item))

            instance_df['algorithm'] = get_np_value(results,'algorithm')
            instance_df['handover times'] = get_np_value(results,'handover times')
            instance_df['total time'] = get_np_value(results,'total time')
            instance_df['avg signal'] = get_np_value(results, 'avg signal')
            instance_df['avg duration'] = get_np_value(results,'avg duration')
            results_df = pd.concat([results_df,instance_df],axis=0)
        except:
            os.error('wrong read in {}'.format(dir))
    results_df.index = np.linspace(start=0, stop=len(results_df) - 1, num=len(results_df))



    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    fig1= sns.boxplot(x="total time", y="avg signal",
                hue="algorithm", palette="Set3",
                data=results_df)

    boxplot_x = list(range(len(results_df['total time'].value_counts())))
    dp = results_df.query("algorithm== 'dp'").groupby('total time')['avg signal'].median()
    mea = results_df.query("algorithm== 'mea'").groupby('total time')['avg signal'].median()
    mst = results_df.query("algorithm== 'mst'").groupby('total time')['avg signal'].median()
    greedy = results_df.query("algorithm== 'greedy'").groupby('total time')['avg signal'].median()


    sns.lineplot(x=boxplot_x,y=dp.values)
    sns.lineplot(x=boxplot_x,y=mea.values)
    sns.lineplot(x=boxplot_x,y=mst.values)
    sns.lineplot(x=boxplot_x,y=greedy.values)


    plt.xlabel('Simulation Time (Second)')
    plt.ylabel('RSSI')

    #handover times
    plt.subplot(1,3,2)


    fig2= sns.boxplot(x="total time", y="handover times",
                hue="algorithm", palette="Set3",
                data=results_df)

    dp_handover = results_df.query("algorithm== 'dp'").groupby('total time')['handover times'].median()
    mea_handover = results_df.query("algorithm== 'mea'").groupby('total time')['handover times'].median()
    mst_handover = results_df.query("algorithm== 'mst'").groupby('total time')['handover times'].median()
    greedy_handover = results_df.query("algorithm== 'greedy'").groupby('total time')['handover times'].median()

    # sns.lineplot(x=list(range(0,len(mst.values))),y=mst.values)
    sns.lineplot(x=boxplot_x, y=dp_handover.values, palette="Set3")
    sns.lineplot(x=boxplot_x, y=mea_handover.values, palette="Set3")
    sns.lineplot(x=boxplot_x, y=mst_handover.values, palette="Set3")
    sns.lineplot(x=boxplot_x, y=greedy_handover.values, palette="Set3")


    plt.xlabel('Simulation Time (Second)')
    plt.ylabel('Handover Times')


    # dp_avg_duration =results_df.query("algorithm== 'dp'").groupby('total time')['avg duration'].median()
    # mea_avg_duration = results_df.query("algorithm== 'mea'").groupby('total time')['avg duration'].median()
    # mst_avg_duration = results_df.query("algorithm== 'mst'").groupby('total time')['avg duration'].median()
    # sns.lineplot(data=results_df,x="total time",y="avg signal",hue='algorithm')
    plt.subplot(1,3,3)

    plt.xlabel('Simulation Time (Second)')
    plt.ylabel('Avg Access Span /Satllite (Second)')
    sns.lineplot(data=results_df,x="total time",y="avg duration",hue='algorithm')

    # plt.figure()
    # sns.lineplot(data=results_df,x="total time",y="avg duration",hue='algorithm')


    # plt.figure()




    # plt.plot(arr[1],arr[0])

    # sns.despine(offset=10, trim=False)
    plt.show()




    print('ok')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    if os.path.exists('../configs/config.yaml'):
        parser.add_argument("--settings", default='../configs/measurement.yaml')
    else:
        parser.add_argument("--settings", default='./configs/measurement.yaml')

    args = parser.parse_args()
    main(args)