import pandas as pd
from path import Path
import random
import math
import numpy as np
import itertools
import datetime
import os
from components.measurement import Procedure,get_df
from utils.tool import json2dict,dict2json,to_csv,read_csv
import errno
class ResultDataset:
    def __init__(self,instances_path,stems,ignore_dirs,ignore_words):
        self.instances_path =instances_path
        self.stems = stems
        self.ignore_dirs = ignore_dirs
        self.ignore_words = ignore_words

    def load(self):
        if len(self.stems) > 0:
            dirs = [self.instances_path / stem for stem in self.stems]
        else:
            dirs = self.instances_path.dirs()

        total_dfs = pd.DataFrame(columns=['algorithm', 'num_handovers', 'sim_duration', 'avg_signal'])
        total_precedures={}
        for dir in dirs:  # 每个dir 是一个时间的sim
            if dir.stem in self.ignore_dirs:
                continue
            if dir.stem[:-4] in self.ignore_words:  # 放弃一些城市
                continue
            sim_duration = dir.stem[-4:]
            procedures = {}
            print(dir)
            if not dir.exists():
                os.error('wrong in {}'.format(dir))
                continue
            try:

                json_files = dir.files('*.json')
                csv_files = dir.files('*.csv')
                json_files.sort()
                csv_files.sort()
                for json_file, csv_file in zip(json_files, csv_files):
                    if json_file.stem != csv_file.stem:
                        os.error("json, csv did not match!")
                        exit(-1)
                    procedure_name = json_file.stem

                    procedure = Procedure(
                        procedure_name=procedure_name,
                        procedure_dict=json2dict(json_file),
                        procedure_value=read_csv(csv_file)
                    )
                    procedures[procedure_name]=procedure
                    # results[procedure_name] = json2dict(json_file)
                    # results[procedure_name]['signal_value'] = read_csv(csv_file)
                    # ProcedureSet(procedure_name,,)

                instance_df = get_df(procedures)# for draw
                total_dfs = pd.concat([total_dfs, instance_df], axis=0)



            except:
                os.error('wrong read in {}'.format(dir))
            total_precedures[sim_duration] = procedures

        total_dfs.index = np.linspace(start=0, stop=len(total_dfs) - 1, num=len(total_dfs))

        self.total_dfs = total_dfs # for drawer
        self.total_precedures = total_precedures
        self.simulation_durations = list(total_dfs['sim_duration'].value_counts().index)
        self.algs = list(total_dfs['algorithm'].value_counts().index)


    def get_value(self,name):
        ret_value = {}
        for alg in self.algs:
            ret_value[alg] = self.total_dfs.query("algorithm== '{}'".format(alg)).groupby('sim_duration')[name].median()
        return ret_value



