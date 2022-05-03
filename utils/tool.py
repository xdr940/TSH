
import json
import time
import numpy as np
import pandas as pd
def time_stat(last_end):
    end = time.time()
    running_time = end - last_end
    # print('--> time cost : %.5f sec' % running_time)
    return running_time

def get_now():
    return time.time()



def json2dict(file):
    with open(file, 'r') as f:
        dict = json.load(fp=f)
        return dict
def dict2json(file,dict):
    with open(file, 'w') as f:
        json.dump(dict, f)


def to_csv(file,series):
    series.to_csv(file)
def read_csv(file):
    return pd.read_csv(file,index_col=0)
