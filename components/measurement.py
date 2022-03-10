
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math


class Measurer:
    def __init__(self):
        pass

    def P_drop(self,procedures):# for a single dir, city-duration
        print('at pdrop')
        procedure_names = list(procedures.keys())
        procedure_names.sort()
        length = len(procedure_names)

        i=0
        user_demands=UserDemands()

        while i < length:
            j=i
            while procedure_names[j][:3] ==procedure_names[i][:3]:
                j+=1
            batch_procedures = [v for k, v in procedures.items() if k in procedure_names[i:j]]
            start,end = batch_procedures[0].start_end
            calls = user_demands.get_calls(start,end)
        for procedure in procedures:# multi-threading here
            drop_vec =  procedure.drop(user_demands)
            # get
            # procedure.inject(user_demands)
            pass
        pass
    def get_calls(self):
        pass


class UserDemands:
    def __init__(self):
        pass
    def get_calls(self,start,end,num=100):
        pass
        print(start,end)

class Procedure:
    def __init__(self,procedure_name,procedure_dict,procedure_value,default_num_calls=100):
        self.name = procedure_name
        self.description = procedure_dict
        self.data =procedure_value
        self.default_num_calls = default_num_calls


        self.tks = self.handover_instants.copy()
        self.tks .insert(0, self.start_end[0])
        self.tks .append(self.start_end[-1])
        self.per_hand_Eavg = self.per_hand()

    def __getattr__(self, item):
        return self.description[item]
    def hand_pds(self):

        num_tks=len(self.tks)
        hand_values =[]
        for i in range(num_tks-1):
            s,e = self.tks[i],self.tks[i+1]
            hand_value = float(self.data.loc[s:e].sum())
            hand_value/=(e-s)
            hand_values.append(hand_value)

        self.pds = list(np.exp(-np.array(hand_values)))


def get_desc_vec(pds,calls):
    length = len(calls)
    ret_01 = np.ones_like([1, length])
    pass

def get_df(procedures):


    instance_df = pd.DataFrame()
    instance_df['procedure_name'] = np.array([procedure.name for procedure in procedures])
    instance_df['algorithm'] = [procedure.algorithm for procedure in procedures]
    instance_df['num_handovers'] = [procedure.num_handovers for procedure in procedures]
    instance_df['sim_duration'] = [procedure.sim_duration for procedure in procedures]
    instance_df['avg_signal'] = [procedure.avg_signal for procedure in procedures]
    instance_df['avg_hand_duration'] = [procedure.avg_hand_duration for procedure in procedures]



    return instance_df

