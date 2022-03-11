
import pandas as pd
import numpy as np
from collections import  Counter
import matplotlib.pyplot as plt
import math


class Measurer:
    def __init__(self,algs=None):
        self.algs = algs

    def P_drop(self,procedures):# for a single dir, city-duration
        print('at pdrop')
        procedure_names = list(procedures.keys())
        procedure_names.sort()
        length = len(procedure_names)

        i=0
        user_demands=UserDemands()

        while i < length: # all procedures # multi-threading here
            j=i
            while procedure_names[j][:3] ==procedure_names[i][:3]: #batch procedures
                j+=1
            batch_procedures = [v for k, v in procedures.items() if k in procedure_names[i:j]]


            start,end = batch_procedures[0].start_end
            calls = user_demands.get_calls(start,end)

            # procedure_batch with call_batch (did not equal)
            algs_vec =[]
            batch_df=pd.DataFrame(columns=[])
            for procedure in batch_procedures:
                vec = procedure.inject(calls)
                algs_vec.append(vec)
            # get
            # procedure.inject(user_demands)
            pass
        pass



class UserDemands:
    def __init__(self):
        pass
    def get_calls(self,start,end,num=100):
        pass
        print(start,end)
        arr = [[74700, 74900], [74710, 75000], [74900, 75100],[74700,74720],[75150,75160]]
        st1 = sorted(arr, reverse=False, key=lambda x: x[1])
        st1 = np.array(st1)

        st2 = sorted(st1, reverse=False, key=lambda x: x[0])
        st2 = np.array(st2)

        return st2


class Procedure:
    def __init__(self,procedure_name,procedure_dict,procedure_value,default_num_calls=100):
        self.name = procedure_name
        self.description = procedure_dict
        self.data =procedure_value
        self.default_num_calls = default_num_calls


        self.tks = self.handover_instants.copy()
        self.tks .insert(0, self.start_end[0])
        self.tks .append(self.start_end[-1])
        self.num_tks = len(self.tks)
        self.pds = self.hand_pds()

    def __getattr__(self, item):
        return self.description[item]
    def hand_pds(self):


        hand_values =[]
        for i in range(self.num_tks-1):
            s,e = self.tks[i],self.tks[i+1]
            hand_value = float(self.data.loc[s:e].sum())
            hand_value/=(e-s)
            hand_values.append(hand_value)

        return list(np.exp(-np.array(hand_values)))
    def inject(self,calls):
        calls_pd_s=[]
        calls_pd_e=[]
        for call in calls:
            for i in range(0,self.num_tks-1):
                if self.tks[i]< call[0] and call[0] <self.tks[i+1]:
                    calls_pd_s.append(i)# if i-1 ==-1, call start is over started
                    break
                i+=1

            for j in range(self.num_tks-1,0,-1):
                if  self.tks[j-1]<call[1] and self.tks[j]:
                    calls_pd_e.append(j-1)
                    break
                j-=1
        ret = np.array([calls_pd_s,calls_pd_e]).transpose([1,0])
        return ret
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

