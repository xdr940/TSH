
import pandas as pd
import numpy as np
from collections import  Counter
import matplotlib.pyplot as plt
import math


class Measurer:
    def __init__(self,procedures):
        self.procedures = procedures
        self.alg_cnt = Counter([item[4:] for item in list(self.procedures.keys())])


    def get_start_end(self):

        start_end_list=[]
        for k,v in self.procedures.items():
            start_end_list.append(tuple(v.start_end))
        return start_end_list
    def Pdrop_run(self,procedures):# for a single dir, city-duration
        print('at pdrop')
        procedure_names = list(procedures.keys())
        procedure_names.sort()
        length = len(procedure_names)

        i=0
        user_demands=CallDataset(None)

        total_dfs = pd.DataFrame(columns=['start','end']+list(self.alg_cnt.keys()))


        while i < length: # all procedures # multi-threading here
            j=i
            while procedure_names[j][:3] ==procedure_names[i][:3]: #batch procedures
                j+=1
            batch_procedures = [v for k, v in procedures.items() if k in procedure_names[i:j]]


            start,end = batch_procedures[0].start_end
            calls = user_demands.get_calls(start,end)

            # procedure_batch with call_batch (did not equal)
            algs_vec =[]
            batch_df=pd.DataFrame(columns=total_dfs.columns)
            for procedure in batch_procedures:
                vec,handover = procedure.inject(calls)
                algs_vec.append(vec)
            batch_df[['start','end']] = calls
            batch_df[list(alg_cnt.keys())] = np.array(algs_vec).T
            total_dfs = pd.concat([total_dfs, batch_df], axis=0)

            print('ok')

    def get_batch_procedure(self):



        procedure_names = list(self.procedures.keys())
        procedure_names.sort()
        length = len(procedure_names)
        i=0
        while i < length:  # all procedures # multi-threading here
            j = i

            while procedure_names[j][:3] == procedure_names[i][:3]  :  # batch procedures
                j += 1
                if j >= length:
                    break

            # batch_procedures = self.procedures[i:j]
            batch_procedures = [v for k, v in self.procedures.items() if k in procedure_names[i:j]]
            i=j
            yield  batch_procedures






class Procedure:
    def __init__(self,procedure_name=None,procedure_dict=None,procedure_value=None,default_num_calls=100):
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
    def inject(self,batch_calls):
        calls_start_instants_pd=np.ones_like(batch_calls[:,0])
        calls_end_instants_pd=np.ones_like(batch_calls[:,0])
        #approach 1
        # for call in batch_calls:
        #     for i in range(0,self.num_tks-1):
        #         if self.tks[i]<= call[0] and call[0] <self.tks[i+1]:
        #             calls_start_instants_pd.append(i)# if i-1 ==-1, call start is over started
        #             break
        #         i+=1
        #     for j in range(0,self.num_tks-1):
        #         if  self.tks[j]<=call[1] and call[1]<self.tks[j+1]:
        #             calls_end_instants_pd.append(j+1)
        #             break
        #         j+=1
        #
        # calls_instants_pd = np.array([calls_start_instants_pd,calls_end_instants_pd]).transpose([1,0])
        #approach 2
        for i in range(self.num_tks-1):
            mask =  (self.tks[i]<=batch_calls[:,0]) &( batch_calls[:,0]< self.tks[i+1])
            calls_start_instants_pd[mask] = i
            mask = (self.tks[i]<= batch_calls[:,1] )&( batch_calls[:,1]< self.tks[i+1])
            calls_end_instants_pd[mask] = i+1

        calls_instants_pd = np.array([calls_start_instants_pd,calls_end_instants_pd]).T[:-1]



        Ph = []
        for instant in calls_instants_pd:
            tmp = 1 - np.array(self.pds[instant[0]:instant[1]])
            ret = 1
            for item in tmp:
                ret *= item
            Ph.append(ret)
        return 1 - np.array(Ph),calls_instants_pd



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

