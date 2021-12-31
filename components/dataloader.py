import pandas as pd
from path import Path
import random
import math
import numpy as np
import itertools
import datetime
import os
import networkx as nx

class AerDataset:
    def __init__(self,config):
        pass
        self.path = Path(config['path'])
        self.files=[]
        for file in config['files']:
            self.files.append(self.path/file)



        self.assigned_units = config['assigned_units']
        self.access_portion = config['access_portion']
        self.time_portion = config['time_portion']
        self.time_len = config['time_len']
        self.access_len = 0

        #time_dir = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.out_dir_path = Path(config['dump_path']) #/ time_dir
        self.out_dir_path.mkdir_p()
        self.dump_file = self.out_dir_path / "{}-{}.csv".format(config['dump_stem'],config['random_seed'])


        self.data_description={}


        self.config=config

    def is_equal(self, s1, s2, tk):
        try:
            a, b = tuple(self.df_align.query(" time >={} and time <={}".format(tk - 1, tk))[s1])
            c, d = tuple(self.df_align.query(" time >={} and time <={}".format(tk - 1, tk))[s2])
        except:
            print(s1,s2,tk)
        if (a > c and b < d) or (a < c and b > d):
            return True
        else:
            return False
    def description(self):
        print("DATA DESCRIPTION")
        for k,v in self.data_description.items():
            print("-> {},:{}",k,v)

    def passes_stat(self):
        for key, value in self.passes_log.items():
            print("{}: {}".format(key, len((value))))

    def get_sublines(self, access_name, value_names, with_time=False):
        for start, end in self.passes_log[access_name]:
            if with_time:

                sub_df = self.df.query("access == '{}' and time >={} and time <={}".format(access_name, start, end))[
                    ['time'] + value_names]
            else:
                sub_df = self.df.query("access == '{}' and time >={} and time <={}".format(access_name, start, end))[
                    value_names]

            yield np.array(sub_df)

    def data_append_value(self,df):
        #append the data to df
        # freq = 12e9#12GHz
        # opt_v = 3e8#300k
        # wave_lambda = 0.025# m
        # Pr = (wave_lambda)**2/( 4*math.pi*df['Range (km)'].astype(float)*1e3)**2
        # data = np.log10(Pr*1000)
        data = df['Range (km)'].max() - df['Range (km)']

        value = pd.Series(name='Max - Range (km)',data=data)
        df =pd.concat([df,value],axis=1)
        return df


    def data_prep(self,data_prep_config):
        if not bool(data_prep_config['Do']):
            return
        print("DATA PRE-PROCCESSING ...")
        #input
        time_portion = self.time_portion
        access_portion = self.access_portion
        time_len = self.time_len


        names = ['access','idx', 'time', 'value']

        # names_type = {'access':str,'idx':np.int32, 'time':np.float32}#, 'value':np.float32}
        not_dup_columns=['value']
        output_columns =   ['access','idx','time']+self.assigned_units

        dfs = []
        print("-> regexing ...")
        for idx,(item,unit) in enumerate(zip(self.files,self.assigned_units)):# 载入距离, 速度, 等不同value其他文件


            df = pd.read_csv(filepath_or_buffer=item,
                             names=names,
                             encoding='utf-16',
                             delimiter=",",
                             # converters=names_type
                             )

            if unit=="Range (km)":
                df['access'] = df['access'].str.replace(r'(.*)-(.*)', r'\1', regex=True)
                df['access'] = df['access'].str.replace(r'Range',r'')
                df['access'] = df['access'].str.replace(r'\(.*?\)',r'')
                df['access'] = df['access'].str.replace(r'\s+', r'', regex=True)



            elif unit=="RangeRate (km/sec)":
                df['access'] = df['access'].str.replace(r'(.*)-(.*)', r'\1', regex=True)
                df['access'] = df['access'].str.replace(r'\(*\)',r'')
                df['access'] = df['access'].str.replace(r'(.*)/(.*)', r'', regex=True)
                df['access'] = df['access'].str.replace(r'RangeRatekmsec',r'')
                df['access'] = df['access'].str.replace(r'\s+', r'', regex=True)

            df['access'] = df['access'].str.replace('term-To-','')
            if idx>0:
                df=df[not_dup_columns]


            df=df.rename(columns={'value': unit})


            dfs.append(df)

        print('-> regex over')


        df = pd.concat(dfs,axis=1)
        df=df[output_columns]


        df.replace(' ', 'nan', inplace=True)
        df['time'] = df['time'].astype(float)
        df = df.query('time%1==0')
        df['time'] = df['time'].astype('int')
        df['Range (km)'] = df['Range (km)'].astype('float64')

        access_dict = dict(df['access'].value_counts())
        access_names = list(access_dict.keys())
        # access selectoin
        random.seed(data_prep_config['random_seed'])
        random.shuffle(access_names)

        start = math.floor(len(access_names) * access_portion[0])
        end = math.ceil(len(access_names) * access_portion[1])
        access_names = access_names[start:end]


        start = math.floor(time_len*time_portion[0])
        end = math.ceil(time_len*time_portion[1])

        df = df.query('{} in access and time >= {} and time <={}  '.format(access_names,start,end))

        print("-> total time:{}s, selected time:{}s".format(time_len,end-start))

        #data type set



        df = self.data_append_value(df)


        df.to_csv(self.dump_file, index=False)



    def load(self):
        '''
        load df and compute passes log
        :return:
        '''
        print("DATA RE-LOADING")



        df = pd.read_csv(self.dump_file)
        df['time'] = np.array(df['time']).astype(np.int32)
        df['access'] = np.array(df['access']).astype(str)


        print(df.describe())
        access_names = list(dict(df ['access'].value_counts()).keys())
        access_names.sort()


        # 每个access 可能有多次过境, 将timestamp记录一下, 后面绘图, 或者数据处理都需要用
        passes_log={}
        for name in access_names:
            time_seriers = df.query("access == '{}'".format(name))['time']
            #相或得到time start or end mask
            fwd = time_seriers.diff(-1)
            pwd = time_seriers.diff(1)
            time_np = np.array(time_seriers)
            inter_mask = np.array(((abs(fwd)>1 ) + (abs(pwd)>1)))
            if time_np[0]%1 ==0:
                inter_mask[0] =True
            else:
                inter_mask[1]=True

            if time_np[-1]%1 ==0:
                inter_mask[-1]=True
            else:
                inter_mask[-2]=True

            time_stamp = np.array( time_np[inter_mask]) #连续的两个代表着开始和结束
            cnt = 0
            passes_log[name]=[]
            while cnt <len(time_stamp):# 如果有2*n个数, 就说明过境n次
                passes_log[name].append(
                    (math.ceil(time_stamp[cnt]),math.floor(time_stamp[cnt+1]))# 可能有多次过境, 所以是append
                )
                cnt=cnt+2

        data_description={}
        data_description['access_num']=len(access_names)
        #returns
        self.df = df
        self.passes_log = passes_log
        self.access_names = access_names
        self.data_description = data_description




    def data_align(self):
        '''
        将prep的数据载入后, 需要通过此函数构建参差表, 具体见readme
        hidden imput:
            self.passes_log
            self.access_names
            self.df
        hidden method
            self.get_sublines
        :param config:
        :return:
        '''
        # for G opt
        print("DATA ALIGNING")
        algorithm_base = self.config['algorithm_base']

        time_access_names = self.access_names.copy()
        time_access_names.insert(0, 'time')
        df_align = pd.DataFrame(columns=time_access_names)


        time_min = math.ceil(self.df['time'].min())
        time_max = math.floor(self.df['time'].max())
        df_align_time = pd.Series(name='time',data=np.linspace(start=int(time_min), stop=int(time_max), num=int(time_max - time_min + 1)))

        df_align['time'] = np.array(df_align_time).astype(np.int32)
        df_align.set_index(['time'], inplace=True)
        # get time lines
        for access_name in self.access_names:
            for line,(start,end) in zip(self.get_sublines(access_name,algorithm_base),self.passes_log[access_name]):
                sub_df = self.df.query("access == '{}' and time >={} and time <={}".format(access_name,start,end))[['time']+algorithm_base]

                time_mask =(df_align.index >= start) *  (df_align.index <= end)
                try:
                    df_align[access_name][time_mask] =list(sub_df[algorithm_base[0]])
                    # without nparray, the mapping will fail due to the series default mapping
                except:
                    print("-> wrong in access: {}".format(access_name))
        print('-> data re-load over')

        #return
        self.df_align = df_align

    def pre_alg(self):
        '''
        算法数据准备, 得到邻接表

        args:
            self.df_align

        :return:
            self.all_tks
            self.acc2tk
            self.tk2acc
        '''
        #如何快速检测函数交点
        # 1. 对每行排序, 并输出arg序号值
        # 2. 对序号值差分
        # 3. 如果有绝对值大或等于1的, 就说明这行出现函数交点
        # 4. 为了
        argsort = np.argsort(np.array(self.df_align[self.access_names].replace(np.nan,np.inf)))
        tk_mask1 = np.abs(argsort - np.concatenate([argsort[0].reshape(1,argsort.shape[1]),argsort[:-1]],0))>0

        argsort = np.argsort(np.array(self.df_align[self.access_names].replace(np.nan, -np.inf)))
        tk_mask2 = np.abs(argsort - np.concatenate([argsort[0].reshape(1, argsort.shape[1]), argsort[:-1]], 0)) > 0

        tk_mask = tk_mask1 +tk_mask2
        tk_mask_zip = tk_mask.sum(1)>0


        # 1. passes tks
        passes_log_np = np.array(list(self.passes_log.values())).reshape([len(self.passes_log), 2])
        pass_tks =  list(np.array(list(self.passes_log.values())).reshape([len(self.passes_log)*2]))
        pass_tks =list(set(pass_tks))
        pass_tks.sort()
        #矫正一下离境时刻, 原来的离境时刻timestamp都大1s,入境时间是对的.


        # 2. all tks
        all_tks = list(self.df_align[tk_mask_zip].index)
        for i,tk in enumerate(all_tks):
            if tk in  list(passes_log_np[:,1]+1):
                all_tks[i] -=1

        #min tks 是最开始的全局时刻, 可能在0-24*3600 之间
        max_tks = passes_log_np.max()
        min_tks = passes_log_np.min()
        all_tks.insert(0,min_tks)
        all_tks.append(max_tks)



        # 3. inter tks 即函数交点
        set_all_tks = set(all_tks)
        set_pass_tks = set(pass_tks)
        inter_tks = list(set_all_tks.difference(set_pass_tks))# bug ,如果恰巧有interfaces tk 和出入境tk同步, 就问题了
        for tk in pass_tks[1:-1]:
            accesses = self.df_align.loc[tk][self.df_align.loc[tk] > 0].index
            for acc1,acc2 in itertools.combinations(accesses,2):
                if self.is_equal(acc1,acc2,tk):
                    inter_tks.append(tk)
                    print(tk)

        inter_tks.sort()#19



        # total tks 是包括了函数交点和 函数起点终点(过境时刻, 离境时刻)
        all_tks2 = list(set_all_tks|set_pass_tks)
        all_tks2 = [int(item) for item in all_tks2]
        all_tks2.sort()#49

        inter_tk_mask = np.array(inter_tks) - min_tks

        query_table = pd.DataFrame(argsort * tk_mask).loc[inter_tk_mask]


        # get inter_tk2access, query what the access at this timestamp(tk), and the timestamp in inter_timestamp (access function intersect)
        inter_tk2access={}
        for index,row in query_table.iterrows():
            inter_tk2access[index+min_tks]=[]
            acc_num = list(row[row > 0])
            for num in acc_num:
                inter_tk2access[index+min_tks].append(self.access_names[num])

        #  same with after one, but the timestamp in star/end timestamps
        ed_tk2access={}
        for k,passes in self.passes_log.items():
            for start,end in passes:
                if start not in ed_tk2access.keys():
                    ed_tk2access[start] =[]
                ed_tk2access[start].append(k)

                if end not in ed_tk2access.keys():
                    ed_tk2access[end] = []
                ed_tk2access[end].append(k)

        tk2acc = inter_tk2access.copy()
        tk2acc.update(ed_tk2access)
        acc2tk={}
        tmp_df = self.df_align.query('time in {} '.format(all_tks))
        for col_name,col_value in tmp_df.iteritems():
            if col_name =='time':
                continue
            acc2tk[col_name]= list(tmp_df.index[col_value>0])
        # then, the acc2tk[acc] is include all tks at the function coverage, thus we should filter out the point that did not on the function
        for acc,tks in acc2tk.items():

            cnt = 0
            while cnt < len(acc2tk[acc]):

                if acc not in tk2acc[acc2tk[acc][cnt]]:
                    acc2tk[acc][cnt] = -1
                cnt+=1
                    # break
            while -1 in acc2tk[acc]:
                acc2tk[acc].remove(-1)

        self.inter_tks = inter_tks
        self.acc2tk = acc2tk
        self.tk2acc = tk2acc
        self.all_tks = all_tks
        self.all_tks.sort()
        for access in self.access_names:
           self.acc2tk[access].sort()
        for tk in all_tks:
            self.tk2acc[tk].sort()

        position = {}
        for acc in self.access_names:
            (tk_in, tk_out) = self.passes_log[acc][0]  # 这里只能允许一个星过境一次, 不够一般性
            y = self.df_align.query(" time >={} and time<={}".format(tk_in, tk_out))[acc].max()
            x = math.ceil(((tk_in + tk_out) / 2 - self.all_tks[0]) / 10)
            position[acc] = (x, y)
        self.position = position









