import pandas as pd
from path import Path
import random
import math
import numpy as np
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
        self.dump = Path(config['dump_path'])/config['dump_stem']

    def data_prep(self):


        input_columns = ['access','idx', 'time', 'value']
        universal_columns = ['access', 'units','idx', 'time', 'value']
        not_dup_columns=['value']
        output_columns =   ['access','idx','time']+self.assigned_units

        # range = pd.read_csv(self.range,header=None,encoding='utf-16',delimiter="\t")
        dfs = []
        units = []
        print("--> regexing ...")
        for idx,(item,unit) in enumerate(zip(self.files,self.assigned_units)):


            df = pd.read_csv(item,header=None,encoding='utf-16',delimiter="\t")
            # df2= df.replace(r'/Range (km)', r'xx', regex=True)

            df.columns = input_columns
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


            if idx>0:
                df=df[not_dup_columns]


            df=df.rename(columns={'value': unit})


            dfs.append(df)

        print('--> regex over')


        to_df = pd.concat(dfs,axis=1)
        to_df=to_df[output_columns]



        access_dict = dict(to_df['access'].value_counts())
        access_names = list(access_dict.keys())

        # access selectoin
        random.shuffle(access_names)
        self.access_len = len(access_names)
        start = math.floor(self.access_len*self.access_portion[0])
        end = math.ceil(self.access_len*self.access_portion[1])
        access_names = access_names[start:end]
        print("--> access total num:{}, selected num{}".format(self.access_len,len(access_names)))


        #time selection
        start = math.floor(self.time_len*self.time_portion[0])
        end = math.ceil(self.time_len*self.time_portion[1])
        to_df.replace(' ','nan',inplace=True)
        to_df[['time']] = to_df[['time']].astype(float)
        ret_df = to_df.query('{} in access and time >= {} and time <={}  '.format(access_names,start,end))

        print("--> time total len:{}, selected len:{}".format(self.time_len,end-start))

        ret_df.to_csv(self.dump,index=False)

    def load(self):
        print("-> loading")

        self.df = pd.read_csv(self.dump)
        print(self.df.describe())

        self.access = list(dict(self.df ['access'].value_counts()).keys())
        self.access.sort()
        print("--> access num:{}".format(len(self.access)))
        # 每个access 可能有多次过境, 将timestamp记录一下, 后面绘图, 或者数据处理都需要用
        self.passes_log={}
        for name in self.access:
            time_ser = self.df.query("access == '{}'".format(name))['time']
            #相或得到time start or end mask
            fwd = time_ser.diff(-1)
            pwd = time_ser.diff(1)
            time_np = np.array(time_ser)
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
            self.passes_log[name]=[]
            while cnt <len(time_stamp):# 如果有2*n个数, 就说明过境n次
                self.passes_log[name].append(
                    [math.ceil(time_stamp[cnt]),math.floor(time_stamp[cnt+1])]
                )
                cnt=cnt+2






    def __getitem__(self, access):

        ret = self.df.query("access=='{}'".format(access))
        return ret
    def data_recons(self,config):
        '''
        将prep的数据载入后, 需要通过此函数构建参差表, 具体见readme

        :param config:
        :return:
        '''
        # for G opt
        pass

        cols = list(self.access)
        cols.sort()
        cols.insert(0, 'time')
        df_recons = pd.DataFrame(columns=cols)



        time_min = math.ceil(self.df['time'].min())
        time_max = math.floor(self.df['time'].max())
        df_recons_time = pd.Series(name='time',data=np.linspace(start=int(time_min), stop=int(time_max), num=int(time_max - time_min + 1)))

        df_recons['time'] = np.array(df_recons_time).astype(np.int32)
        for idx,access_name in enumerate(cols[1:]):
            for start,end in self.passes_log[access_name]:
                sub_df = self.df.query("access == '{}' and time >={} and time <={}".format(access_name,start,end))[config['base']]

                time_mask =(df_recons['time'] >= start) *  (df_recons['time'] <= end)
                try:
                    df_recons[access_name][time_mask] =np.array(sub_df[config['base'][1]]) # without nparray, the mapping will fail due to the series default mapping
                except:
                    print("--> wrong in access: {}".format(access_name))
        print('over')



    def filter(self):
        '''
        只选取部分access, 部分时间的数据, 用来简单测试
        :return:
        '''
        pass



