import pandas as pd
from path import Path
import random
import math
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



        access = dict(to_df['access'].value_counts())
        access_names = list(access.keys())

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
        df = pd.read_csv(self.dump)
        print(df.describe())
        pass

    def filter(self):
        '''
        只选取部分access, 部分时间的数据, 用来简单测试
        :return:
        '''
        pass



