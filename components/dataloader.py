import pandas as pd
from path import Path
import random
import math
import numpy as np
import itertools
import datetime
import os
import networkx as nx
from utils.tool import time_stat,get_now
from components.AcTik import Tik
# m: 卫星数量,
# n:时刻数量,n-关键时刻数量(n- << n)

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
        print("\nDATA PRE-PROCCESSING ...")
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
                             delimiter=",|\t",
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
        print("\nDATA RE-LOADING")



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
        print('-> data re-load over')

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
        print("\nDATA ALIGNING")
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

        #return
        self.df_align = df_align

    def data_parse(self):
        print('\nDATA PARSING')
        self.__tiks_init()
        self.__get_positions()
        self.__get_acc2tk()
        self.__accs_init()
        print('-> parse over')


    def __tiks_init(self):
        '''
        算法数据准备, get tks
        args:
            self.df_align
            passes_log

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
        start = get_now()
        argsort = np.argsort(np.array(self.df_align[self.access_names].replace(np.nan,np.inf)))
        tk_mask1 = np.abs(argsort - np.concatenate([argsort[0].reshape(1,argsort.shape[1]),argsort[:-1]],0))>0

        argsort = np.argsort(np.array(self.df_align[self.access_names].replace(np.nan, -np.inf)))
        tk_mask2 = np.abs(argsort - np.concatenate([argsort[0].reshape(1, argsort.shape[1]), argsort[:-1]], 0)) > 0

        tk_mask = tk_mask1 +tk_mask2
        tk_mask_zip = tk_mask.sum(1)>0

        #进去出去都是 1,补上
        tk_mask_zip[0] = True
        tk_mask_zip[-1]=True


        # 1. passes tks
        passes_log_np = np.array(list(self.passes_log.values())).reshape([len(self.passes_log), 2])
        pass_tks =  list(np.array(list(self.passes_log.values())).reshape([len(self.passes_log)*2]))
        pass_tks =list(set(pass_tks))
        pass_tks.sort()

        max_tks = passes_log_np.max()
        min_tks = passes_log_np.min()

        pass_out_tks =  list(passes_log_np[:,1])
        pass_out_tks.sort()
        # 2. all tks #矫正一下离境时刻, 原来的离境时刻timestamp都大了1s,入境时间是对的.
        all_tks = list(self.df_align[tk_mask_zip].index)
        for i,tk in enumerate(all_tks):
            if tk-1 in  pass_out_tks:
                all_tks[i] -=1
                tk_mask_zip[tk-min_tks-1] = True
                # tk_mask_zip[tk-min_tks] = False




        all_tks_supremum = list(self.df_align[tk_mask_zip].index)
        #min tks 是最开始的全局时刻, 可能在0-24*3600 之间




        tiks = {}
        for tk in all_tks_supremum:
            row = self.df_align.loc[tk]
            ss = list(row[True ^ pd.isnull(row)].index)
            tik =Tik(tk)
            # tiks[tk]['pass_in'] = []
            # tiks[tk]['pass_out'] = []
            # tiks[tk]['inter']=[]

            if tk ==min_tks:
                #是否为首时刻
                [tik.addPass(addPassIn=si) for si in ss]

                pass
            elif tk ==max_tks:#末时刻
                [tik.addPass(addPassOut=si) for si in ss]
                pass
            else:#中间时刻
                for si in ss:
                    if pd.isnull(self.df_align[si][tk-1]):#前一个时刻,si为nan, si该时刻为pass in
                        tik.addPass(addPassIn=si)
                        # tik.passIn.add(si)
                    if pd.isnull(self.df_align[si][tk+1]):# 后一个时刻, si为nan, si该时刻为pass out
                        tik.addPass(addPassOut=si)

                for si,sj in itertools.combinations(ss,2):#任选两个,查看是否inter
                    if self.is_equal(si,sj,tk):
                        tik.addPass(addPassInter={si,sj})

            tik.rebuild()
            if tik.class_id=='O':
                pass
            else:
                tiks[tk]= tik



        #get
        # inter tks
        # pass in tks
        # pass out tks
        inter_tks = []
        passIn_tks = []
        passOut_tks = []
        all_tks = list(tiks.keys())

        for tk, tik in tiks.items():
            if len(tik.passInter) != 0:
                inter_tks.append(tk)
            if len(tik.passIn)!=0:
                passIn_tks.append(tk)
            if len(tik.passOut) != 0:
                passOut_tks.append(tk)

        inter_tks.sort()
        passIn_tks.sort()
        passOut_tks.sort()
        all_tks.sort()

    #returns
        self.inter_tks = inter_tks
        self.passIn_tks = passIn_tks
        self.passOut_tks = passOut_tks
        self.all_tks = all_tks
        self.tiks = tiks
    #
        cost = time_stat(start)

        print("-> tks init over, num of tks, and cost time:%.5f sec\n"%cost,
              "--> inter tks:{}\n".format(len(inter_tks)),
              "--> pass in tks:{}\n".format(len(passIn_tks)),
              "--> pass out tks:{}\n".format(len(passOut_tks)),
              "--> all tks:{}\n".format(len(all_tks)))

    def __accs_init(self):
        accs=[]
        si_names = self.access_names.copy()
        while len(si_names)>0:
            min_si = si_names[0]
            for si in si_names:
                if self.acc2tk[si][0] < self.acc2tk[min_si][0]:
                    min_si =si
            accs.append(min_si)
            si_names.remove(min_si)
        self.accs = accs

    def __get_positions(self):# 迁移到drawer去
        position = {}
        for acc in self.access_names:
            (tk_in, tk_out) = self.passes_log[acc][0]  # 这里只能允许一个星过境一次, 不够一般性
            y = self.df_align.query(" time >={} and time<={}".format(tk_in, tk_out))[acc].max()
            x = math.ceil(((tk_in + tk_out) / 2 - self.all_tks[0]) / 10)
            position[acc] = (x, y)
        self.position = position



    def __get_acc2tk(self):
        '''
        根据tiks, 输出acc2tk dict
        :return:
        '''
        # self.tikss
        # self.
        #O(mn-)
        acc2tk={}
        for si in self.access_names: #O(m)
            acc2tk[si]=[]
            if si =='s3217':
                pass
            for tk ,tik in self.tiks.items():#O(n-)
                if tik.is_inInter(si):
                    acc2tk[si].append(tk)

                if si in tik.getPass('In'):
                    acc2tk[si].append(tk)

                elif si in tik.getPass('Out'):
                    acc2tk[si].append(tk)

            acc2tk[si] = list(set(acc2tk[si]))# 去重
            acc2tk[si].sort()




        self.acc2tk = acc2tk





    def getIntersection(self,tk):
        return self.tiks[tk]['inter']

    def getInterTk(self,si,sj):
        # print(si,sj)

        for tk in self.inter_tks:
            inters = self.tiks[tk].getPass('Inter')
            for inter in inters:
                if inter == set([si,sj]):
                    return tk

        return None



