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
import threading
import errno
# m: 卫星数量,
# n:时刻数量,n-关键时刻数量(n- << n)
from tqdm import tqdm
class AerDataset:
    def __init__(self,config,terminal=False):
        pass
        self.path = Path(config['path'])

        self.terminal = terminal

        #data prep
        self.doOrNot = bool(config['Do'])
        self.assigned_units = config['assigned_units']
        self.access_portion = config['access_portion']
        self.time_duration = config['time_duration']
        self.files = []
        for file in config['files']:
            self.files.append(self.path / file)

        self.dump_file = self.path / "{}.csv".format(config['dump_stem'])

        #data align
        self.algorithm_base = config['algorithm_base']
        self.input_metrics = config['input_metrics']
        self.output_metrics = config['output_metrics']
        self.access_len = 0

        #time_dir = datetime.datetime.now().strftime("%m-%d-%H:%M")



        self.data_description={}

    def get_time(self,t):
        return self.df_align.index[t]

    def is_equal(self, s1, s2, tk):
        try:
            a, b = tuple(self.df_align.query(" time >={} and time <={}".format(tk - 1, tk))[s1])
            c, d = tuple(self.df_align.query(" time >={} and time <={}".format(tk - 1, tk))[s2])
        except IOError:

            print("equal func wrong at {},{},{}".format(s1,s2,tk))
            return
        if (a > c and b < d) or (a < c and b > d):
            return True
        else:
            return False
    def description(self):
        print("DATA DESCRIPTION")
        for k,v in self.data_description.items():
            print("-> {},:{}",k,v)

    def passes_stat(self):
        for key, value in self.crossLog.items():
            print("{}: {}".format(key, len((value))))

    def get_sublines(self, access_name, value, with_time=False):
        '''
        hidden param
            self.crossLog
            self.df
        called by
            self.load_align
            out
        :param access_name:
        :param value_names:
        :param with_time:
        :return:
        '''
        for start, end in self.crossLog[access_name]:
            if with_time:

                sub_df = self.df.query("access == '{}' and time >={} and time <={}".format(access_name, start, end))[
                    ['time'] + value]
            else:
                sub_df = self.df.query("access == '{}' and time >={} and time <={}".format(access_name, start, end))[
                    value]

            yield np.array(sub_df)

    def data_append_value(self,df,input_metrics,output_metrics):
        #append the data to df
        # freq = 12e9#12GHz
        # opt_v = 3e8#300k
        # wave_lambda = 0.025# m
        # Pr = (wave_lambda)**2/( 4*math.pi*df['Range (km)'].astype(float)*1e3)**2
        # data = np.log10(Pr*1000)
        #
        f       = 12 #Ghz
        log_2_10 = 3.321928094887362
        lg12 = 1.0791812
        EIRP  = 35     #dBW
        GT    = 8.53   #dBi/K
        Gt    = 40 #dB
        Gr    = 40 #dB
        k     = -228.6 #dBW/K
        L     = 30     #dB
        B     = 2      #GHz
        # d = df['Range (km)'].max() - df['Range (km)']
        d = df[input_metrics]
        d = d['Range (km)']
        Lf = 92.45 + 20 * np.log10(d) + 20 * np.log10(f)

        for item in output_metrics:
            if item =='Cno(dBHz)':

                Cno = EIRP +GT -k-Lf-100

                Cno = Cno.rename( item)
                df = pd.concat([df, Cno], axis=1)
            if item =='CT(dBW/K)':
                CT = EIRP + GT -Lf

                CT = CT.rename(item)
                df = pd.concat([df, CT], axis=1)
            if item == 'Pr(dBW)':
                Pr = EIRP + Gr - Lf+100
                Pr = Pr.rename(item)
                df = pd.concat([df, Pr], axis=1)
            if item =='(Ct)Gbps':
                pass
            if item =='CT(W/K)' and 'CT(dBW/K)' in df.columns:
                pass



        return df


    def data_prep(self):
        '''
        :param :
            self.time_portion
            self.access_portion
            self.assigned_units
            self.files
            self.random_seed
        :return:
            a data frame file
        '''
        if self.doOrNot is False:
            return
        if self.terminal:
            print("\nDATA PRE-PROCCESSING ...")
        #input

        files = self.files
        assigned_units = self.assigned_units

        input_metrics = self.input_metrics
        output_metrics = self.output_metrics

        names = ['access','idx', 'time', 'value']

        # names_type = {'access':str,'idx':np.int32, 'time':np.float32}#, 'value':np.float32}
        not_dup_columns=['value']
        output_columns =   ['access','idx','time']+assigned_units

        dfs = []
        if self.terminal:
            print("-> regexing ...")
        for idx,(item,unit) in enumerate(zip(files,assigned_units)):# 载入距离, 速度, 等不同value其他文件


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

        if self.terminal:

            print('-> regex over')


        df = pd.concat(dfs,axis=1)
        df=df[output_columns]


        df.replace(' ', 'nan', inplace=True)
        df['time'] = df['time'].astype(float)
        df = df.query('time%1==0')
        df['time'].loc[:] = df['time'].loc[:].astype('int')
        df['Range (km)'].loc[:] = df['Range (km)'].loc[:].astype('float64')



        df = self.data_append_value(
            df,
            input_metrics=input_metrics,
            output_metrics=output_metrics
        )
        if self.terminal:
            print('-> data saved at:{}'.format(self.dump_file))
        df.to_csv(self.dump_file, index=False)



    def load_align(self,random_seed=None):
        '''
        :input:
            self.dump_file
        :return:
            self.df = df #for data align
            self.crossLog = crossLog
            self.access_names = access_names
            self.data_description = data_description

        '''
        if self.terminal:
            print("\nDATA RE-LOADING AND ALIGN")

        time_duration = self.time_duration
        access_portion = self.access_portion

        df = pd.read_csv(self.dump_file)
        algorithm_base = self.algorithm_base
        output_metrics = self.output_metrics
        df['time'] = np.array(df['time']).astype(np.int32)
        df['access'] = np.array(df['access']).astype(str)

        if self.terminal:
            print(df[['Range (km)']+output_metrics].describe())

        if random_seed:
            random.seed(random_seed)

        # time selection
        time_len = int(df['time'].max())
        start=0
        end= 0
        if type(time_duration)==list:
            start = math.floor(time_len * time_duration[0])
            end = math.ceil(time_len * time_duration[1])
        elif type(time_duration) == int:
            start = random.randrange(0,time_len-time_duration)
            end = start+time_duration
        else:
            print('ERROR IN TIME DURATION')
            exit(-1)
        df = df.query('time >= {} and time <={}  '.format(start, end))


        # access selectoin
        access_dict = dict(df['access'].value_counts())
        access_names = list(access_dict.keys())
        random.shuffle(access_names)
        start = math.floor(len(access_names) * access_portion[0])
        end = math.ceil(len(access_names) * access_portion[1])
        access_names = access_names[start:end]
        df = df.query('access in {}  '.format(access_names))




        # data type set


        # 每个access 可能有多次过境, 将timestamp记录一下, 后面绘图, 或者数据处理都需要用
        crossLog={}
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
            crossLog[name]=[]
            while cnt <len(time_stamp):# 如果有2*n个数, 就说明过境n次
                crossLog[name].append(
                    (math.ceil(time_stamp[cnt]),math.floor(time_stamp[cnt+1]))# 可能有多次过境, 所以是append
                )
                cnt=cnt+2

        self.crossLog = crossLog
        self.df = df

        #data align
        time_access_names = access_names.copy()
        time_access_names.insert(0, 'time')
        df_align = pd.DataFrame(columns=time_access_names)

        time_min = math.ceil(df['time'].min())
        time_max = math.ceil(df['time'].max())

        df_align_time = pd.Series(name='time', data=np.linspace(start=int(time_min), stop=int(time_max),
                                                                num=int(time_max - time_min + 1)))

        df_align['time'] = np.array(df_align_time).astype(np.int32)
        df_align.set_index(['time'], inplace=True)
        # get time lines
        for access_name in access_names:
            for line, (start, end) in zip(self.get_sublines(access_name, algorithm_base), crossLog[access_name]):
                sub_df = df.query("access == '{}' and time >={} and time <={}".format(access_name, start, end))[
                    ['time'] + algorithm_base]

                time_mask = (df_align.index >= start) * (df_align.index <= end)
                try:
                    df_align[access_name][time_mask] = list(sub_df[algorithm_base[0]])
                    # if go wrong here, check the original access file,
                except :

                    print("-> wrong in access: {}, check the original file".format(access_name))

        # returns
        data_description = {}
        data_description['access_num'] = len(access_names)
        data_description['time_min'] = time_min
        data_description['time_max'] = time_max
        self.data_description = data_description

        self.access_names = access_names
        self.df_align = df_align





    def data_parse(self):
        if self.terminal:
            print('\n-> DATA PARSING...')
        start = get_now()

        self.__tiks_init()
        self.__get_positions()
        self.__get_acc2tk()
        self.__accs_init()

        time_stat(start)

    def make_tik(self,all_tks_supremum,tiks ,start,stop):


        for i in  range(start,stop):
            tk = all_tks_supremum[i]
            pass
            if tk in tiks.keys():
                return

        # for tk in tqdm( all_tks_supremum):#短板, 时间过长

            row = self.df_align.loc[tk]
            ss = list(row[True ^ pd.isnull(row)].index)
            tik =Tik(tk)
            # tiks[tk]['pass_in'] = []
            # tiks[tk]['pass_out'] = []
            # tiks[tk]['inter']=[]


            for si in ss:
                if pd.isnull(self.df_align[si][tk-1]):#前一个时刻,si为nan, si该时刻为pass in
                    tik.addPass(addPassIn=si)
                    # tik.passIn.add(si)
                if pd.isnull(self.df_align[si][tk+1]):# 后一个时刻, si为nan, si该时刻为pass out
                    tik.addPass(addPassOut=si)

            for si,sj in itertools.combinations(ss,2):#任选两个,查看是否inter
                if self.is_equal(si,sj,tk):
                    if len(tik.getPass('Inter'))==0 : #是首个inter点
                        tik.addPass(addPassInter={si,sj})
                        continue#for
                    if tik.is_inInter(si) is False and tik.is_inInter(sj) is False:# 没有元素存在list里
                        tik.addPass(addPassInter={si,sj})
                        continue#for

                        #双方中有一方, 存在list的任意元素中(set), {si,sj}就并到set[i]中
                    i = 0
                    while not {si,sj}&tik.getPass('Inter')[i]:
                        i+=1
                    tik.getPass('Inter')[i]|={si,sj}

            tik.rebuild()
            if tik.class_id=='O':
                pass
            else:
                tiks[tk]= tik


    def __tiks_init(self):
        '''
        算法数据准备, get tks
        args:
            self.df_align
            crossLog

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

        #进去出去都是 1,补上
        tk_mask_zip[0] = True
        tk_mask_zip[-1]=True


        # 1. passes tks
        crossLog_np = np.array(list(self.crossLog.values())).reshape([len(self.crossLog), 2])
        pass_tks =  list(np.array(list(self.crossLog.values())).reshape([len(self.crossLog)*2]))
        pass_tks =list(set(pass_tks))
        pass_tks.sort()

        max_tks = crossLog_np.max()
        min_tks = crossLog_np.min()

        pass_out_tks =  list(crossLog_np[:,1])
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
        if self.terminal:
            print('--> all tks sup:{}'.format(len(all_tks_supremum)))



        tiks = {}
        #边界先处理
        for tk in [min_tks,max_tks]:
            row = self.df_align.loc[tk]
            ss = list(row[True ^ pd.isnull(row)].index)
            tik = Tik(tk)
            [tik.addPass(addPassIn=si) for si in ss]
            tik.rebuild()
            if tik.class_id=='O':
                pass
            else:
                tiks[tk]= tik

        # 多线程处理最耗时部分
        threads = []
        if len(all_tks_supremum)<100:
            pass
            t = threading.Thread(target=self.make_tik, args=(all_tks_supremum, tiks, 1, len(all_tks_supremum)-1))
            t.start()
            threads.append(t)
        else:
            each_thread_carry = 10
            for n in tqdm(range(1,len(all_tks_supremum),each_thread_carry)):
                if n +each_thread_carry<=len(all_tks_supremum):
                    stop = n + each_thread_carry
                else:
                    stop = len(all_tks_supremum)
                t = threading.Thread(target=self.make_tik,args=(all_tks_supremum,tiks,n,stop))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

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
        if self.terminal:

            print("--> tks init over, num of tks:"
                  "\n--> inter tks:{}".format(len(inter_tks)),
                  "\n--> pass in tks:{}".format(len(passIn_tks)),
                  "\n--> pass out tks:{}".format(len(passOut_tks)),
                  "\n--> all tks:{}".format(len(all_tks)))

    def __accs_init(self):
        accs=[]
        si_names = self.access_names.copy()
        while len(si_names)>0:
            min_si = si_names[0]
            for si in si_names:
                try:
                    if self.acc2tk[si][0] < self.acc2tk[min_si][0]:
                        min_si =si
                except:
                    print('error',si)
            accs.append(min_si)
            si_names.remove(min_si)
        self.accs = accs

    def __get_positions(self):# 迁移到drawer去
        position = {}
        for acc in self.access_names:
            (tk_in, tk_out) = self.crossLog[acc][0]  # 这里只能允许一个星过境一次, 不够一般性
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







    def getInterTk(self,si,sj):
        # print(si,sj)
        if si =='none' and sj is not 'none':
            return self.acc2tk[sj][0]
        if  si is not 'none' and sj =='none':
            return self.acc2tk[si][-1]

        for tk in self.inter_tks:
            inters = self.tiks[tk].getPass('Inter')
            for inter in inters:
                if {si,sj}&inter =={si,sj}:
                    return tk



