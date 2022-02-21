import numpy as np
class Stator:
    def __init__(self,data):
        self.data = data
        pass
    def data_stat(self):
        print('-> DATA STATOR')

        los = np.sum(self.data.df_align>=0,axis=1)
        std_los = los.std()
        hist,bins = np.histogram(los, bins=10, range=(0, 9), density=True)
        avg_los = np.linspace(start=0, stop=9, num=10)*hist
        avg_los = avg_los.sum()
        total_pass_num = 0
        durations=[]
        for sat,log in self.data.crossLog.items():
            total_pass_num+=len(log)
            for item in log:
                durations.append(item[-1]-item[0])

        period = self.data.all_tks[-1]-self.data.all_tks[0]
        print('--> pass num:{} for {} second, {:.2f} passed per avg minute'.format(total_pass_num,period,total_pass_num/period*3600))
        print('--> avg passed in:{:.2f}, std:{:.2f}, avg_time_los:{:.2f}'.format(avg_los,std_los,np.mean(durations)))
        print('--> durations:{}'.format(durations))
        pass