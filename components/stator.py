import numpy as np
import matplotlib.pyplot as plt
class Stator:
    def __init__(self,data):
        self.data = data
        self.duration = (data.df_align.index[0],data.df_align.index[-1])
        pass
    def data_stat(self):
        print('\n-> DATA STAT')

        los = np.sum(self.data.df_align>=0,axis=1)
        std_los = los.std()
        hist,bins = np.histogram(los, bins=40, range=(0, 39), density=True)
        avg_los = np.linspace(start=0, stop=39, num=40)*hist
        avg_los = avg_los.sum()
        avg_los = los.mean()
        total_pass_num = 0
        durations=[]
        for sat,log in self.data.crossLog.items():
            total_pass_num+=len(log)
            for item in log:
                durations.append(item[-1]-item[0])

        # time_end - time_start
        period = self.data.df_align.index[-1] - self.data.df_align.index[0]
        print('--> pass num:{} for {} second, {:.2f} passed per avg hour'.format(total_pass_num,period,total_pass_num/period*3600))
        print('--> avg passed in:{:.2f}, std:{:.2f}, avg_time_los:{:.2f}'.format(avg_los,std_los,np.mean(durations)))
        print('--> durations:{}'.format(durations))

    def solution_stat(self,final_solution,final_value,algorithm):

        print("\n-> PROBLEM STAT")
        carrier ={}

        final_hop = len(final_solution)
        # 找到每个解
        # 的分量的交点,后面移动到 data prep 更合适

        total_time = self.duration[1] - self.duration[0]

        # 断路时间
        disconn_time = 0
        disconn_times = 0
        for i in range(1, len(final_solution) - 1):
            if final_solution[i] == 'none' and final_solution[i - 1] != 'none' and final_solution[i + 1] != 'none':
                disconn_time += (self.data.getInterTk('none', final_solution[i + 1]) - self.data.getInterTk(
                    final_solution[i - 1], 'none'))
                disconn_times += 1

        print('--> solution stat' +
              '\n--> final solution:\t{}'.format(final_solution) +
              # '\n--> opt value:\t{:.2f}'.format(final_opt_value)+
              '\n--> handover times: \t{}, disconn times:{}'.format(final_hop, disconn_times) +
              '\n--> avg duration:\t{:.2f}(s).'.format(total_time / final_hop) +
              '\n--> avg alg base:\t{:.2f}.'.format(np.mean(final_value)) +
              '\n--> total time:\t{:.2f}(s), disconn time:\t{:.2f}({:.2f}%)'.format(total_time, disconn_time,
                                                                                    100 * disconn_time / total_time)

              )
        carrier['algorithm'] = algorithm
        carrier['final_solution'] = final_solution
        carrier['handover times']= final_hop-1
        carrier['avg duration'] = total_time / (final_hop-1)
        carrier['total time'] = int(total_time)
        carrier['duration'] = [int(self.duration[0]),int(self.duration[1])]
        carrier['avg signal'] = np.mean(final_value)
        return carrier
