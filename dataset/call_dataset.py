import numpy as np
import errno
from collections import Counter
import matplotlib.pyplot as plt
class CallDataset:
    def __init__(self,start_end_list):
        self.start_end_dict=dict(Counter(start_end_list))
        self.mus = [100,200,300]  #avg call durations
        np.random.seed(1)
    def call_test(self,start,end,num=100):
        print(start,end)
        arr = [[74700, 74900], [74710, 75000], [74900, 75100],[74700,74720],[75150,75160]]
        st1 = sorted(arr, reverse=False, key=lambda x: x[1]-x[0])
        st1 = np.array(st1)

        st2 = sorted(st1, reverse=False, key=lambda x: x[0])
        st2 = np.array(st2)

        yield st2

    def get_batch_calls(self):
        num_calls = 500
        time_step=10

        start_end_list = list(self.start_end_dict.keys())

        for s,e in start_end_list:
            this_duration = e-s
            Ex = this_duration/2
            lab = 1 /Ex
            X = np.linspace(start=time_step, stop=this_duration, num=int(this_duration/time_step))
            pdf = lab * np.exp(-X * lab)
            call_duration = pdf * num_calls*time_step
            call_duration = call_duration.astype(np.int32)

            starts =  np.random.randint(s,e,[sum(call_duration)])
            end=[]
            pre = 0
            for idx,interval_num in enumerate(call_duration):
                interval_num+=pre
                end.append(starts[pre:interval_num]+(idx+1)*time_step)
                pre = interval_num
            end =np.concatenate(end,axis=0)

            valid_mask = end<=e
            starts=starts[valid_mask]
            end= end[valid_mask]


            starts=np.expand_dims(starts,axis=1)
            end=np.expand_dims(end,axis=1)
            start_end = np.concatenate([starts,end],1)

            # start_end = sorted(start_end, reverse=False, key=lambda x: x[1] - x[0])
            # start_end = np.array(start_end)
            #
            # start_end = sorted(start_end, reverse=False, key=lambda x: x[0])
            # start_end = np.array(start_end)


            yield start_end





