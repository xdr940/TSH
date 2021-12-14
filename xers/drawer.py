
import matplotlib.pyplot as plt
import numpy as np
class Drawer:
    def __init__(self):
        self.colors=['r','g','b','c']
        self.fig = plt.figure(1)

        pass
    def run(self):
        plt.show()

        pass



    def __get_data_item__(self):
        pass
    def __call__(self, idx,access_name,sub_df):
        # time_arr = np.array(sub_df).T

        for sub_idx,col in enumerate(sub_df.columns[1:]):
            plt.subplot(2,1,sub_idx+1)# without time

            plt.plot(sub_df['time'],sub_df[col],'{}.'.format(self.colors[idx%len(self.colors)]))
            plt.title(col)

            # line.set_drawstyle('_draw_steps_pre')

        pass

