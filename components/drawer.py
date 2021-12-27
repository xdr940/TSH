
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
class Drawer:
    def __init__(self):
        self.colors=['r','g','b','c','m','y','k']
        self.access_names = ['s5012','s2618','s2518']

        pass




    def __get_data_item__(self):
        pass

    def drawAer(self,data,config ):

        fig = plt.figure(1)
        # time_arr = np.array(sub_df).T
        for sub_idx, col in enumerate(config['data_show_lines'][1:]): #different value lines
            plt.subplot(len(config['data_show_lines'])-1, 1, sub_idx + 1)  # without time

            for idx, access_name in enumerate(data.access_names):# different access lines in a value
                for line in data.get_sublines(access_name,[col],with_time=True):# different subline in a line


                    plt.plot(line.T[0],line.T[1],'{}'.format(self.colors[idx%(len(self.colors))]))
                # print('idx: {}, mod:{}'.format(idx,idx%(len(self.colors))))
            plt.legend(data.access_names)
            # plt.title(col)
        return fig
        # plt.show()
            # line.set_drawstyle('_draw_steps_pre')


    def drawGraph(self,G,position=None):
        fig = plt.figure(2)
        if position ==None:
            position = nx.spring_layout(G)

        nx.draw(G,
                pos=position,node_color = 'b', edge_color = 'r', with_labels = True,font_size = 18, node_size = 120)
        return fig