
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# using acc2tk,inter_tks
class Drawer:
    def __init__(self):
        self.colors=['r','g','b','c','m','y','k']

        pass




    def __get_data_item__(self):
        pass

    def drawAer(self,data,config,position=None ):



        fig = plt.figure(1)
        for sub_idx, col in enumerate(config['data_show_lines'][1:]): #different value lines
            plt.subplot(len(config['data_show_lines'])-1, 1, sub_idx + 1)  # without time

            for idx, access_name in enumerate(data.access_names):# different access lines in a value
                for line in data.get_sublines(access_name,[col],with_time=True):# different subline in a line


                    plt.plot(line.T[0],line.T[1],'{}'.format(self.colors[idx%(len(self.colors))]))

                    if position:
                        plt.text(position[access_name][0]*10+8640,position[access_name][1],access_name,fontsize=10, color = "k", style = "italic")

        return fig

    def drawAerSolution(self,config,data,final_solution,position,inter_tks,data_processed):

        fig = plt.figure(2)

        solution_length = len(final_solution)
        ret_tks = {}
        for cnt in range(1, solution_length -1):
            s_pre, s, s_next = final_solution[cnt - 1], final_solution[cnt], final_solution[cnt + 1]
            if s_pre == 'none':
                ret_tks[s] = (data_processed.acc2tk[s][0], inter_tks[(s, s_next)])
                continue
            if s_next == 'none':
                ret_tks[s] = (inter_tks[(s_pre, s)], data_processed.acc2tk[s][-1])

                continue
            if s == 'none':
                continue

            ret_tks[s] = (inter_tks[(s_pre, s)], inter_tks[(s, s_next)])
        # 最后一个星,补上
        ret_tks[final_solution[-1]] = (inter_tks[(final_solution[-2], final_solution[-1])], data_processed.acc2tk[final_solution[-1]][-1])



        for sub_idx, col in enumerate(config['data_show_lines'][1:]):  # different value lines
            plt.subplot(len(config['data_show_lines']) - 1, 1, sub_idx + 1)  # without time

            for idx, access_name in enumerate(data.access_names):  # different access lines in a value
                for line in data.get_sublines(access_name, [col], with_time=True):  # different subline in a line

                    # if access_name not in total_solution:
                            # plt.plot(line.T[0], line.T[1], color=[0.2,0.3,0.4,0.5])
                    plt.plot(line.T[0], line.T[1], '{}'.format(self.colors[idx % (len(self.colors))]),alpha=0.2)

                    if access_name in ret_tks.keys():
                        best_mask = (line.T[0]>= ret_tks[access_name][0]) *(line.T[0] <= ret_tks[access_name][1])
                        plt.plot(
                            line.T[0][best_mask],
                            line.T[1][best_mask],
                            'r')
                        plt.text(position[access_name][0] * 10 + data.all_tks[0], position[access_name][1], access_name,
                                 fontsize=10, color="k", style="italic")

        return fig



    def drawGraph(self,G,position=None,final_solution=None):
        font_size =14
        edge_color=[]
        fig = plt.figure(3)
        if position ==None:
            position = nx.spring_layout(G)
        if final_solution==None:
            nx.draw(G,
                pos=position,node_color = 'b', edge_color = 'k', with_labels = True,font_size = font_size, node_size = 120)
        else:

            for (si_prev,si_next) in G.edges:
                if si_prev in final_solution and si_next in final_solution:
                    edge_color.append('r')
                else:
                    edge_color.append('k')

            nx.draw(G,
                    pos=position,
                    node_color='b',
                    edge_color=edge_color,
                    with_labels=True,
                    font_size=font_size,
                    node_size=120
                    )

        return fig