
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
        self.x_min = int(data.df['time'].min())
        self.x_max = int(data.df['time'].max())
        self.y_min = data.df[config['data_show_lines'][-1]].min()
        self.y_max = data.df[config['data_show_lines'][-1]].max()
        self.margin = self.y_max - self.y_min

        plt.xlim([self.x_min - 100, self.x_max + 100])
        plt.ylim([self.y_min -self.margin/10 , self.y_max + self.margin/10 ])

        for sub_idx, col in enumerate(config['data_show_lines'][1:]): #different value lines
            plt.plot(len(config['data_show_lines'])-1, 1, sub_idx + 1)  # without time

            for idx, access_name in enumerate(data.access_names):# different access lines in a value
                for line in data.get_sublines(access_name,[col],with_time=True):# different subline in a line


                    plt.plot(line.T[0],line.T[1],'{}'.format(self.colors[idx%(len(self.colors))]))

                    if position:
                        plt.text(position[access_name][0]*10+self.x_min,position[access_name][1],access_name,fontsize=10, color = "k", style = "italic")

        return fig

    def drawAerSolution(self,config,data,final_solution,position,inter_tk_dict,data_processed):

        fig = plt.figure(2)

        solution_length = len(final_solution)

        # ret tks 就是每个access 实际画出的部分
        # 例如 s2420:(8640,inter_tk(s2420,s2520))
        ret_tks = {}
        for cnt in range(1, solution_length -1):
            s_pre, s, s_next = final_solution[cnt - 1], final_solution[cnt], final_solution[cnt + 1]
            if s_pre == 'none':
                ret_tks[s] = (data_processed.acc2tk[s][0], inter_tk_dict[(s, s_next)])
                continue
            if s_next == 'none':
                ret_tks[s] = (inter_tk_dict[(s_pre, s)], data_processed.acc2tk[s][-1])

                continue
            if s == 'none':
                continue

            ret_tks[s] = (inter_tk_dict[(s_pre, s)], inter_tk_dict[(s, s_next)])
        # 最后一个星,补上
        ret_tks[final_solution[-1]] = (inter_tk_dict[(final_solution[-2], final_solution[-1])], data_processed.acc2tk[final_solution[-1]][-1])

        # x_min = int(data.df['time'].min())
        # x_max = int(data.df['time'].max())
        # y_min = int(data.df[config['data_show_lines'][-1]].min())
        # y_max = int(data.df[config['data_show_lines'][-1]].max())
        # M = y_max-y_min

        plt.xlim([self.x_min - 100, self.x_max + 100])
        plt.ylim([self.y_min-self.margin/10 , self.y_max+self.margin/10 ])

        #
        for sub_idx, col in enumerate(config['data_show_lines'][1:]):  # different value lines
            plt.plot(len(config['data_show_lines']) - 1, 1, sub_idx + 1)  # without time

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
        plt.title("Access graph")

        plt.xlabel("Time (second)")


        return fig

    def drawAccessSolution(self,config,data,final_solution,position,inter_tk_dict,data_processed):
        fig = plt.figure(4)
        solution_length = len(final_solution)

        # ret tks 就是每个access 实际画出的部分
        # 例如 s2420:(8640,inter_tk(s2420,s2520))
        ret_tks = {}
        for cnt in range(1, solution_length - 1):
            s_pre, s, s_next = final_solution[cnt - 1], final_solution[cnt], final_solution[cnt + 1]
            if s_pre == 'none':
                ret_tks[s] = (data_processed.acc2tk[s][0], inter_tk_dict[(s, s_next)])
                continue
            if s_next == 'none':
                ret_tks[s] = (inter_tk_dict[(s_pre, s)], data_processed.acc2tk[s][-1])

                continue
            if s == 'none':
                continue

            ret_tks[s] = (inter_tk_dict[(s_pre, s)], inter_tk_dict[(s, s_next)])
        # 最后一个星,补上
        ret_tks[final_solution[-1]] = (
        inter_tk_dict[(final_solution[-2], final_solution[-1])], data_processed.acc2tk[final_solution[-1]][-1])



        plt.xlim([self.x_min - 100, self.x_max + 100])
        # plt.ylim([self.y_min - self.margin / 10, self.y_max + self.margin / 10])
        end_with ={0:8640}# height=0, end_tk=0
        height_dict={'none':0}#access name : height
        max_height=0
        for sub_idx, col in enumerate(config['data_show_lines'][1:]):  # different value lines
            plt.plot(len(config['data_show_lines']) - 1, 1, sub_idx + 1)  # without time

            for access_name in data.access_names:
                for line in data.get_sublines(access_name,[col],with_time=True):# 基本为1
                    start_tk = line.T[0][0]
                    min_space = 10000
                    suited_height=0
                    for height,end_tk in end_with.items():
                        space_time = start_tk - end_tk
                        if space_time<0:
                            continue
                        if space_time < min_space:
                            suited_height =height
                            min_space = space_time

                    if min_space!=10000:
                        height_dict[access_name]=suited_height
                        end_with[suited_height]=line.T[0][-1]
                    else:
                        max_height+=1
                        height_dict[access_name]=max_height
                        end_with[max_height] = line.T[0][-1]






        #
        for sub_idx, col in enumerate(config['data_show_lines'][1:]):  # different value lines
            plt.plot(len(config['data_show_lines']) - 1, 1, sub_idx + 1)  # without time

            for idx, access_name in enumerate(data.access_names):  # different access lines in a value
                for line in data.get_sublines(access_name, [col], with_time=True):  # different subline in a line

                    # if access_name not in total_solution:
                    # plt.plot(line.T[0], line.T[1], color=[0.2,0.3,0.4,0.5])
                    plt.plot(line.T[0], np.ones_like(line.T[0])*height_dict[access_name], '{}'.format(self.colors[idx % (len(self.colors))]), alpha=0.5)

                    if access_name in ret_tks.keys():
                        best_mask = (line.T[0] >= ret_tks[access_name][0]) * (line.T[0] <= ret_tks[access_name][1])
                        plt.plot(
                            line.T[0][best_mask],
                            height_dict[access_name]*np.ones_like(line.T[0])[best_mask],
                            'r')
                        plt.text(position[access_name][0] * 10 + data.all_tks[0], height_dict[access_name], access_name,
                                 fontsize=10, color="k", style="italic")
        plt.title("Access graph")
        plt.yticks([])
        plt.xlabel("Time (second)")
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
            red_edges = []
            for prev,next in zip(final_solution[:-1],final_solution[1:]):
                red_edges.append((prev,next))
            for edge in G.edges:
                if edge in red_edges:
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