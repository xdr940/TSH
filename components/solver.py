import itertools

import networkx as nx
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from components.AcTik import Tik
#using tk2acc,acc2tk,crossLog, inter_tks
from utils.tool import dfs_depth,max_d
class TimeStamp:
    def __init__(self):
        pass

class Solver:
    def __init__(self):
        pass

class DpSolver:
    def __init__(self,data,alg_base="Max - Range (km)"):
        self.data = data
        #tk2acc
        #acc2tk
        #data.access_name
        #df_align
        #is_eual

        self.alg_base =alg_base
        self.position=None

    def s_prev(self,si):
        return self.__s_prev(si)
    def __s_prev(self,i_access,tj=None):
        '''
        等价于节点的父节点
        :param i_access:
        :param tj: 一般不需要, 如果时间跨度很长, 卫星可能转两圈的时候就需要
        :return:
        '''

        def tks_half( i_access, half):
            try:
                mid_tk = self.__si_max_tk(i_access)
            except:
                print(i_access)
                return
            half_list = []
            for item in self.data.acc2tk[i_access]:
                if half == 'last':
                    if item > mid_tk:
                        half_list.append(item)
                elif half == 'first':
                    if item < mid_tk:
                        half_list.append(item)
            return half_list

        prevs = []
        first_half = tks_half(i_access,'first')
        for tk in first_half:
            tik = self.data.tiks[tk]
            if tik.class_id == 'I':
                continue
            elif tik.class_id =='II':
                inters = list( tik.passInter[0])
                inters.remove(i_access) #可能多个inter?
                for inter_one in inters:
                    if self.__si_max_tk(inter_one) < tk and self.data.is_equal(i_access,inter_one,tk):
                        prevs.append(inter_one)
            elif tik.class_id =='III':
                for inters in tik.passInter:
                    inters = list(inters)
                    if i_access in inters:
                        inters.remove(i_access)  # 可能多个inter?
                        for inter_one in inters:
                            if self.__si_max_tk(inter_one) < tk and self.data.is_equal(i_access, inter_one, tk):
                                prevs.append(inter_one)

            else:
                continue
        # if len(prevs) ==0:
        #     return ['none']
        # else:
        return prevs



    def __trave(self):
        '''
        最小最大深度遍历法
        :return:
        '''
        # print('->trave')
        # for depth in dfs_depth(self.G):
        #     print(depth)


        # for head in self.roots:
        #     trave_list.append( head)
        #     # trave_list.extend(list(dict(nx.bfs_successors(self.G,head)).values()))
        #     listoflist = dict(nx.bfs_successors(self.G,head)).values()
        #     for ls in listoflist:
        #         trave_list.extend(ls)

        print("trave as :{}".format(self.data.accs))
        return self.data.accs





    def __integ(self,i_access,tj,tj_next):
        if i_access not in self.data.access_names:
            print(" sat num error")

            return
        else:
            tmp =  self.data.df_align.query("time >= {} and time <= {}".format(tj, tj_next))[i_access]
            if pd.isnull(tmp).sum():
                print(" section error")
                return
            else:
                return tmp.sum()


    def __si_max_tk(self,si,tj=None):
        try:#max 可能有多个值
            max_time = self.data.df_align.query("{} == {}".format(si, self.data.df_align[si].max())).index
            ret = math.ceil(max_time[0])
            return ret
        except:
            print(si)


    def build_graph(self,weights='1'):
        assert(weights in ['1','tk'])
        print("\nGRAPH BUILDING")
        #build the graph whose access as the node

        self.G = nx.DiGraph(date='2021-12-22', name='handover')
        self.G.add_nodes_from(self.data.access_names)
        for tk in self.data.all_tks[1:]:# 根据tks 来建图
            tik = self.data.tiks[tk]

            if tik.class_id =='I' : #初始点或者结束点
                continue

            elif tik.class_id =='II': #普通inter
                try:
                    s1,s2 = tik.passInter[0]
                except:
                    pass
                    print('error')
                if self.__si_max_tk(s1) < tk and s1 in self.__s_prev(s2):
                    if weights=='tk':
                        self.G.add_weighted_edges_from([(s1, s2, tk)])
                    elif weights=='1':
                        self.G.add_weighted_edges_from([(s1, s2, 1)])

                elif self.__si_max_tk(s1) > tk and s2 in self.__s_prev(s1):
                    if weights=='tk':
                        self.G.add_weighted_edges_from([(s2, s1, tk)])

                    elif weights == '1':

                        self.G.add_weighted_edges_from([(s2, s1, 1)])



            elif tik.class_id =='III': #III类tik
                # 处理 inter点
                inters = tik.getPass('Inter')
                for inter in inters:
                    for s1,s2 in  itertools.combinations(inter, 2) :

                        if self.__si_max_tk(s1) < tk:
                            if weights=='tk':
                                self.G.add_weighted_edges_from([(s1, s2, tk)])
                            elif weights=='1':
                                self.G.add_weighted_edges_from([(s1, s2, 1)])

                        else:
                            if weights=='tk':
                                self.G.add_weighted_edges_from([(s2, s1, tk)])
                            elif weights=='1':
                                self.G.add_weighted_edges_from([(s2, s1, 1)])

                # 处理passIn
                # 处理passOut
                # 包括2个以上卫星函数相交
                # 或者不同的函数相交点恰好在一个时刻
        #把函数的顶点和中点作为点位置
        #from none
        #sub graph heads

        print("--> graph finished")
        print(self.G)

    def mst_run(self):
        # nx.draw(self.G,with_labels=True)
        # plt.show()

        path = nx.dijkstra_path(self.G, source='s2520', target='s2616')
        path.insert(0,'none')
        return path
        # print(nx.shortest_path(tmp_graph, source='s2520', target='s2519'))
        # path = nx.all_pairs_shortest_path(self.G)
        # print(path)

    def get_roots_and_leaves(self):
        roots = []
        leaves =[]

        for si in self.__trave():
            if len(self.__s_prev(si))==0 and self.G.out_degree[si] !=0:
                # 这里不等等价为子图个数, 如果相位相同两个初始access, 会出错
                # 过滤掉1个点的子图
                roots.append(si)
            if len(self.__s_prev(si))!=0 and self.G.out_degree[si] ==0:
                leaves.append(si)

        # buble sort for each sub-graph
        for i in range(len(roots)):
            for j in range(i+1,len(roots)):
                if self.data.crossLog[roots[i]][0][0] >self.data.crossLog[roots[j]][0][0]:
                    tmp = roots[i]
                    roots[i] = roots[j]
                    roots[j]= tmp

        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                if self.data.crossLog[leaves[i]][0][0] > self.data.crossLog[leaves[j]][0][0]:
                    tmp = leaves[i]
                    leaves[i] = leaves[j]
                    leaves[j] = tmp


        self.roots = roots
        self.leaves = leaves


        print("-> graph roots num:{}\n {}".format(len(self.roots),self.roots))
        print("-> graph leaves num:{} \n{}".format(len(self.leaves),self.leaves))


    def run(self):
        self.get_roots_and_leaves()

        print("\nPROBLEM SOVING BY DP")

        hop={}
        opt={}
        hop['none'] = 0
        opt['none']=0
        route={}
        for root in self.roots:#先对根节点(in-degree ==0)的都预处理
            tj,tj_next = self.data.acc2tk[root][0],self.data.acc2tk[root][-1]
            opt[root] = (hop['none']* opt['none'] + self.__integ(root,tj,tj_next))/(hop['none']+1)
            hop[root] = hop['none'] + 1
            route[root] = 'none'

        for si in tqdm(self.__trave()):
            # if si =='s3017':
            #     pass
            pre_sis = self.__s_prev(si)
            tj_next = self.data.acc2tk[si][-1]

            if len(pre_sis)==0  : # 没有前置si,si即为相位最靠前的卫星

                tj = self.data.acc2tk[si][0]
                opt[si] =(hop['none']* opt['none'] + self.__integ(si,tj,tj_next))/(hop['none']+1)
                hop[si] = hop['none']+1
                route[si] = 'none'

            else:
                tmp_opt = 0
                arg_opt=0
                for pre_si in pre_sis:

                    if (pre_si ,si)not in self.G.edges: #这里遍历方法不同可能会出错
                        print('error in( {},{})'.format(pre_si,si))
                        continue
                    tj = self.G.edges[pre_si,si]['weight']
                    si_last_integ = self.__integ(pre_si,tj,self.data.acc2tk[pre_si][-1])
                    integ = self.__integ(si,tj,tj_next)
                    tmp = (hop[pre_si]*opt[pre_si] - si_last_integ + integ)/(hop[pre_si]+1)
                    if tmp > tmp_opt:
                        tmp_opt = tmp
                        arg_opt = pre_si

                opt[si] =tmp_opt
                hop[si] = hop[arg_opt]+1
                route[si] = arg_opt

            # last_si = si

        #returns
        self.opt = opt
        self.hop= hop
        self.route = route


        #post process

        paths = {}  # all paths in all sub graphs
        for leaf in self.leaves:
            paths[leaf] = []
            paths[leaf].append(leaf)

            end_node = leaf
            while end_node != 'none':
                paths[leaf].append(self.route[end_node])
                end_node = self.route[end_node]
            paths[leaf].reverse()

        candidate_paths = {}
        for root in self.roots:
            argmax_leaf = None
            sub_opt = 0
            for leaf in self.leaves:
                if root in paths[leaf] and self.opt[leaf] > sub_opt:
                    sub_opt = self.opt[leaf]
                    argmax_leaf = leaf
            candidate_paths[root] = argmax_leaf

        # 选择候选路径中几个最大覆盖的多个路径为最后结果
        solutions = []

        hop = []
        opt_values = []
        for index_node, path in paths.items():
            for root, leaf in candidate_paths.items():
                if root == None or leaf == None:
                    continue
                if leaf == index_node and root in path:
                    # best paths cover all sub graphs
                    print("-> (sub) path:\n{}".format(path))
                    print("path value: {}, path handover times:{}\n".format(self.opt[leaf], self.hop[leaf]))
                    solutions.append(path)
                    hop.append(self.hop[leaf])
                    opt_values.append(self.opt[leaf])

        max_len = 0
        final_solution = None
        final_hop = 0
        final_opt_value = 0
        for p in solutions:
            start = self.data.acc2tk[p[1]][0]
            end = self.data.acc2tk[p[-1]][-1]
            this_len = end - start
            if this_len > max_len:  # 持续时间大于目前最大
                max_len = this_len
                final_solution = p
                final_hop = self.hop[p[-1]]
                final_opt_value = self.opt[p[-1]]
            elif self.__path_tk(p, 'tail') < self.__path_tk(final_solution, 'head'):  # 持续时间段在目前前面, 可以合并
                p.extend(final_solution)

                final_opt_value = final_hop * final_opt_value + self.hop[p[-1]] * self.opt[p[-1]]

                final_hop = final_hop + self.hop[p[-1]] + 1

                final_opt_value /= final_hop

                final_solution = p

            elif self.__path_tk(p, 'head') > self.__path_tk(final_solution, 'tail'):  # 持续时间在目前后面,可以合并

                final_opt_value = final_hop * final_opt_value + self.hop[p[-1]] * self.opt[p[-1]]

                final_hop = final_hop + self.hop[p[-1]] + 1

                final_opt_value /= final_hop

                final_solution.extend(p)

        self.final_hop = final_hop
        self.final_solution = final_solution
        return  final_solution

            # print('opt: ',opt)
            # print('k:',k)
    def rss_run(self):
        print("\nPROBLEM SOVING BY RSS")
        tmp_list = []
        for idx,row in self.data.df_align.iterrows():
            row = row.replace(np.nan, 0)
            if row.sum()==0:
                tmp_list.append('none')
            else:
                max_acc = np.argmax(row)
                tmp_list.append(self.data.df_align.columns[max_acc])
        final_solution = ['none']

        #去掉成片的重复access name
        i=0
        final_solution.append(tmp_list[0])
        while True:
            if tmp_list[i] == final_solution[-1]:
                pass
            else:
                final_solution.append(tmp_list[i])
            i+=1
            if i >= len(tmp_list):
                break

        print('rss list :{}'.format(final_solution))
        self.final_solution = final_solution
        self.final_hop = len(self.final_solution)
        return final_solution
   
    def get_inter_tks(self,final_solution):
        inter_tk_dict = {}
        for s_prev, s_next in zip(final_solution[1:-1], final_solution[2:]):
            inter_tk_dict[s_prev, s_next] = self.data.getInterTk(s_prev, s_next)

        return  inter_tk_dict


    def result_stat(self):

        # 记录所有最大联通子图的可能路径
        print("\nPROBLEM STAT")


        final_solution = self.final_solution
        final_hop = self.final_hop
        # 找到每个解
        # 的分量的交点,后面移动到 data prep 更合适
        inter_tk_dict=self.get_inter_tks(final_solution)




        total_time = self.__path_tk(final_solution, 'tail')-self.__path_tk(final_solution, 'head')


        #断路时间
        disconn_time=0
        disconn_times=0
        for i in range(1,len(final_solution)-1):
            if final_solution[i]=='none'and final_solution[i-1]!='none'and final_solution[i+1]!='none':
                disconn_time +=(self.data.getInterTk( 'none',final_solution[i+1]) - self.data.getInterTk(final_solution[i-1], 'none'))
                disconn_times +=1

        self.inter_tk_dict = inter_tk_dict
        print('\n-> solution stat'+
              '\n--> best solution:\t{}'.format(final_solution)+
              # '\n--> opt value:\t{:.2f}'.format(final_opt_value)+
              '\n--> handover times: \t{}, disconn times:{}'.format(final_hop,disconn_times)+
              '\n--> avg duration:\t{:.2f}(s).'.format(total_time/final_hop)+
              # '\n--> avg alg base:\t{:.2f}.'.format(final_opt_value *final_hop/ total_time) +
              '\n--> total time:\t{:.2f}(s), disconn time:\t{:.2f}({:.2f}%)'.format(total_time,disconn_time,100*disconn_time/total_time)

              )



        #returns


    def get_selected_alg_base(self):
        inter_tk_dict = self.inter_tk_dict
        final_solution = self.final_solution
        data = self.data
        nan_assign = 0

        solution_length = len(final_solution)

        ret_tks = {}
        for cnt in range(1, solution_length - 1):
            s_pre, s, s_next = final_solution[cnt - 1], final_solution[cnt], final_solution[cnt + 1]
            if s_pre == 'none':
                ret_tks[s] = (self.data.acc2tk[s][0], inter_tk_dict[(s, s_next)])
                continue
            if s_next == 'none':
                ret_tks[s] = (inter_tk_dict[(s_pre, s)], self.data.acc2tk[s][-1])

                continue
            if s == 'none':
                continue

            ret_tks[s] = (inter_tk_dict[(s_pre, s)], inter_tk_dict[(s, s_next)])
        # 最后一个星,补上
        ret_tks[final_solution[-1]] = (
        inter_tk_dict[(final_solution[-2], final_solution[-1])], self.data.acc2tk[final_solution[-1]][-1])


        arrs=[]
        for access_name in ret_tks.keys():

            # for line in data.get_sublines(access_name, [alg_base], with_time=True):
            line = data.df_align[access_name]
                # if access_name in ret_tks.keys():
            best_mask = (line.index >= ret_tks[access_name][0]) * (line.index <= ret_tks[access_name][1])
            if len(arrs) > 0:
                start = arrs[-1].index[-1]
                stop = line[best_mask].index[0]
                if start+1 != stop:#s将断线的地方设置为0
                    time_index = np.linspace(start= start+1,stop=stop,num=stop-start)
                    arrs.append(pd.Series(np.ones_like(time_index)*nan_assign,index=time_index))
                    # print("--> {} to {} is null".format(arrs[-1].index[-1],line[best_mask].index[0]))


            arrs.append(line[best_mask][:-1])

        return pd.concat(arrs,axis=0)
        # pass

    def _get_inter_tks(self,solution):
        pass


    def __path_tk(self,path,headOrTail):
        if headOrTail=='head':
            return  self.data.acc2tk[path[1]][0]
        if headOrTail =="tail":
            return  self.data.acc2tk[path[-1]][-1]



class GreedySolver:
    def __init__(self):
        pass


