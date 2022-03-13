import itertools

import networkx as nx
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from components.AcTik import Tik
#using tk2acc,acc2tk,crossLog, inter_tks
from utils.tool import get_now,time_stat
class TimeStamp:
    def __init__(self):
        pass



class Solver:
    def __init__(self,data,alg_base="Max - Range (km)",terminal=False):
        self.data = data
        self.terminal = terminal
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

    def __s_next(self,si):
        return list(self.G.succ[si].keys())

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
        if self.terminal:
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
        if self.terminal:
            print("\n-> GRAPH BUILDING")
        #build the graph whose access as the node

        self.G = nx.DiGraph(date='2022-3-3', name='handover')
        self.G.add_nodes_from(self.data.access_names)
        for tk in tqdm(self.data.all_tks[1:]):# 根据tks 来建图
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
        self.get_roots_and_leaves2()
        if self.terminal:
            print("--> graph finished")
            print(self.G)
    def get_roots_and_leaves2(self):
        pass
        roots=[]
        leaves=[]
        for acc in self.data.access_names:
            if self.data.acc2tk[acc][0] == self.data.all_tks[0]:
                roots.append(acc)
            if self.data.acc2tk[acc][-1] == self.data.all_tks[-1]:
                leaves.append(acc)

        self.roots = roots
        self.leaves = leaves
        if self.terminal:

            print("--> graph roots num:{} {}".format(len(self.roots), self.roots))
            print("--> graph leaves num:{} {}".format(len(self.leaves), self.leaves))

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
        if self.terminal:
            print("-> graph roots num:{}\n {}".format(len(self.roots),self.roots))
            print("-> graph leaves num:{} \n{}".format(len(self.leaves),self.leaves))



    def mst_run(self):
        # nx.draw(self.G,with_labels=True)
        # plt.show()
        paths={}
        for root in self.roots:
            for leaf in self.leaves:
                try:
                    path = nx.dijkstra_path(self.G, source=root, target=leaf)
                    paths[root,leaf]=path
                except:
                    continue
                    pass
        if self.terminal:

            print('roots:{}'.format(self.roots))
            print('leaves:{}'.format(self.leaves))
        root_leaf = sorted(paths.values(),reverse=True,key=lambda x:len(x))
        final_solutoin = self.max_cover(paths,mode='mst')
        root_leaf[0].insert(0,'none')
        return final_solutoin
        # print(nx.shortest_path(tmp_graph, source='s2520', target='s2519'))
        # path = nx.all_pairs_shortest_path(self.G)
        # print(path)
    def gd_run(self):
        final_solution=[]



        x_ks = self.roots
        while x_ks:
            max_intg =0
            max_x =None
            for x in x_ks:
                if len(final_solution):
                    tj, tj_next = self.data.getInterTk(final_solution[-1],x), self.data.acc2tk[x][-1]
                else:
                    tj, tj_next = self.data.acc2tk[x][0], self.data.acc2tk[x][-1]
                tmp_intg = self.__integ(x, tj, tj_next)
                if max_intg<tmp_intg:
                    max_intg = tmp_intg
                    max_x = x
            final_solution.append(max_x)
            x_ks = list(self.G.succ[max_x].keys())
        final_solution.insert(0,'none')
        return final_solution

    def dp_run(self):
        def check_start_stamp(access):
            pass
            if self.data.acc2tk[access][0] == self.data.all_tks[0]:
                return self.data.all_tks[0]
            elif route_dict[access]=='none':
                return self.data.acc2tk[access][0]
            else:
                return check_start_stamp(route_dict[access])
        # self.get_roots_and_leaves()
        if self.terminal:

            print("\nPROBLEM SOVING BY DP")

        hop_dict={}
        opt_dict={}
        hop_dict['none'] = 0
        opt_dict['none']=0
        route_dict={}
        for root in self.roots:#先对根节点(in-degree ==0)的都预处理
            tj,tj_next = self.data.acc2tk[root][0],self.data.acc2tk[root][-1]
            opt_dict[root] = (hop_dict['none']* opt_dict['none'] + self.__integ(root,tj,tj_next))/(hop_dict['none']+1)
            hop_dict[root] = hop_dict['none'] + 1
            route_dict[root] = 'none'

        for si in tqdm(self.__trave()):
            # if si =='s3017':
            #     pass
            pre_sis = self.__s_prev(si)
            tj_next = self.data.acc2tk[si][-1]

            if len(pre_sis)==0  : # 没有前置si,si即为相位最靠前的卫星
                # if si not in self.roots:
                #     continue
                tj = self.data.acc2tk[si][0]
                opt_dict[si] =(hop_dict['none']* opt_dict['none'] + self.__integ(si,tj,tj_next))/(hop_dict['none']+1)
                hop_dict[si] = hop_dict['none']+1
                route_dict[si] = 'none'

            else:
                acc_opt ={}
                for pre_si in pre_sis:

                    if (pre_si ,si)not in self.G.edges: #这里遍历方法不同可能会出错
                        print('error in( {},{})'.format(pre_si,si))
                        continue
                    tj = self.G.edges[pre_si,si]['weight']
                    si_last_integ = self.__integ(pre_si,tj,self.data.acc2tk[pre_si][-1])
                    integ = self.__integ(si,tj,tj_next)
                    acc_opt[pre_si] = (hop_dict[pre_si]*opt_dict[pre_si] - si_last_integ + integ)/(hop_dict[pre_si]+1)
                    # if tmp > tmp_opt :
                    #     tmp_opt = tmp
                    #     arg_opt = pre_si
                # if acc_opt:
                acc_opt  = sorted(acc_opt.items(),reverse=True, key=lambda x: x[1])
                for item in acc_opt:
                    if route_dict[item[0]]=='none' and self.data.acc2tk[item[0]][0]!=self.data.all_tks[0]:
                        continue

                    opt_dict[si] =item[1]
                    hop_dict[si] = hop_dict[item[0]]+1
                    route_dict[si] = item[0]
                    break


        self.opt_dict=opt_dict

        #post process

        paths = {}  # all paths in all sub graphs
        for leaf in self.leaves:
            paths[leaf] = []
            paths[leaf].append(leaf)

            end_node = leaf
            while end_node != 'none':
                paths[leaf].append(route_dict[end_node])
                end_node = route_dict[end_node]
            paths[leaf].reverse()
            paths[leaf]=paths[leaf][1:]
            paths[paths[leaf][0],leaf] = paths[leaf]
            del paths[leaf]
        if self.terminal:
            print('--> paths:')
            for path in paths.values():
                print(path)
        final_solution = self.max_cover(paths,mode='dp')#最大覆盖方法筛选
        return final_solution


    def __keep(self,path_i,path_j,mode):
        '''
        确定是留2个还是留任意一个, 共计三种状态

        :param path0:
        :param path1:
        :return:
            0: 留0
            1: 留1
            2:全留
        '''
        test_list = [path_i[0],path_i[1],path_j[0],path_j[1]]
        ret = list(np.argsort(test_list))
        if len(set(test_list))==2:

            if mode=='dp' and path_i[2]>path_j[2] :
                return 0
            elif mode =='mst' and path_i[3]<path_j[3] :
                return 0
            else:
                return 1

            # 相等
        elif len(set(test_list))==3:
            if test_list[1]==test_list[2]:
            # if ret ==[0,1,2,3]:
                return 2
            # 0|--|1
            #    2|--|3
            #[0,1,1,2]-->[0,1,2,3]
            # elif ret ==[2,0,3,1]:
            if test_list[0]==test_list[3]:
                return 2
            #   0|--|1
            #2|--|3
            # [1,2,0,1]--> [2,0,3,1]

            # if ret ==[0,2,1,3]:
            if test_list[1]==test_list[3]:
                # |----|
                #   |--|
                # [0 ,2, 1, 2]
                return 0
            if test_list[0]==test_list[2] and test_list[3]>test_list[1]:
                # 0|--|1
                # 2|----|3
                # [0,1,0,2]
                return 1


            if test_list[0]==test_list[2] and test_list[1]>test_list[3]:
                # |----|
                # |--|
                # [0,2,0,1] -->[0,2,3,1]
                return 0
        elif len(set(test_list))==4:
            pass

            if ret == [2, 0, 1, 3]  :
                #   |--|
                #  |----|
                #  [1,2,0,3]--> [2,0,1,3]
                return 1
            elif ret == [0, 2, 3, 1]:
                #
                #    |----|
                #     |--|
                #   [0,3,1,2] -> [0,2,3,1]
                return 0


            if ret == [0, 2, 1, 3] :
                # |---|
                #   |---|
                if (path_i[1]-path_i[0]) >(path_j[1]-path_j[0]):
                    return 0
                else:
                    return 1
                # return 2


            if ret ==[1, 3, 0, 2]:
                #      |---|
                #    |---|
                if (path_i[1]-path_i[0]) >(path_j[1]-path_j[0]):
                    return 0
                else:
                    return 1
            if ret == [0, 1, 2 , 3 ] or [2,3,0,1]:
                # |--|
                #      |--|
                #
                #        |--|
                #  |--|
                return 2
                #不想交



    def max_cover(self,paths_dict,mode):
        # 按照结束时刻, 从大到小排列
        # for path in paths
        #
        paths_list_detail=[]
        if mode=='dp':
            for (root,leaf), path in paths_dict.items():
                paths_list_detail.append(
                    (
                        (root,leaf),#table[i][0]
                        (
                            self.data.acc2tk[path[0]][0],   #table[i][1][0] start
                            self.data.acc2tk[path[-1]][-1], #table[i][1][1] end
                            self.opt_dict[leaf],             #table[i][1][2] opt value
                            len(path)
                        )
                    )
                )
        elif mode =='mst':
            for (root,leaf), path in paths_dict.items():
                paths_list_detail.append(
                    (
                        (root,leaf),#table[i][0]
                        (
                            self.data.acc2tk[path[0]][0],   #table[i][1][0] start
                            self.data.acc2tk[path[-1]][-1], #table[i][1][1] end
                            1 ,            #table[i][1][2] opt value
                            len(path)
                        )
                    )
                )


        # 按照覆盖时间从小到大排列
        # 如果有被完全覆盖的, drop掉
        # 如果有恰好覆盖的, drop opt低的
        # 找最大覆盖的n个


        # sort1 按照结束时间倒序排



        # path_table=[
        #     (('1','2'),  (7, 10, 23)),
        #     (('3', '4'), (4, 10, 4)),
        #     (('6', '5'), (2, 10, 12)),
        #     (('7', '8'), (9, 12, 3)),
        #     (('9', '10'),(9, 13, 3)),
        #     (('11', '12'),(1, 2, 3)),
        #     (('13', '14'),(7, 10, 3)),
        #
        # ]
        # path_dict={}
        # path_table = sorted(path_table,reverse=True,key=lambda x:(x[1][1]-x[1][0]))
        # for tab in path_table:
        #     print('\n', tab)
        # print('-->')
        del_set = set([])
        for path_i,path_j in itertools.combinations(paths_list_detail,2):
            if path_i[0] in del_set or path_j[0] in del_set:
                continue
            sta = self.__keep(path_i[1],path_j[1],mode)


            if sta ==0:
                del_set.add(path_j[0])
            elif sta ==1 :
                del_set.add(path_i[0])
        filted_paths_dict={}
        for item,value in paths_dict.items():
            if item not in del_set:
                value.insert( 0,'none')
                filted_paths_dict[item] = value

        filted_paths_list = sorted(filted_paths_dict.values(),reverse=False,key=lambda x:self.data.acc2tk[x[1]][0])
        final_solution=[]
        for path in filted_paths_list:
            final_solution+=path
        return final_solution




    def mea_run(self):
        '''
        最强信号选择
        :return:
        '''
        if self.terminal:

            print("\n-> PROBLEM SOVING BY RSS")
        start = get_now()

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

        time_stat(start)
        return final_solution
   
    def get_inter_tks(self,final_solution):
        inter_tk_dict = {}
        inter_tk_list = []
        for s_prev, s_next in zip(final_solution[1:-1], final_solution[2:]):
            tk = self.data.getInterTk(s_prev, s_next)
            inter_tk_dict[s_prev, s_next] = tk
            inter_tk_list.append(tk)
        return  inter_tk_dict,inter_tk_list




    def get_selected_alg_base(self,inter_tk_dict,final_solution):
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



    def __path_tk(self,path,headOrTail):
        if headOrTail=='head':
            return  self.data.acc2tk[path[1]][0]
        if headOrTail =="tail":
            return  self.data.acc2tk[path[-1]][-1]



class GreedySolver:
    def __init__(self):
        pass


