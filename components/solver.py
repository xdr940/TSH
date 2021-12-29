
import networkx as nx
import pandas as pd
import math
from tqdm import tqdm
class TimeStamp:
    def __init__(self):
        pass


class DpSolver:
    def __init__(self,data,alg_base="Max - Range (km)"):
        self.data = data
        self.alg_base =alg_base
        self.position=None
        pass


    def s_prev(self,i_access,tj=None):
        '''
        等价于节点的父节点
        :param i_access:
        :param tj: 一般不需要, 如果时间跨度很长, 卫星可能转两圈的时候就需要
        :return:
        '''
        prevs = []
        first_half = self.tks_half(i_access,'first')
        for tk in first_half:
            for acc in self.data.tk2acc[tk]:
                if acc == i_access:
                    continue
                elif self.si_max_tk(acc) < tk and self.is_equal(i_access,acc,tk):
                    prevs.append(acc)

                # if tk in self.tks_half(acc,'last'):
                #     if acc not in prevs :
                #         prevs.append(acc)
                else:
                    continue
        # if len(prevs) ==0:
        #     return ['none']
        # else:
        return prevs
        #
        #
        # first_half = set(first_half)
        # for i_other in self.data.access_names:
        #     if i_other == i_access:
        #         continue
        #     if len(set(self.tks_half(i_other,'first'))&first_half)!=0:
        #         prevs.append(i_other)
        # pass
    def func(self,si,tk):
        return math.ceil(self.data.df_align.query(" time =={}".format(tk))[si])
    def is_equal(self,s1,s2,tk):
        a,b = tuple(self.data.df_align.query(" time >={} and time <={}".format(tk-1,tk))[s1])
        c,d = tuple(self.data.df_align.query(" time >={} and time <={}".format(tk-1,tk))[s2])
        if (a>c and b<d) or (a<c and b>d):
            return True
        else:
            return False
    def s_next(self,i_access,tj=None):
        '''
        只选择前半段在本接入星后半段的
        :param i_access:
        :param tj:  一般不需要, 如果时间跨度很长, 卫星可能转两圈的时候就需要
        :return:
        '''
        if i_access =='s4913':
            pass
        nexts=[]
        last_half = self.tks_half(i_access,'last')
        last_half = set(last_half)
        for i_other in self.data.access_names:
            if i_other == i_access:
                continue
            inter_tk = set(self.tks_half(i_other,'first'))&last_half
            if len(inter_tk)!=0 and self.func(i_access,inter_tk) == self.func(i_other,inter_tk):
                nexts.append(i_other)
        return nexts



    def tks_half(self,i_access,half):
        mid =( self.data.acc2tk[i_access][0]+self.data.acc2tk[i_access][-1])/2
        half_list=[]
        for item in self.data.acc2tk[i_access]:
            if half =='last':
                if item >mid:
                    half_list.append(item)
            elif half =='first':
                if item <mid:
                    half_list.append(item)
        return half_list

    def integ(self,i_access,tj,tj_next):
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

    def opt_value(self,St):
        ret = 0
        for idx,(i_access,tj) in enumerate(St):
            if i_access !='none':
                if idx < len(St)-1:
                    tj_next = St[idx+1][1]
                else:
                    tj_next = self.data.acc2tk[i_access][-1]

                ret += self.integ(i_access, tj, tj_next)
            else:
                ret+=0

        return ret/len(St)
    def si_max_tk(self,si,tj=None):
        return math.ceil(self.data.df_align.query("{} == {}".format(si,self.data.df_align[si].max()))['time'])
    def build_graph(self):

        #build the graph whose access as the node
        self.G = nx.DiGraph(date='2021-12-22', name='handover')
        self.G.add_nodes_from(self.data.access_names)
        for time in self.data.total_tks[1:]:# 根据tks 来建图
            s_at_tk = self.data.tk2acc[time]

            if len(s_at_tk) ==1 : #初始点或者结束点
                continue
            # elif len(s_at_tk) == 2 : #普通的交点
            #     # if self.func(s_at_tk[0],time)==self.func(s_at_tk[1],time):
            #     if self.is_equal(s_at_tk[0],s_at_tk[1],time):
            #         if self.si_max_tk(s_at_tk[0])< time:
            #         # if s_at_tk[0] in self.s_prev(s_at_tk[1]) and s_at_tk[1] in self.s_next(s_at_tk[0]):
            #             self.G.add_weighted_edges_from([(s_at_tk[0], s_at_tk[1], time)])
            #             print(s_at_tk[0], s_at_tk[1], time)
            #         # elif s_at_tk[1] in self.s_prev(s_at_tk[0]) and s_at_tk[0] in self.s_next(s_at_tk[1]):
            #         elif self.si_max_tk(s_at_tk[1]) < time:
            #
            #             self.G.add_weighted_edges_from([(s_at_tk[1], s_at_tk[0], time)])
            #             print(s_at_tk[1], s_at_tk[0], time)
            elif len(s_at_tk) >=2: #则为缠结
                for i in range(len(s_at_tk)):
                    for j in range(1,len(s_at_tk)):
                        # if self.func(s_at_tk[i], time) == self.func(s_at_tk[j], time):
                        if self.is_equal(s_at_tk[i], s_at_tk[j], time):
                            if self.si_max_tk(s_at_tk[i]) < time:
                                # if s_at_tk[0] in self.s_prev(s_at_tk[1]) and s_at_tk[1] in self.s_next(s_at_tk[0]):
                                self.G.add_weighted_edges_from([(s_at_tk[i], s_at_tk[j], time)])
                                # print(s_at_tk[i], s_at_tk[j], time)
                            elif self.si_max_tk(s_at_tk[j]) < time:

                                self.G.add_weighted_edges_from([(s_at_tk[j], s_at_tk[i], time)])
                                # print(s_at_tk[j], s_at_tk[i], time)


                # 包括2个以上卫星函数相交
                # 或者不同的函数相交点恰好在一个时刻
                pass

        #把函数的顶点和中点作为点位置
        position = {}
        for acc  in self.data.access_names:
            (tk_in, tk_out) = self.data.passes_log[acc][0]#这里只能允许一个星过境一次, 不够一般性
            y = self.data.df_align.query(" time >={} and time<={}".format(tk_in, tk_out))[acc].max()
            x = math.ceil(((tk_in+tk_out)/2 - 8640)/10)
            position[acc] = (x,y)
        self.position = position


        #from none
        #sub graph heads
        roots = []
        leaves =[]

        for si in self.data.access_names:
            if len(self.s_prev(si))==0 and self.G.out_degree[si] !=0:
                # 这里不等等价为子图个数, 如果相位相同两个初始access, 会出错
                # 过滤掉1个点的子图
                roots.append(si)
            if len(self.s_prev(si))!=0 and self.G.out_degree[si] ==0:
                leaves.append(si)

        # buble sort for each sub-graph
        for i in range(len(roots)):
            for j in range(i+1,len(roots)):
                if self.data.passes_log[roots[i]][0][0] >self.data.passes_log[roots[j]][0][0]:
                    tmp = roots[i]
                    roots[i] = roots[j]
                    roots[j]= tmp

        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                if self.data.passes_log[leaves[i]][0][0] > self.data.passes_log[leaves[j]][0][0]:
                    tmp = leaves[i]
                    leaves[i] = leaves[j]
                    leaves[j] = tmp


        self.roots = roots
        self.leaves = leaves

        print("--> graph finished")
        print(self.G)
        print("-> graph roots num:{}\n {}".format(len(self.roots),self.roots))
        print("-> graph leaves num:{} \n{}".format(len(self.leaves),self.leaves))











    def trave(self):
        trave_list = []
        for head in self.roots:
            trave_list.append( head)
            # trave_list.extend(list(dict(nx.bfs_successors(self.G,head)).values()))
            listoflist = dict(nx.bfs_successors(self.G,head)).values()
            for ls in listoflist:
                trave_list.extend(ls)

        return trave_list


    def run(self):
        print("\n\n--> at run ")
        k={}
        opt={}
        k['none'] = 0
        opt['none']=0
        route={}
        for root in self.roots:#先对根节点(in-degree ==0)的都预处理
            tj,tj_next = self.data.acc2tk[root][0],self.data.acc2tk[root][-1]
            opt[root] = (k['none']* opt['none'] + self.integ(root,tj,tj_next))/(k['none']+1)
            k[root] = k['none'] + 1
            route[root] = 'none'

        for acc in tqdm(self.trave()):
            pre_sis = self.s_prev(acc)
            tj_next = self.data.acc2tk[acc][-1]

            if len(pre_sis)==0  : # 没有前置si,acc即为相位最靠前的卫星

                tj = self.data.acc2tk[acc][0]
                opt[acc] =(k['none']* opt['none'] + self.integ(acc,tj,tj_next))/(k['none']+1)
                k[acc] = k['none']+1
                route[acc] = 'none'

            else:
                tmp_opt = 0
                arg_opt=0
                for pre_si in pre_sis:


                    tj = self.G.edges[pre_si,acc]['weight']
                    acc_last_integ = self.integ(pre_si,tj,self.data.acc2tk[pre_si][-1])
                    integ = self.integ(acc,tj,tj_next)
                    tmp = (k[pre_si]*opt[pre_si] - acc_last_integ + integ)/(k[pre_si]+1)
                    if tmp > tmp_opt:
                        tmp_opt = tmp
                        arg_opt = pre_si

                opt[acc] =tmp_opt
                k[acc] = k[arg_opt]+1
                route[acc] = arg_opt

            # last_acc = acc

        #returns
        self.opt = opt
        self.k = k
        self.route = route

            # print('opt: ',opt)
            # print('k:',k)

    def result_stat(self):

        # 记录所有最大联通子图的可能路径
        print("--> result stat...")
        paths = {} # all paths in all sub graphs
        for leaf in self.leaves:
            paths[leaf]=[]
            paths[leaf].append(leaf)

            end_node = leaf
            while end_node != 'none':
                paths[leaf].append(self.route[end_node])
                end_node = self.route[end_node]
            paths[leaf].reverse()

        candidate_paths={}
        for root in self.roots:
            argmax_leaf = None
            sub_opt = 0
            for leaf in self.leaves:
                if root in paths[leaf] and self.opt[leaf]>sub_opt :
                    sub_opt = self.opt[leaf]
                    argmax_leaf = leaf
            candidate_paths[root] = argmax_leaf

        #选择候选路径中几个最大覆盖的多个路径为最后结果
        solution = []
        hop=[]
        opt_values = []
        for index_node,path in paths.items():
            for root,leaf in candidate_paths.items():
                if leaf ==index_node and root in path:
                    # best paths cover all sub graphs
                    print("-> (sub) path:\n{}".format(path))
                    print("path value: {}, path handover times:{}\n".format(self.opt[leaf],self.k[leaf]))
                    solution.append(path)
                    hop.append(self.k[leaf])
                    opt_values.append(self.opt[leaf])





        # 找到解的分量的交点,后面移动到 data prep 更合适
        inter_tk_dict={}
        cnt = 0
        for path in solution:
            for s_prev,s_next in zip(path[1:-1],path[2:]):
                while True:
                    if self.is_equal(s_prev,s_next,self.data.inter_tks[cnt]):
                        inter_tk_dict[s_prev,s_next] = self.data.inter_tks[cnt]
                        cnt+=1
                        break
                    else:
                        cnt+=1


        #returns
        self.solution = solution
        self.hop = hop
        self.opt_values = opt_values
        self.inter_tk_dict = inter_tk_dict




