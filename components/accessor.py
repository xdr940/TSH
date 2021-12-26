
import networkx as nx
import pandas as pd
class TimeStamp:
    def __init__(self):
        pass


class Accessor:
    def __init__(self,data,alg_base="Max - Range (km)"):
        self.data = data
        self.alg_base =alg_base
        pass


    def run(self):
        opt_tab = {}
        opt_route = {}
        opt_route[('s2520',8640)] = 'none'
        opt_route[('s2420',8640)] = 'none'


    def opt(self,si,tj):
        if self.s_prev(si,tj)==None:#如果前面没有卫星切过来的



        pass



    def t_prev(self,i_access,tj):
        '''
        tj在si_prev的后面,and tj在si的前面, return si_prev last half
        :param i_access:
        :param tj:
        :return:
        '''
        for acc in self.data.tk2acc[tj] :# 顶多两次
            if acc ==i_access:
                continue
            else:
                return self.tks_half(acc,'first')

        pass
    def t_next(self,i_access,tj):
        '''

        :param i_access:
        :param tj:
        :return:
        '''
        return self.tks_half(i_access,'last')


    def s_prev(self,i_access,tj):
        '''

        :param i_access:
        :param tj: 一般不需要, 如果时间跨度很长, 卫星可能转两圈的时候就需要
        :return:
        '''
        prevs = []
        first_half = self.tks_half(i_access,'first')
        first_half = set(first_half)
        for i_other in self.data.access_names:
            if i_other == i_access:
                continue
            if len(set(self.tks_half(i_other,'first'))&first_half)!=0:
                prevs.append(i_other)
        pass

    def s_next(self,i_access,tj):
        '''
        只选择前半段在本接入星后半段的
        :param i_access:
        :param tj:  一般不需要, 如果时间跨度很长, 卫星可能转两圈的时候就需要
        :return:
        '''
        nexts=[]
        last_half = self.tks_half(i_access,'last')
        last_half = set(last_half)
        for i_other in self.data.access_names:
            if i_other == i_access:
                continue
            if len(set(self.tks_half(i_other,'first'))&last_half)!=0:
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

    def build_graph(self):
        G_acc = {}
        for tin in self.data.total_tks:
            for acc in self.data.tk2acc[tin]:
                if tin not in self.data.acc2tk[acc]:
                    continue
                for tout in self.data.acc2tk[acc]:
                    if tin >= tout:
                        continue
                    if (tin, tout) not in G_acc:
                        G_acc[tin, tout] = []
                    G_acc[tin, tout].append(acc)
                    break
        G_weight = {}

        self.nx_G = nx.MultiDiGraph(date='2021-12-22', name='handover')
        self.nx_G.add_nodes_from(self.data.total_tks)
        self.nx_G.add_edges_from(G_acc.keys())



