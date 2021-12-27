
import networkx as nx
import pandas as pd
import math
class TimeStamp:
    def __init__(self):
        pass


class Accessor:
    def __init__(self,data,alg_base="Max - Range (km)"):
        self.data = data
        self.alg_base =alg_base
        self.position=None
        pass


    def run(self):
        k={}
        opt={}
        k['none'] = 0
        opt[self.data.total_tks[0]]=0


        # build direc graph


        pass
        # for
        # opt_tab = {}
        # opt_route = {}
        # opt_route[('s2520',8640)] = 'none'
        # opt_route[('s2420',8640)] = 'none'


    def opt(self,si,tj):
        # if self.s_prev(si,tj)==None:#如果前面没有卫星切过来的



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


    def s_prev(self,i_access,tj=None):
        '''

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
                if tk in self.tks_half(acc,'last'):
                    if acc not in prevs :
                        prevs.append(acc)
                else:
                    continue
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

        #build the graph whose access as the node
        self.G = nx.DiGraph(date='2021-12-22', name='handover')
        self.G.add_nodes_from(self.data.access_names)
        for time in self.data.total_tks[1:]:
            accs = self.data.tk2acc[time]
            if len(accs) ==1 : #初始点或者结束点
                continue
            else:
                if accs[0] in self.s_prev(accs[1]):
                    self.G.add_weighted_edges_from([(accs[0], accs[1], time)])
                    print(accs[0], accs[1], time)
                elif accs[1] in self.s_prev(accs[0]):
                    self.G.add_weighted_edges_from([(accs[1], accs[0], time)])
                    print(accs[1], accs[0], time)


        #把函数的顶点和中点作为点位置
        position = {}
        for acc  in self.data.access_names:
            (tk_in, tk_out) = self.data.passes_log[acc][0]#这里只能允许一个星过境一次, 不够一般性
            y = self.data.df_align.query(" time >={} and time<={}".format(tk_in, tk_out))[acc].max()
            x = math.ceil(((tk_in+tk_out)/2 - 8640)/10)
            position[acc] = (x,y)
        self.position = position


        #from none
        from_none = []
        for si in self.data.access_names:
            if len(self.s_prev(si))==0:
                from_none.append(si)
        # buble sort for each sub-graph
        for i in range(len(from_none)):
            for j in range(i+1,len(from_none)):
                if self.data.passes_log[from_none[i]][0][0] >self.data.passes_log[from_none[j]][0][0]:
                    tmp = from_none[i]
                    from_none[i] = from_none[j]
                    from_none[j]= tmp
        print(from_none)

        self.from_none = from_none





        print(self.G)


    def trave(self):
        trave_list = []
        for head in self.from_none:
            trave_list.append( head)
            # trave_list.extend(list(dict(nx.bfs_successors(self.G,head)).values()))
            listoflist = dict(nx.bfs_successors(self.G,head)).values()
            for ls in listoflist:
                trave_list.extend(ls)

        return trave_list


