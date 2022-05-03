import unittest
from utils.yaml_wrapper import YamlHandler
import argparse

from components.dataloader import AerDataset
from components.solver import DpSolver

parser = argparse.ArgumentParser(description="stk-conn")
parser.add_argument("--settings", default='../configs/config.yaml')
args = parser.parse_args()



class TestDataloader(unittest.TestCase):
    def test_data(self):
        yml = YamlHandler(args.settings)
        config = yml.read_yaml()

        # load data
        data = AerDataset(config)

        # split data
        data.data_prep(config)

        # split data reload and process
        data.load()
        data.data_align()
        data.pre_alg()


        # assert ed-tk2acc
        # self.assertEqual(data.tk2acc[8640],['s2420','s2520'])
        # self.assertEqual(data.tk2acc[8692],['s5113'])
        # self.assertEqual(data.tk2acc[8730],['s5013'])
        # self.assertEqual(data.tk2acc[8850],['s2519'])
        # self.assertEqual(data.tk2acc[8956],['s5112'])
        # self.assertEqual(data.tk2acc[9005],['s5012'])
        # self.assertEqual(data.tk2acc[9084],['s2618'])
        # self.assertEqual(data.tk2acc[9121],['s2518'])
        #
        # #assert inter-tk2acc
        #
        # self.assertEqual(data.tk2acc[8687],['s2420','s2520'])
        # self.assertEqual(data.tk2acc[8732],['s2420','s5113'])
        # self.assertEqual(data.tk2acc[8747],['s2420','s5013'])
        # self.assertEqual(data.tk2acc[8761],['s5013','s5113'])
        # self.assertEqual(data.tk2acc[8862],['s2519','s5013'])
        #
        #
        # #assert acc2tk
        # self.assertEqual(data.acc2tk['s2420'],[8640,8687,8732,8747,8765])
        # self.assertEqual(data.acc2tk['s5013'],[8730,8747,8761,8862,8875])



        data.data_append_value(data)
        #assert build G


    def testAccessor(self):
        yml = YamlHandler(args.settings)
        config = yml.read_yaml()

        # load data
        data = AerDataset(config)

        # split data
        data.data_prep(config)

        # split data reload and process
        data.load()
        data.data_align(config)
        data.pre_alg()

        accessor = Accessor(data)
        accessor.build_graph()
        # print(accessor.trave())

        accessor.run()
        # drawer = Drawer()
        # fig = drawer.drawGraph(accessor.G)
        #
        # accessor.s_prev('s5013')
        # self.assertEqual(accessor.tks_half('s5013','last'),[8862,8875])
        # self.assertEqual(accessor.tks_half('s2517','last'),[9514,9524,9535])
        #
        # self.assertEqual(accessor.s_next('s2517'),['s5110'])

        # accessor.tks_half()

        St=[
            ('s2520',8640),
            ('s2420',8687),
            # ('s5113',8709),
            ('s5013',8747),
            # ('s5013',8761),
            ('s2519',8862),
            ('s5112',8977),
            ('s5012',9045),
            ('s2518',9132),
            ('none',9266),
            ('s2517',9394),
            ('s5110',9514),
            ('s2616',9627)
        ]   #15657

        # print(accessor.opt_value(St))



