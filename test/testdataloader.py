import unittest
from utils.yaml_wrapper import YamlHandler
import argparse



from dataset.dataloader import AerDataset


parser = argparse.ArgumentParser(description="stk-conn")
parser.add_argument("--settings", default='../configs/config.yaml')
args = parser.parse_args()



class TestDataloader(unittest.TestCase):

    def test_alg(self):
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


        # assert ed-tk2acc
        self.assertEqual(data.tk2acc[8640],['ss3-To-s2420','ss3-To-s2520'])
        self.assertEqual(data.tk2acc[8692],['ss3-To-s5113'])
        self.assertEqual(data.tk2acc[8730],['ss3-To-s5013'])
        self.assertEqual(data.tk2acc[8850],['ss3-To-s2519'])
        self.assertEqual(data.tk2acc[8956],['ss3-To-s5112'])
        self.assertEqual(data.tk2acc[9005],['ss3-To-s5012'])
        self.assertEqual(data.tk2acc[9084],['ss3-To-s2618'])
        self.assertEqual(data.tk2acc[9121],['ss3-To-s2518'])

        #assert inter-tk2acc

        self.assertEqual(data.tk2acc[8687],['ss3-To-s2420','ss3-To-s2520'])
        self.assertEqual(data.tk2acc[8732],['ss3-To-s2420','ss3-To-s5113'])
        self.assertEqual(data.tk2acc[8747],['ss3-To-s2420','ss3-To-s5013'])
        self.assertEqual(data.tk2acc[8761],['ss3-To-s5013','ss3-To-s5113'])
        self.assertEqual(data.tk2acc[8862],['ss3-To-s2519','ss3-To-s5013'])


        #assert acc2tk
        self.assertEqual(data.acc2tk['ss3-To-s2420'],[8640,8687,8732,8747,8765])
        self.assertEqual(data.acc2tk['ss3-To-s5013'],[8730,8747,8761,8862,8875])









