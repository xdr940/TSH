import unittest
from components.dataloader import Tik
class TestTik(unittest.TestCase):
    def test_tik(self):
        tiks =[]
        tik = Tik(stamp=8640,passIn=['s2420','s2520'])
        tik.classify()
        print(tik)
        pass
