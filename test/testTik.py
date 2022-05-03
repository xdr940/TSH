import unittest
from components.AcTik import Tik
class TestTik(unittest.TestCase):
    def test_tik(self):
        tiks =[]
        tik = Tik(stamp=8640)
        tik.addPass(addPassIn=['s2520','s2420'])
        tik.rebuild()
        print(tik)
        self.assertEqual(tik.is_inInter('s2420'),False)
        self.assertEqual(tik.is_in('s2420'),True)

