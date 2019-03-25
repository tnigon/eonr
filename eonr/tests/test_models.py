# -*- coding: utf-8 -*-
import unittest

import numpy as np
try:
    from eonr import EONR
except ImportError:
    import sys
    sys.path.append(".")
    from eonr import EONR

class Test_Models(unittest.TestCase):

    def setUp(self):
        self.my_eonr = EONR()

    def test_update_eonr_R(self):
        R = np.random.rand(1)
        self.my_eonr.R = R
        self.my_eonr.models.update_eonr(self.my_eonr)
        self.assertEqual(self.my_eonr.R, self.my_eonr.models.R)

    def test_update_eonr_cost_n_fert(self):
        cost_n_fert = np.random.rand(1)
        self.my_eonr.cost_n_fert = cost_n_fert
        self.my_eonr.models.update_eonr(self.my_eonr)
        self.assertEqual(self.my_eonr.cost_n_fert,
                         self.my_eonr.models.cost_n_fert)

    def test_update_eonr_coefs_grtn(self):
        coefs_grtn = {}  # TODO: use data to generate
        self.my_eonr.coefs_grtn = coefs_grtn
        self.my_eonr.models.update_eonr(self.my_eonr)
        self.assertEqual(self.my_eonr.coefs_grtn,
                         self.my_eonr.models.coefs_grtn)

    def test_update_eonr_coefs_social(self):
        coefs_social = {}  # TODO: use data to generate
        self.my_eonr.coefs_social = coefs_social
        self.my_eonr.models.update_eonr(self.my_eonr)
        self.assertEqual(self.my_eonr.coefs_social,
                         self.my_eonr.models.coefs_social)

    # TODO: add more test functions

if __name__ == '__main__':
    unittest.main()
