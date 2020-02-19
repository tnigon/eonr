# -*- coding: utf-8 -*-
'''
Set up a test to run through all of the functions (and various options) and
provide a report on how many were returned with errors (coverage).

Then this file can be run anytime a change is made and changes are pushed to
see if anything was broken

The following test uses a very small "test" datacube that is only 3x3x240
(8,640 bytes)
'''
import numpy as np
import os
import shutil, tempfile
import unittest

from eonr import EONR


NAME_DATA = r'minnesota_2012.csv'
FILENAME_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', NAME_DATA)
# print(FILENAME_HDR)
if not os.path.isfile(FILENAME_DATA):
    FILENAME_DATA = os.path.join(os.path.dirname(os.getcwd()), 'data', NAME_DATA)


class Test_Models_update_eonr(unittest.TestCase):

    def setUp(self):
        self.my_eonr = EONR()

    def tearDown(self):
        self.my_eonr = None

    def test_update_eonr_R(self):
        R = np.random.rand(1)[0]
        self.my_eonr.R = R
        self.my_eonr.models.update_eonr(self.my_eonr)
        self.assertEqual(self.my_eonr.R, self.my_eonr.models.R)

    def test_update_eonr_cost_n_fert(self):
        cost_n_fert = np.random.rand(1)[0]
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


def suite():
    suite = unittest.TestSuite()

    # Test_Models_update_eonr
    suite.addTest(Test_Models_update_eonr('test_update_eonr_R'))
    suite.addTest(Test_Models_update_eonr('test_update_eonr_cost_n_fert'))
    suite.addTest(Test_Models_update_eonr('test_update_eonr_coefs_grtn'))
    suite.addTest(Test_Models_update_eonr('test_update_eonr_coefs_social'))


    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())