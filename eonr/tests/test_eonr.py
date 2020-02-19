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
import pandas as pd
import shutil, tempfile
import unittest

from eonr import EONR


NAME_DATA = r'minnesota_2012.csv'
FILENAME_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', NAME_DATA)
# print(FILENAME_HDR)
if not os.path.isfile(FILENAME_DATA):
    FILENAME_DATA = os.path.join(os.path.dirname(os.getcwd()), 'data', NAME_DATA)


class Test_EONR_set_init_vs_post(unittest.TestCase):
    '''
    Test to be sure results are the same if setting variables before or
    after initialization
    '''
    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        model = None
        col_n_app = 'rate_n_applied_kgha'
        col_yld = 'yld_grain_dry_kgha'
        unit_currency = 'USD'
        unit_fert = 'kg'
        unit_grain = 'kg'
        unit_area = 'ha'
        cost_n_fert = 0.88
        price_grain = 0.157
        self.df_data = pd.read_csv(FILENAME_DATA)

        self.my_eonr1 = EONR(base_dir=self.test_dir, model=model)
        self.my_eonr1.set_column_names(col_yld=col_yld,
                                       col_n_app=col_n_app)
        self.my_eonr1.set_units(unit_currency=unit_currency, unit_fert=unit_fert,
                                unit_grain=unit_grain, unit_area=unit_area)
        self.my_eonr1.update_econ(cost_n_fert=cost_n_fert,
                                  price_grain=price_grain)
        self.my_eonr1.calculate_eonr(self.df_data)
        self.my_eonr2 = EONR(cost_n_fert=cost_n_fert,
                             price_grain=price_grain,
                             col_n_app=col_n_app,
                             col_yld=col_yld,
                             unit_currency=unit_currency,
                             unit_grain=unit_grain,
                             unit_fert=unit_fert,
                             unit_area=unit_area,
                             model=None,
                             base_dir=self.test_dir)
        self.my_eonr2.calculate_eonr(self.df_data)

    @classmethod
    def tearDownClass(self):
        self.my_eonr1 = None
        self.my_eonr2 = None
        self.df_data = None
        shutil.rmtree(self.test_dir)
        self.test_dir = None

    def test_init_eonr(self):
        self.assertAlmostEqual(self.my_eonr1.eonr, 162.3, 1,
                               'init EONR result is not correct')
    def test_init_mrtn(self):
        self.assertAlmostEqual(self.my_eonr1.mrtn, 767.93, 2,
                               'init MRTN result is not correct')
    def test_init_crit_x(self):
        self.assertAlmostEqual(self.my_eonr1.coefs_grtn['crit_x'], 177.440, 1,
                               'init crit_x result is not correct')
    def test_init_pl_l_90(self):
        pl_l_90 = self.my_eonr1.df_ci[self.my_eonr1.df_ci['level'] == 0.9]['pl_l'].item()
        self.assertAlmostEqual(pl_l_90, 130.46, 1,
                               'init profile lower 90th result is not correct')
    def test_init_pl_u_90(self):
        pl_u_90 = self.my_eonr1.df_ci[self.my_eonr1.df_ci['level'] == 0.9]['pl_u'].item()
        self.assertAlmostEqual(pl_u_90, 207.83, 1,
                               'init profile upper 90th result is not correct')
    def test_init_wald_l_90(self):
        wald_l_90 = self.my_eonr1.df_ci[self.my_eonr1.df_ci['level'] == 0.9]['wald_l'].item()
        self.assertAlmostEqual(wald_l_90, 107.489, 1,
                               'init wald lower 90th result is not correct')
    def test_init_wald_u_90(self):
        wald_u_90 = self.my_eonr1.df_ci[self.my_eonr1.df_ci['level'] == 0.9]['wald_u'].item()
        self.assertAlmostEqual(wald_u_90, 217.237, 1,
                               'init wald upper 90th result is not correct')

    def test_post_eonr(self):
        self.assertAlmostEqual(self.my_eonr2.eonr, 162.3, 1,
                               'post EONR result is not correct')
    def test_post_mrtn(self):
        self.assertAlmostEqual(self.my_eonr2.mrtn, 767.93, 2,
                               'post MRTN result is not correct')
    def test_post_crit_x(self):
        self.assertAlmostEqual(self.my_eonr2.coefs_grtn['crit_x'], 177.440, 1,
                               'post crit_x result is not correct')
    def test_post_pl_l_90(self):
        pl_l_90 = self.my_eonr2.df_ci[self.my_eonr2.df_ci['level'] == 0.9]['pl_l'].item()
        self.assertAlmostEqual(pl_l_90, 130.46, 1,
                               'post profile lower 90th result is not correct')
    def test_post_pl_u_90(self):
        pl_u_90 = self.my_eonr2.df_ci[self.my_eonr2.df_ci['level'] == 0.9]['pl_u'].item()
        self.assertAlmostEqual(pl_u_90, 207.83, 1,
                               'post profile upper 90th result is not correct')
    def test_post_wald_l_90(self):
        wald_l_90 = self.my_eonr2.df_ci[self.my_eonr2.df_ci['level'] == 0.9]['wald_l'].item()
        self.assertAlmostEqual(wald_l_90, 107.489, 1,
                               'post wald lower 90th result is not correct')
    def test_post_wald_u_90(self):
        wald_u_90 = self.my_eonr2.df_ci[self.my_eonr2.df_ci['level'] == 0.9]['wald_u'].item()
        self.assertAlmostEqual(wald_u_90, 217.237, 1,
                               'post wald upper 90th result is not correct')

    def test_init_post_col_n_app(self):
        self.assertEqual(self.my_eonr1.col_n_app, self.my_eonr2.col_n_app,
                         '"col_n_app" is not the same between "init" and "post" instance.')
    def test_init_post_col_yld(self):
        self.assertEqual(self.my_eonr1.col_yld, self.my_eonr2.col_yld,
                         '"col_yld" is not the same between "init" and "post" instance.')
    def test_init_post_unit_currency(self):
        self.assertEqual(self.my_eonr1.unit_currency, self.my_eonr2.unit_currency,
                         '"unit_currency" is not the same between "init" and "post" instance.')
    def test_init_post_unit_fert(self):
        self.assertEqual(self.my_eonr1.unit_fert, self.my_eonr2.unit_fert,
                         '"unit_fert" is not the same between "init" and "post" instance.')
    def test_init_post_unit_grain(self):
        self.assertEqual(self.my_eonr1.unit_grain, self.my_eonr2.unit_grain,
                         '"unit_grain" is not the same between "init" and "post" instance.')
    def test_init_post_unit_area(self):
        self.assertEqual(self.my_eonr1.unit_area, self.my_eonr2.unit_area,
                         '"unit_area" is not the same between "init" and "post" instance.')
    def test_init_post_cost_n_fert(self):
        self.assertEqual(self.my_eonr1.cost_n_fert, self.my_eonr2.cost_n_fert,
                         '"cost_n_fert" is not the same between "init" and "post" instance.')
    def test_init_post_price_grain(self):
        self.assertEqual(self.my_eonr1.price_grain, self.my_eonr2.price_grain,
                         '"price_grain" is not the same between "init" and "post" instance.')


class Test_EONR_model_results(unittest.TestCase):
    '''
    Test to be sure results are the same if setting variables before or
    after initialization
    '''
    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()
        col_n_app = 'rate_n_applied_kgha'
        col_yld = 'yld_grain_dry_kgha'
        unit_currency = 'USD'
        unit_fert = 'kg'
        unit_grain = 'kg'
        unit_area = 'ha'
        cost_n_fert = 0.88
        price_grain = 0.157
        self.df_data = pd.read_csv(FILENAME_DATA)

        self.my_eonr_q = EONR(cost_n_fert=cost_n_fert,
                              price_grain=price_grain,
                              col_n_app=col_n_app,
                              col_yld=col_yld,
                              unit_currency=unit_currency,
                              unit_grain=unit_grain,
                              unit_fert=unit_fert,
                              unit_area=unit_area,
                              model='quadratic',
                              base_dir=self.test_dir)
        self.my_eonr_q.calculate_eonr(self.df_data)

    @classmethod
    def tearDownClass(self):
        self.my_eonr_q = None
        self.df_data = None
        shutil.rmtree(self.test_dir)
        self.test_dir = None

    def test_quadratic_eonr(self):
        self.assertAlmostEqual(self.my_eonr_q.eonr, 174.238, 1,
                               'quadratic EONR result is not correct')
    def test_quadratic_mrtn(self):
        self.assertAlmostEqual(self.my_eonr_q.mrtn, 770.206, 1,
                               'quadratic MRTN result is not correct')
    def test_quadratic_crit_x(self):
        self.assertAlmostEqual(self.my_eonr_q.coefs_grtn['crit_x'], 191.581, 1,
                               'quadratic crit_x result is not correct')
    def test_quadratic_pl_l_90(self):
        pl_l_90 = self.my_eonr_q.df_ci[self.my_eonr_q.df_ci['level'] == 0.9]['pl_l'].item()
        self.assertAlmostEqual(pl_l_90, 152.882, 1,
                               'quadratic profile lower 90th result is not correct')
    def test_quadratic_pl_u_90(self):
        pl_u_90 = self.my_eonr_q.df_ci[self.my_eonr_q.df_ci['level'] == 0.9]['pl_u'].item()
        self.assertAlmostEqual(pl_u_90, 222.560, 1,
                               'quadratic profile upper 90th result is not correct')
    def test_quadratic_wald_l_90(self):
        wald_l_90 = self.my_eonr_q.df_ci[self.my_eonr_q.df_ci['level'] == 0.9]['wald_l'].item()
        self.assertAlmostEqual(wald_l_90, 132.785, 1,
                               'quadratic wald lower 90th result is not correct')
    def test_quadratic_wald_u_90(self):
        wald_u_90 = self.my_eonr_q.df_ci[self.my_eonr_q.df_ci['level'] == 0.9]['wald_u'].item()
        self.assertAlmostEqual(wald_u_90, 215.691, 1,
                               'quadratic wald upper 90th result is not correct')

def suite():
    suite = unittest.TestSuite()

    # Test_EONR_set_column_names_init
    suite.addTest(Test_EONR_set_init_vs_post('test_init_eonr'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_mrtn'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_crit_x'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_pl_l_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_pl_u_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_wald_l_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_wald_u_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_eonr'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_mrtn'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_crit_x'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_pl_l_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_pl_u_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_wald_l_90'))
    suite.addTest(Test_EONR_set_init_vs_post('test_post_wald_u_90'))

    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_col_n_app'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_col_yld'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_unit_currency'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_unit_fert'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_unit_grain'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_unit_area'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_cost_n_fert'))
    suite.addTest(Test_EONR_set_init_vs_post('test_init_post_price_grain'))



    suite.addTest(Test_EONR_model_results('test_quadratic_eonr'))
    suite.addTest(Test_EONR_model_results('test_quadratic_mrtn'))
    suite.addTest(Test_EONR_model_results('test_quadratic_crit_x'))
    suite.addTest(Test_EONR_model_results('test_quadratic_pl_l_90'))
    suite.addTest(Test_EONR_model_results('test_quadratic_pl_u_90'))
    suite.addTest(Test_EONR_model_results('test_quadratic_wald_l_90'))
    suite.addTest(Test_EONR_model_results('test_quadratic_wald_u_90'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())