# -*- coding: utf-8 -*-
'''
Tests pre-calculation settings (column names, units, prices, costs).
'''
import numpy as np
import unittest
from eonr import EONR


class Test_EONR_set_column_names_init(unittest.TestCase):
    def tearDown(self):
        self.my_eonr = None

    def test_set_col_crop_nup(self):
        col_crop_nup = 'nup_total_kgha'
        self.my_eonr = EONR(col_crop_nup=col_crop_nup)
        self.assertEqual(self.my_eonr.col_crop_nup, col_crop_nup,
                         'init `col_crop_nup` not set correctly.')

    def test_set_col_col_n_avail(self):
        col_n_avail = 'n_available_kgha'
        self.my_eonr = EONR(col_n_avail=col_n_avail)
        self.assertEqual(self.my_eonr.col_n_avail, col_n_avail,
                         'init `col_n_avail` not set correctly.')

    def test_set_col_location(self):
        col_location = 'site'
        self.my_eonr = EONR(col_location=col_location)
        self.assertEqual(self.my_eonr.col_location, col_location,
                         'init `col_location` not set correctly.')

    def test_set_col_n_app(self):
        col_n_app = 'rate_n_applied_kgha'
        self.my_eonr = EONR(col_n_app=col_n_app)
        self.assertEqual(self.my_eonr.col_n_app, col_n_app,
                         'init `col_n_app` not set correctly.')

    def test_set_col_time_n(self):
        col_n_app = 'N timing'
        self.my_eonr = EONR(col_n_app=col_n_app)
        self.assertEqual(self.my_eonr.col_n_app, col_n_app,
                         'init `col_n_app` not set correctly.')

    def test_set_col_year(self):
        col_year = 'date'
        self.my_eonr = EONR(col_year=col_year)
        self.assertEqual(self.my_eonr.col_year, col_year,
                         'init `col_year` not set correctly.')

    def test_set_col_yld(self):
        col_yld  = 'yld_grain_dry_kgha'
        self.my_eonr = EONR(col_yld =col_yld)
        self.assertEqual(self.my_eonr.col_yld , col_yld ,
                         'init `col_yld ` not set correctly.')


class Test_EONR_set_column_names_post(unittest.TestCase):
    def setUp(self):
        self.my_eonr = EONR()

    def tearDown(self):
        self.my_eonr = None

    def test_set_col_crop_nup(self):
        col_crop_nup = 'nup_total_kgha'
        self.my_eonr.set_column_names(col_crop_nup=col_crop_nup)
        self.assertEqual(self.my_eonr.col_crop_nup, col_crop_nup,
                         'post `col_crop_nup` not set correctly.')

    def test_set_col_col_n_avail(self):
        col_n_avail = 'n_available_kgha'
        self.my_eonr.set_column_names(col_n_avail=col_n_avail)
        self.assertEqual(self.my_eonr.col_n_avail, col_n_avail,
                         'post `col_n_avail` not set correctly.')

    def test_set_col_location(self):
        col_location = 'site'
        self.my_eonr.set_column_names(col_location=col_location)
        self.assertEqual(self.my_eonr.col_location, col_location,
                         'post `col_location` not set correctly.')

    def test_set_col_n_app(self):
        col_n_app = 'rate_n_applied_kgha'
        self.my_eonr.set_column_names(col_n_app=col_n_app)
        self.assertEqual(self.my_eonr.col_n_app, col_n_app,
                         'post `col_n_app` not set correctly.')

    def test_set_col_time_n(self):
        col_n_app = 'N timing'
        self.my_eonr.set_column_names(col_n_app=col_n_app)
        self.assertEqual(self.my_eonr.col_n_app, col_n_app,
                         'post `col_n_app` not set correctly.')

    def test_set_col_year(self):
        col_year = 'date'
        self.my_eonr.set_column_names(col_year=col_year)
        self.assertEqual(self.my_eonr.col_year, col_year,
                         'post `col_year` not set correctly.')

    def test_set_col_yld(self):
        col_yld  = 'yld_grain_dry_kgha'
        self.my_eonr.set_column_names(col_yld=col_yld)
        self.assertEqual(self.my_eonr.col_yld , col_yld ,
                         'post `col_yld ` not set correctly.')


class Test_EONR_set_units_init(unittest.TestCase):
    def tearDown(self):
        self.my_eonr = None

    def test_set_unit_currency(self):
        unit_currency = 'USD'
        self.my_eonr = EONR(unit_currency=unit_currency)
        self.assertEqual(self.my_eonr.unit_currency, unit_currency,
                         'init `unit_currency` not set correctly.')

    def test_set_unit_fert(self):
        unit_fert = 'kg'
        self.my_eonr = EONR(unit_fert=unit_fert)
        self.assertEqual(self.my_eonr.unit_fert, unit_fert,
                         'init `unit_fert` not set correctly.')

    def test_set_unit_grain(self):
        unit_grain = 'kg'
        self.my_eonr = EONR(unit_grain=unit_grain)
        self.assertEqual(self.my_eonr.unit_grain, unit_grain,
                         'init `unit_grain` not set correctly.')

    def test_set_unit_area(self):
        unit_area = 'ha'
        self.my_eonr = EONR(unit_area=unit_area)
        self.assertEqual(self.my_eonr.unit_area, unit_area,
                         'init `unit_area` not set correctly.')


class Test_EONR_set_units_post(unittest.TestCase):
    def setUp(self):
        self.my_eonr = EONR()

    def tearDown(self):
        self.my_eonr = None

    def test_set_unit_currency(self):
        unit_currency = 'USD'
        self.my_eonr.set_units(unit_currency=unit_currency)
        self.assertEqual(self.my_eonr.unit_currency, unit_currency,
                         'post `unit_currency` not set correctly.')

    def test_set_unit_fert(self):
        unit_fert = 'kg'
        self.my_eonr.set_units(unit_fert=unit_fert)
        self.assertEqual(self.my_eonr.unit_fert, unit_fert,
                         'post `unit_fert` not set correctly.')

    def test_set_unit_grain(self):
        unit_grain = 'kg'
        self.my_eonr.set_units(unit_grain=unit_grain)
        self.assertEqual(self.my_eonr.unit_grain, unit_grain,
                         'post `unit_grain` not set correctly.')

    def test_set_unit_area(self):
        unit_area = 'ha'
        self.my_eonr.set_units(unit_area=unit_area)
        self.assertEqual(self.my_eonr.unit_area, unit_area,
                         'post `unit_area` not set correctly.')


class Test_EONR_update_econ_init(unittest.TestCase):
    def tearDown(self):
        self.my_eonr = None

    def test_set_cost_n_fert(self):
        cost_n_fert = 0.88
        self.my_eonr = EONR(cost_n_fert=cost_n_fert)
        self.assertEqual(self.my_eonr.cost_n_fert, cost_n_fert,
                         'init `cost_n_fert` not set correctly.')

    def test_set_cost_fixed(self):
        costs_fixed = 7.50
        self.my_eonr = EONR(costs_fixed=costs_fixed)
        self.assertEqual(self.my_eonr.costs_fixed, costs_fixed,
                         'init `costs_fixed` not set correctly.')

    def test_set_cost_n_social(self):
        cost_n_social = 1.0
        self.my_eonr = EONR(cost_n_social=cost_n_social)
        self.assertEqual(self.my_eonr.cost_n_social, cost_n_social,
                         'init `cost_n_social` not set correctly.')

    def test_set_price_grain(self):
        price_grain = 0.157
        self.my_eonr = EONR(price_grain=price_grain)
        self.assertEqual(self.my_eonr.price_grain, price_grain,
                         'init `price_grain` not set correctly.')
    def test_set_price_ratio(self):
        cost_n_fert = 88
        price_grain = 0.157
        self.my_eonr = EONR(cost_n_fert=cost_n_fert, price_grain=price_grain)
        self.assertEqual(self.my_eonr.price_ratio, cost_n_fert / price_grain,
                         'init `price_ratio` not calculated correctly.')


class Test_EONR_update_econ_post(unittest.TestCase):
    def setUp(self):
        self.my_eonr = EONR()

    def tearDown(self):
        self.my_eonr = None

    def test_set_cost_n_fert(self):
        cost_n_fert = 0.88
        self.my_eonr.update_econ(cost_n_fert=cost_n_fert)
        self.assertEqual(self.my_eonr.cost_n_fert, cost_n_fert,
                         'post `cost_n_fert` not set correctly.')

    def test_set_cost_fixed(self):
        costs_fixed = 7.50
        self.my_eonr.update_econ(costs_fixed=costs_fixed)
        self.assertEqual(self.my_eonr.costs_fixed, costs_fixed,
                         'post `costs_fixed` not set correctly.')

    def test_set_cost_n_social(self):
        cost_n_social = 1.0
        self.my_eonr.update_econ(cost_n_social=cost_n_social)
        self.assertEqual(self.my_eonr.cost_n_social, cost_n_social,
                         'post `cost_n_social` not set correctly.')

    def test_set_price_grain(self):
        price_grain = 0.157
        self.my_eonr.update_econ(price_grain=price_grain)
        self.assertEqual(self.my_eonr.price_grain, price_grain,
                         'post `price_grain` not set correctly.')
    def test_set_price_ratio(self):
        cost_n_fert = 88
        price_grain = 0.157
        self.my_eonr.update_econ(cost_n_fert=cost_n_fert,
                                 price_grain=price_grain)
        self.assertEqual(self.my_eonr.price_ratio, cost_n_fert / price_grain,
                         'post `price_ratio` not calculated correctly.')

def suite():
    suite = unittest.TestSuite()

    # Test_EONR_set_column_names_init
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_crop_nup'))
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_col_n_avail'))
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_location'))
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_n_app'))
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_time_n'))
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_year'))
    suite.addTest(Test_EONR_set_column_names_init('test_set_col_yld'))

    # Test_EONR_set_column_names_post
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_crop_nup'))
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_col_n_avail'))
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_location'))
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_n_app'))
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_time_n'))
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_year'))
    suite.addTest(Test_EONR_set_column_names_post('test_set_col_yld'))

    # Test_EONR_set_units_init
    suite.addTest(Test_EONR_set_units_init('test_set_unit_currency'))
    suite.addTest(Test_EONR_set_units_init('test_set_unit_fert'))
    suite.addTest(Test_EONR_set_units_init('test_set_unit_grain'))
    suite.addTest(Test_EONR_set_units_init('test_set_unit_area'))

    # Test_EONR_set_units_post
    suite.addTest(Test_EONR_set_units_post('test_set_unit_currency'))
    suite.addTest(Test_EONR_set_units_post('test_set_unit_fert'))
    suite.addTest(Test_EONR_set_units_post('test_set_unit_grain'))
    suite.addTest(Test_EONR_set_units_post('test_set_unit_area'))

    # Test_EONR_update_econ_init
    suite.addTest(Test_EONR_update_econ_init('test_set_cost_n_fert'))
    suite.addTest(Test_EONR_update_econ_init('test_set_cost_fixed'))
    suite.addTest(Test_EONR_update_econ_init('test_set_cost_n_social'))
    suite.addTest(Test_EONR_update_econ_init('test_set_price_grain'))
    suite.addTest(Test_EONR_update_econ_init('test_set_price_ratio'))

    # Test_EONR_update_econ_post
    suite.addTest(Test_EONR_update_econ_post('test_set_cost_n_fert'))
    suite.addTest(Test_EONR_update_econ_post('test_set_cost_fixed'))
    suite.addTest(Test_EONR_update_econ_post('test_set_cost_n_social'))
    suite.addTest(Test_EONR_update_econ_post('test_set_price_grain'))
    suite.addTest(Test_EONR_update_econ_post('test_set_price_ratio'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())