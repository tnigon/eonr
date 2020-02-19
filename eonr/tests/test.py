# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:17:40 2020

@author: nigo0024

This test runner allows me to load in each of the test modules, then load all
the tests from each of those modules into a test suite before runnning them all
"""

import unittest

import test_eonr
import test_models
import test_settings

# initialize
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_eonr))
suite.addTests(loader.loadTestsFromModule(test_models))
suite.addTests(loader.loadTestsFromModule(test_settings))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)

#fname_hdr = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr'
#fname_hdr_spec = r'F:\\nigo0024\Documents\hs_process_demo\Wells_rep2_20180628_16h56m_pika_gige_7_plot_611-cube-to-spec-mean.spec.hdr'
#runner = unittest.TextTestRunner(verbosity=2)
#runner.run(test_hsio.suite())
