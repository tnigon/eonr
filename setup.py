# -*- coding: utf-8 -*-
"""
© 2018 Regents of the University of Minnesota. All rights reserved.
"""
__copyright__ = 'Regents of the University of Minnesota. All rights reserved.'
__author__ = 'Tyler Nigon'
__license__ = (
        '"EONR" is copyrighted by the Regents of the University of Minnesota.'
        'It can be freely used for educational and research purposes by '
        'non-profit institutions and US government agencies only. Other '
        'organizations are allowed to use "EONR" only for evaluation '
        'purposes, and any further uses will require prior approval. The '
        'software may not be sold or redistributed without prior approval. '
        'One may make copies of the software for their use provided that the '
        'copies are not sold or distributed, are used under the same terms '
        'and conditions.'
        'As unestablished research software, this code is provided on an '
        '"as is" basis without warranty of any kind, either expressed or '
        'implied. The downloading, or executing any part of this software '
        'constitutes an implicit agreement to these terms. These terms and '
        'conditions are subject to change at any time without prior notice.')
__email__ = 'nigo0024@umn.edu'


import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

def history():
    with open('HISTORY.md') as history_file:
        return history_file.read()

requirements = [
    'matplotlib',
    'numpy',
    'pandas',
    'scikits.bootstrap',
    'scipy',
    'seaborn',
    'uncertainties'
]

test_requirements = [
    # TODO: put package test requirements here
]

setuptools.setup(name='EONR',
                 version='1.0',
                 description='A tool for calculating economic optimum nitrogen rates',
                 long_description=readme(),
                 long_description_content_type="text/markdown",
                 url='https://github.com/tnigon/eonr',
                 author='Tyler J. Nigon',
                 author_email='nigo0024@umn.edu',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 classifiers=[
                         'Development Status :: 2 - Pre-Alpha',
                         'Intended Audience :: Science/Research',
                         #        'License :: OSI Approved :: ISC License (ISCL)',
                         'Natural Language :: English',
                         'Operating System :: Microsoft :: Windows',
                         'Programming Language :: Python :: 3',
                         ],
                include_package_data=True,
                install_requires=requirements,
                test_suite='tests',
                tests_require=test_requirements,
                zip_safe=False)
