# -*- coding: utf-8 -*-
"""
Copyright &copy; 2019 Tyler J Nigon. All rights reserved.
"""

__copyright__ = '2019 Tyler J Nigon. All rights reserved.'
__author__ = 'Tyler J Nigon'
__license__ = (
        'The MIT license'

        'Permission is hereby granted, free of charge, to any person '
        'obtaining a copy of this software and associated documentation files '
        '(the "Software"), to deal in the Software without restriction, '
        'including without limitation the rights to use, copy, modify, merge, '
        'publish, distribute, sublicense, and/or sell copies of the Software, '
        'and to permit persons to whom the Software is furnished to do so, '
        'subject to the following conditions:'

        'The above copyright notice and this permission notice shall be '
        'included in all copies or substantial portions of the Software.'

        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, '
        'EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF '
        'MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND '
        'NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS '
        'BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN '
        'ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN '
        'CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE '
        'SOFTWARE.')
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
                 version='1.0.0',
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
