# -*- coding: utf-8 -*-
__copyright__ = '2019-2020 Tyler J Nigon. All rights reserved.'
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


from eonr.models import Models
from eonr.plotting_tools import Plotting_tools
from eonr.eonr import EONR

name = 'eonr'
__version__ = '0.2.1'

__all__ = ['EONR',
           'Models',
           'Plotting_tools']
