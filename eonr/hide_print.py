# -*- coding: utf-8 -*-
"""
© 2019 Regents of the University of Minnesota. All rights reserved.

EONR is copyrighted by the Regents of the University of Minnesota. It can
be freely used for educational and research purposes by non-profit
institutions and US government agencies only. Other organizations are allowed
to use EONR only for evaluation purposes, and any further uses will require
prior approval. The software may not be sold or redistributed without prior
approval. One may make copies of the software for their use provided that the
copies, are not sold or distributed, are used under the same terms and
conditions.
As unestablished research software, this code is provided on an "as is" basis
without warranty of any kind, either expressed or implied. The downloading, or
executing any part of this software constitutes an implicit agreement to these
terms. These terms and conditions are subject to change at any time without
prior notice.
"""
import os
import sys


class Hide_print(object):
    '''
    Disables printing temporarily (used by get_likelihood()
    optimization)
    Use:
        with _hide_print():
            print('This will not print')
            <other code you don't want to print>
    '''
    def __init__(self):
        self.active = False

    def __enter__(self):
        self.active = True
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        self.active = False
