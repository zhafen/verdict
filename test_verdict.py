'''Testing for verdict.py
'''

from mock import patch
import h5py
import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import copy
import unittest

import verdict

########################################################################
########################################################################

def test_multiply():

    d = verdict.Dict( { 1 : 1, 2 : 2 } )

    expected = { 1 : 2, 2 : 4, }

    actual = d*2
    assert actual == expected

    actual = 2*d
    assert actual == expected

########################################################################

def test_multiply_verdict_dict():

    d1 = verdict.Dict( { 1 : 1, 2 : 2 } )
    d2 = verdict.Dict( { 1 : 2, 2 : 3 } )

    expected = { 1 : 2, 2 : 6, }

    actual = d1*d2
    assert actual == expected

    actual = d2*d1
    assert actual == expected

