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

def test_nested():

    class TestClassA( object ):
        def __init__( self, key ):
            self.foo = 1234
            self.key = key
    class TestClassB( object ):
        def __init__( self, key ):
            self.test_class_a = TestClassA( key )
            self.key = key

    d = {}
    expected = {}
    expected2 = {}
    for i in range( 3 ):
        d[i] = TestClassB( i )
        expected[i] = 1234
        expected2[i] = i

    smart_d = verdict.Dict( d )

    actual = smart_d.test_class_a.foo
    assert actual == expected

    actual = smart_d.key
    assert actual == expected2

    actual = smart_d.test_class_a.key
    assert actual == expected2

########################################################################

def test_nested_method():

    class TestClassA( object ):
        def foo( self, x ):
            return x**2
    class TestClassB( object ):
        def __init__():
            self.test_class_a = TestClassA()

    d = {}
    expected = {}
    for i in range( 3 ):
        d[i] = TestClassB()
        expected[i] = 4

    smart_d = verdict.Dict( d )

    actual = smart_d.test_class_a.foo( 2 )

    assert actual == expected

########################################################################

def test_depth():
    '''Depth of the Dict.
    '''

    d = verdict.Dict( {
        'A' : verdict.Dict( {
            'i' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 1.,
                    'b': 3.,
                } ),
                2 : verdict.Dict( {
                    'a': 5.,
                    'b': 7.,
                } ),
            } ),
            'ii' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 10.,
                    'b': 30.,
                } ),
                2 : verdict.Dict( {
                    'a': 50.,
                    'b': 70.,
                } ),
            } ),
        } ),
        'B' : verdict.Dict( {
            'i' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 2.,
                    'b': 4.,
                } ),
                2 : verdict.Dict( {
                    'a': 6.,
                    'b': 8.,
                } ),
            } ),
            'ii' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 11.,
                    'b': 31.,
                } ),
                2 : verdict.Dict( {
                    'a': 51.,
                    'b': 71.,
                } ),
            } ),
        } ),
    } )

    assert d.depth() == 4

########################################################################

def test_getitem_split():

    d = verdict.Dict({
        'a/b': 1,
        'a' : { 'c': 2 },
        'b' : { 'c': { 'd': 3 } },
    })

    assert d['a/b'] == 1
    assert d['a/c'] == 2
    assert d['b/c/d'] == 3

def test_call_custom_kwargs():

    class TestClassA( object ):
        def foo( self, x ):
            return x**2

    d = verdict.Dict( { 1 : TestClassA(), 2 : TestClassA(), } )

    kwargs = { 1 : { 'x' : 10}, 2 : { 'x' : 100}, }
    actual = d.foo.call_custom_kwargs( kwargs )
    expected = { 1 : 100, 2 : 10000, }
    assert actual == expected

########################################################################

def test_call_iteratively():

    class TestClassA( object ):
        def foo( self, x ):
            return x**2

    d = verdict.Dict( { 1 : TestClassA(), 2 : TestClassA(), } )

    actual = d.foo.call_iteratively( [ 1, 2, ] )
    expected = { 1 : [1, 4,], 2 : [1, 4, ] }
    assert actual == expected

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

########################################################################

def test_divide():

    d = verdict.Dict( { 1 : 2, 2 : 4 } )
    expected = { 1 : 1.0, 2 : 2.0, }
    actual = d/2
    assert actual == expected

    d = verdict.Dict( { 1 : 2, 2 : 4 } )
    expected = { 1 : 2, 2 : 1, }
    actual = 4 // d
    assert actual == expected

########################################################################

def test_divide_verdict_dict():

    d1 = verdict.Dict( { 1 : 9, 2 : 4 } )
    d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

    expected = { 1 : 3, 2 : 2, }

    actual = d1/d2
    assert actual == expected

########################################################################

def test_add():

    d = verdict.Dict( { 1 : 1, 2 : 2 } )
    expected = { 1 : 2, 2 : 3, }
    actual = d + 1
    assert actual == expected

    d = verdict.Dict( { 1 : 1, 2 : 2 } )
    expected = { 1 : 2, 2 : 3, }
    actual = 1 + d
    assert actual == expected

########################################################################

def test_add_verdict_dict():

    d1 = verdict.Dict( { 1 : 9, 2 : 4 } )
    d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

    expected = { 1 : 12, 2 : 6, }

    actual = d1 + d2
    assert actual == expected

########################################################################

def test_subtract():

    d = verdict.Dict( { 1 : 1, 2 : 2 } )
    expected = { 1 : 0, 2 : 1, }
    actual = d - 1
    assert actual == expected

    d = verdict.Dict( { 1 : 1, 2 : 2 } )
    expected = { 1 : 0, 2 : -1, }
    actual = 1 - d
    assert actual == expected

########################################################################

def test_subtract_verdict_dict():

    d1 = verdict.Dict( { 1 : 9, 2 : 4 } )
    d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

    expected = { 1 : 6, 2 : 2, }

    actual = d1 - d2
    assert actual == expected

########################################################################

def test_sum_contents():

    d1 = verdict.Dict( { 1 : 1, 2 : 2, 3 : 3, } )

    assert 6 == d1.sum_contents()

########################################################################

def test_keymax_and_keymin():

    d1 = verdict.Dict( { 1 : 1, 2 : 2, 3 : 3, } )

    assert (3, 3) == d1.keymax()
    assert (1, 1) == d1.keymin()

    d1 = verdict.Dict( { 5 : 4, 2 : -1, 3 : 0, } )

    assert (5, 4) == d1.keymax()
    assert (2, -1) == d1.keymin()

########################################################################

def test_transpose():

    d = verdict.Dict( {
        'a': { 1: 1, 2: 2, },
        'b': { 1: 3, 2: 4, },
    } )

    expected = verdict.Dict( {
        1: { 'a': 1, 'b': 3, },
        2: { 'a': 2, 'b': 4, },
    } )

    assert d.transpose() == expected

########################################################################

def test_array():

    d = verdict.Dict( {
        'a': 1,
        'c': 2,
        'b': 3,
    } )

    expected = np.array([ 1., 3., 2. ])

    actual = d.array()

    npt.assert_allclose( expected, actual )

########################################################################

def test_to_df():

    d = verdict.Dict( {
        1 : verdict.Dict( {
            'a': 1,
            'b': 2,
        } ),
        2 : verdict.Dict( {
            'a': 3,
            'b': 4,
        } ),
    } )

    expected = pd.DataFrame( {
        'name': [ 'a', 'b' ],
        1: [ 1, 2, ],
        2: [ 3, 4, ],
    } )
    expected.set_index( 'name', inplace=True )

    actual = d.to_df()

    assert actual.equals( expected )

########################################################################

def test_to_df_nested():
    '''Test that converting to a DF for nested Dicts only converts the
    innermost.
    '''

    d = verdict.Dict( {
        'A' : verdict.Dict( {
            'i' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 1.,
                    'b': 3.,
                } ),
                2 : verdict.Dict( {
                    'a': 5.,
                    'b': 7.,
                } ),
            } ),
            'ii' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 10.,
                    'b': 30.,
                } ),
                2 : verdict.Dict( {
                    'a': 50.,
                    'b': 70.,
                } ),
            } ),
        } ),
        'B' : verdict.Dict( {
            'i' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 2.,
                    'b': 4.,
                } ),
                2 : verdict.Dict( {
                    'a': 6.,
                    'b': 8.,
                } ),
            } ),
            'ii' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': 11.,
                    'b': 31.,
                } ),
                2 : verdict.Dict( {
                    'a': 51.,
                    'b': 71.,
                } ),
            } ),
        } ),
    } )

    expected = verdict.Dict( {
        'A' : verdict.Dict( {
            'i' : d['A']['i'].to_df(),
            'ii' : d['A']['ii'].to_df(),
        } ),
        'B' : verdict.Dict( {
            'i' : d['B']['i'].to_df(),
            'ii' : d['B']['ii'].to_df(),
        } ),
    } )

    actual = d.to_df()

    for key, item in expected.items():
        for i_key, i_item in item.items():
            assert i_item.equals( actual[key][i_key] )

