'''Testing for verdict.py
'''

import copy
import json
from mock import patch
import h5py
import numpy as np
import numpy.testing as npt
import os
import pandas as pd
import scipy.sparse as ss
import unittest

import verdict

########################################################################
########################################################################

class TestVerDictStartup( unittest.TestCase ):

    def test_default( self ):

        d = { 'a' : 1, 'b' : 2 }

        verdict_dict = verdict.Dict( d )

        self.assertEqual( verdict_dict['b'], 2 )
        self.assertEqual( len( verdict_dict ), 2 )

    ########################################################################

    def test_nested( self ):

        orig = {
            'A' : {
                'i' : {
                    1 : {
                        'a': 1.,
                        'b': 3.,
                    },
                    2 : {
                        'a': 5.,
                        'b': 7.,
                    },
                },
                'ii' : {
                    1 : {
                        'a': 10.,
                        'b': 30.,
                    },
                    2 : {
                        'a': 50.,
                        'b': 70.,
                    },
                },
            },
            'B' : {
                'i' : {
                    1 : {
                        'a': 2.,
                        'b': 4.,
                    },
                    2 : {
                        'a': 6.,
                        'b': 8.,
                    },
                },
                'ii' : {
                    1 : {
                        'a': 11.,
                        'b': 31.,
                    },
                    2 : {
                        'a': 51.,
                        'b': 71.,
                    },
                },
            },
        }

        d = verdict.Dict( orig )

        for key, item in d.items():
            assert isinstance( item, verdict.Dict )
            for i_key, i_item in item.items():
                    assert isinstance( i_item, verdict.Dict )
                    for ii_key, ii_item in i_item.items():
                        assert isinstance( i_item, verdict.Dict )
                        for iii_key, iii_item in ii_item.items():
                            self.assertEqual(
                                orig[key][i_key][ii_key][iii_key],
                                d[key][i_key][ii_key][iii_key],
                            )


    ########################################################################

    def test_from_defaults_and_variations( self ):

        class TestClassA( object ):
            def __init__( self, a, b ):
                self.a = a
                self.b = b
            def return_contents( self ):
                return self.a, self.b

        defaults = { 'a' : 1, 'b' : 1 }
        variations = {
            1 : {},
            2 : { 'b' : 2 },
        }

        result = verdict.Dict.from_class_and_args( TestClassA, variations, defaults, )

        assert isinstance( result, verdict.Dict )

        expected = { 1 : ( 1, 1, ), 2 : ( 1, 2, ), }
        actual = result.return_contents()
        assert expected == actual

########################################################################

class TestVerDict( unittest.TestCase ):

    def test_nested( self ):

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
        self.assertEqual( expected, actual )

        actual = smart_d.key
        self.assertEqual( expected2, actual )

        actual = smart_d.test_class_a.key
        self.assertEqual( expected2, actual )

    ########################################################################

    def test_nested_method( self ):

        class TestClassA( object ):
            def foo( self, x ):
                return x**2
        class TestClassB( object ):
            def __init__( self ):
                self.test_class_a = TestClassA()

        d = {}
        expected = {}
        for i in range( 3 ):
            d[i] = TestClassB()
            expected[i] = 4

        smart_d = verdict.Dict( d )

        actual = smart_d.test_class_a.foo( 2 )

        self.assertEqual( expected, actual )

    ########################################################################

    def test_depth( self ):
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

    def test_setitem_create( self ):

        d = verdict.Dict({
            'a/b': 1,
            'a' : { 'c': 2 },
        })

        d.setitem( 'standard', 3 )
        d.setitem( 'a', 3, 'c' )
        d.setitem( 'a', 2, 'e' )
        d.setitem( 'b', 3, 'c', 'd' )

        assert d['standard'] == 3
        assert d['a']['c'] == 3
        assert d['a']['e'] == 2
        assert d['b']['c']['d'] == 3

    ########################################################################

    def test_call_custom_kwargs( self ):

        class TestClassA( object ):
            def foo( self, x ):
                return x**2

        d = verdict.Dict( { 1 : TestClassA(), 2 : TestClassA(), } )

        kwargs = { 1 : { 'x' : 10}, 2 : { 'x' : 100}, }
        actual = d.foo.call_custom_kwargs( kwargs )
        expected = { 1 : 100, 2 : 10000, }
        self.assertEqual( expected, actual )

    ########################################################################

    def test_call_iteratively( self ):

        class TestClassA( object ):
            def foo( self, x ):
                return x**2

        d = verdict.Dict( { 1 : TestClassA(), 2 : TestClassA(), } )

        actual = d.foo.call_iteratively( [ 1, 2, ] )
        expected = { 1 : [1, 4,], 2 : [1, 4, ] }
        self.assertEqual( expected, actual )

    ########################################################################

    def test_multiply( self ):

        d = verdict.Dict( { 1 : 1, 2 : 2 } )

        expected = { 1 : 2, 2 : 4, }

        actual = d*2
        self.assertEqual( expected, actual )

        actual = 2*d
        self.assertEqual( expected, actual )

    ########################################################################

    def test_multiply_verdict_dict( self ):

        d1 = verdict.Dict( { 1 : 1, 2 : 2 } )
        d2 = verdict.Dict( { 1 : 2, 2 : 3 } )

        expected = { 1 : 2, 2 : 6, }

        actual = d1*d2
        self.assertEqual( expected, actual )

        actual = d2*d1
        self.assertEqual( expected, actual )

    ########################################################################

    def test_divide( self ):

        d = verdict.Dict( { 1 : 2, 2 : 4 } )
        expected = { 1 : 1.0, 2 : 2.0, }
        actual = d/2
        self.assertEqual( expected, actual )

        d = verdict.Dict( { 1 : 2, 2 : 4 } )
        expected = { 1 : 2, 2 : 1, }
        actual = 4 // d
        self.assertEqual( expected, actual )

    ########################################################################

    def test_divide_verdict_dict( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : 4 } )
        d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

        expected = { 1 : 3, 2 : 2, }

        actual = d1/d2
        self.assertEqual( expected, actual )

    ########################################################################

    def test_add( self ):

        d = verdict.Dict( { 1 : 1, 2 : 2 } )
        expected = { 1 : 2, 2 : 3, }
        actual = d + 1
        self.assertEqual( expected, actual )

        d = verdict.Dict( { 1 : 1, 2 : 2 } )
        expected = { 1 : 2, 2 : 3, }
        actual = 1 + d
        self.assertEqual( expected, actual )

    ########################################################################

    def test_add_verdict_dict( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : 4 } )
        d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

        expected = { 1 : 12, 2 : 6, }

        actual = d1 + d2
        self.assertEqual( expected, actual )

    ########################################################################

    def test_subtract( self ):

        d = verdict.Dict( { 1 : 1, 2 : 2 } )
        expected = { 1 : 0, 2 : 1, }
        actual = d - 1
        self.assertEqual( expected, actual )

        d = verdict.Dict( { 1 : 1, 2 : 2 } )
        expected = { 1 : 0, 2 : -1, }
        actual = 1 - d
        self.assertEqual( expected, actual )

    ########################################################################

    def test_subtract_verdict_dict( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : 4 } )
        d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

        expected = { 1 : 6, 2 : 2, }

        actual = d1 - d2
        self.assertEqual( expected, actual )

    ########################################################################

    def test_diff( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : 2 } )
        d2 = verdict.Dict( { 1 : 3, 2 : 2 } )

        expected = { 1 : ( 9, 3 ), }

        actual = d1.diff( d2 )
        self.assertEqual( expected, actual )

    ########################################################################

    def test_diff_nested( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : { 'a': 1.0, 'b': 2.0, } } )
        d2 = verdict.Dict( { 1 : 3, 2 : { 'a': 1.0, 'b': 2.1, } } )

        expected = { 1 : ( 9, 3 ), 2 : { 'b': ( 2.0, 2.1 ) } }

        actual = d1.diff( d2 )
        self.assertEqual( expected, actual )

    ########################################################################

    def test_diff_value_to_dict( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : 5 } )
        d2 = verdict.Dict( { 1 : 3, 2 : { 'a': 1.0, 'b': 2.1, } } )

        expected = { 1 : ( 9, 3 ), 2 : ( 5, verdict.Dict({ 'a': 1.0, 'b': 2.1, }) ) }

        actual = d1.diff( d2 )
        self.assertEqual( expected, actual )

    ########################################################################

    def test_diff_new( self ):

        d1 = verdict.Dict( { 1 : 9, } )
        d2 = verdict.Dict( { 1 : 3, 2 : { 'a': 1.0, 'b': 2.1, } } )

        expected = { 1 : ( 9, 3 ), 2 : ( None, verdict.Dict({ 'a': 1.0, 'b': 2.1 }) ) }

        actual = d1.diff( d2 )
        self.assertEqual( expected, actual )

    ########################################################################

    def test_diff_deleted( self ):

        d1 = verdict.Dict( { 1 : 9, 2 : { 'a': 1.0, 'b': 2.1, } } )
        d2 = verdict.Dict( { 1 : 3, } )

        expected = { 1 : ( 9, 3 ), 2 : ( verdict.Dict({ 'a': 1.0, 'b': 2.1 }), None ) }

        actual = d1.diff( d2 )
        self.assertEqual( expected, actual )

    ########################################################################

    def test_sum_contents( self ):

        d1 = verdict.Dict( { 1 : 1, 2 : 2, 3 : 3, } )

        self.assertEqual( 6, d1.sum_contents() )

    ########################################################################

    def test_keymax_and_keymin( self ):

        d1 = verdict.Dict( { 1 : 1, 2 : 2, 3 : 3, } )

        self.assertEqual( (3, 3), d1.keymax() )
        self.assertEqual( (1, 1), d1.keymin() )

        d1 = verdict.Dict( { 5 : 4, 2 : -1, 3 : 0, } )

        self.assertEqual( (5, 4), d1.keymax() )
        self.assertEqual( (2, -1), d1.keymin() )

    ########################################################################

    def test_transpose( self ):

        d = verdict.Dict( {
            'a': { 1: 1, 2: 2, },
            'b': { 1: 3, 2: 4, },
        } )

        expected = verdict.Dict( {
            1: { 'a': 1, 'b': 3, },
            2: { 'a': 2, 'b': 4, },
        } )

        self.assertEqual( d.transpose(), expected )

    ########################################################################

    def test_array( self ):

        d = verdict.Dict( {
            'a': 1,
            'c': 2,
            'b': 3,
        } )

        expected = np.array([ 1., 3., 2. ])

        actual = d.array()

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_to_df( self ):

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

    def test_to_df_nested( self ):
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

########################################################################

class TestVerDictHDF5( unittest.TestCase ):

    def setUp( self ):

        self.savefile = 'to_hdf5_test.hdf5'
        self.kwargs = {}

    def tearDown( self ):

        # Delete spurious files
        if os.path.isfile( self.savefile ):
            os.remove( self.savefile )

    ########################################################################

    def test_to_hdf5( self ):

        # Test data
        d = verdict.Dict( {
            1 : verdict.Dict( {
                'a': np.array([ 1., 2. ]),
                'b': np.array([ 3., 4. ]),
            } ),
            2 : verdict.Dict( {
                'a': np.array([ 5., 6. ]),
                'b': np.array([ 7., 8. ]),
            } ),
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    f[str(key)][inner_key][...],
                )

        # Make sure attributes save
        npt.assert_allclose( f.attrs['x'], attrs['x'] )

    ########################################################################

    def test_to_hdf5_additional_nesting( self ):

        # Test data
        d = verdict.Dict( {
            'i' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': np.array([ 1., 2. ]),
                    'b': np.array([ 3., 4. ]),
                } ),
                2 : verdict.Dict( {
                    'a': np.array([ 5., 6. ]),
                    'b': np.array([ 7., 8. ]),
                } ),
            } ),
            'ii' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': np.array([ 10., 20. ]),
                    'b': np.array([ 30., 40. ]),
                } ),
                2 : verdict.Dict( {
                    'a': np.array([ 50., 60. ]),
                    'b': np.array([ 70., 80. ]),
                } ),
            } ),
        } )

        # Try to save
        d.to_hdf5( self.savefile, **self.kwargs )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                for ii_key, ii_item in inner_item.items():
                    npt.assert_allclose(
                        ii_item,
                        f[key][str(inner_key)][ii_key][...],
                    )

    ########################################################################

    def test_to_hdf5_condensed( self ):

        savefile = 'to_hdf5_test.hdf5'

        # Test data
        d = verdict.Dict( {
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
        } )

        # Try to save
        d.to_hdf5( self.savefile, condensed=True, **self.kwargs )

        expected = {
            'i': {
                'name': np.array([ 'a', 'b' ]),
                '1': np.array([ 1., 3., ]),
                '2': np.array([ 5., 7., ]),
            },
            'ii': {
                'name': np.array([ 'a', 'b' ]),
                '1': np.array([ 10., 30., ]),
                '2': np.array([ 50., 70., ]),
            },
        }

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in expected.items():
            for inner_key, inner_item in item.items():
                try:
                    npt.assert_allclose(
                        inner_item,
                        f[key][inner_key][...],
                    )
                except TypeError:
                    assert len( np.setdiff1d(
                        inner_item,
                        f[key][inner_key][...].astype( str ),
                    ) ) == 0

    ########################################################################

    def test_to_hdf5_condensed_shallow( self ):

        savefile = 'to_hdf5_test.hdf5'

        # Test data
        d = verdict.Dict( {
            1 : verdict.Dict( {
                'a': 1.,
                'b': 3.,
            } ),
            2 : verdict.Dict( {
                'a': 5.,
                'b': 7.,
            } ),
        } )

        # Try to save
        d.to_hdf5( self.savefile, condensed=True, **self.kwargs )

        expected = {
            'name': np.array([ 'a', 'b' ]),
            '1': np.array([ 1., 3., ]),
            '2': np.array([ 5., 7., ]),
        }

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in expected.items():
            try:
                npt.assert_allclose(
                    item,
                    f[key][...],
                )
            except TypeError:
                assert len( np.setdiff1d(
                    item,
                    f[key][...].astype( str ),
                ) ) == 0

    ########################################################################

    def test_to_hdf5_single_arr_exception( self ):

        # Test data
        d = verdict.Dict( {
            1 : {
                'a': np.array([ 1., 2. ]),
                'b': np.array([ 3., 4. ]),
            },
        } )
        d['c'] = {
            'a': np.array([ 5., 6. ]),
        }
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    f[str(key)][inner_key][...],
                )

        # Make sure attributes save
        npt.assert_allclose( f.attrs['x'], attrs['x'] )

    ########################################################################

    def test_jagged_arr_to_filled_arr( self ):

        # Simple case
        arr = [
            [ 1, 2, 3 ],
            [ 1, 2, ],
            [ 1, 2, ],
        ]
        actual, _ = verdict.jagged_arr_to_filled_arr( arr, fill_value=np.nan )
        expected = np.array([
            [ 1, 2, 3 ],
            [ 1, 2, np.nan, ],
            [ 1, 2, np.nan, ],
        ])
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_jagged_arr_to_filled_arr_str( self ):

        # Simple case
        arr = [
            [ '1', '2', '3' ],
            [ '1', '2', ],
            [ '1', '2', ],
        ]
        actual, _ = verdict.jagged_arr_to_filled_arr( arr, )
        expected = np.array([
            [ '1', '2', '3' ],
            [ '1', '2', 'nan', ],
            [ '1', '2', 'nan', ],
        ])
        npt.assert_equal( expected, actual )

    ########################################################################

    def test_jagged_arr_to_filled_arr_nested( self ):

        # Nested
        arr = [
            [
                [ 1, 2, ],
                [ 1, 2, ],
                [ 1, 2, 3, ],
            ],
            [
                [ 4, 5, ],
                [ 4, 5, ],
                [ 4, 5, ],
            ],
        ]
        actual, _ = verdict.jagged_arr_to_filled_arr( arr, fill_value=np.nan )
        expected = np.array([
            [
                [ 1, 2, np.nan ],
                [ 1, 2, np.nan, ],
                [ 1, 2, 3, ],
            ],
            [
                [ 4, 5, np.nan, ],
                [ 4, 5, np.nan, ],
                [ 4, 5, np.nan, ],
            ],
        ])
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_jagged_arr_to_filled_arr_messy( self ):

        # Messy
        arr = np.array([
            [
                [
                    [ 1, 2, 3 ],
                    [ 1, 2, ],
                    [ 1, ],
                ],
                [
                    [ 4, 5, ],
                    [ 4, 5, ],
                ],
             ],
             [
                 [
                     [ 1, ],
                     [ 1, 2, ],
                     [ 1, ],
                 ],
             ],
        ])
        expected = np.array([
            [
                [
                    [ 1, 2, 3 ],
                    [ 1, 2, np.nan, ],
                    [ 1, np.nan, np.nan, ],
                ],
                [
                    [ 4, 5, np.nan, ],
                    [ 4, 5, np.nan, ],
                    [ np.nan, np.nan, np.nan, ],
                ],
            ],
            [
                [
                    [ 1, np.nan, np.nan, ],
                    [ 1, 2, np.nan, ],
                    [ 1, np.nan, np.nan, ],
                ],
                [
                    [ np.nan, np.nan, np.nan, ],
                    [ np.nan, np.nan, np.nan, ],
                    [ np.nan, np.nan, np.nan, ],
                ],
            ],
        ])

        actual, _ = verdict.jagged_arr_to_filled_arr( arr, fill_value=np.nan )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_to_hdf5_jagged_lists( self ):

        # Test data
        d = verdict.Dict( {
            1 : {
                'a': [
                    np.array([ 1, 2, ]),
                    np.array([ 1, 2, 3, 4 ]),
                ],
                'b': [
                    np.array([ 'a', 'b', ]),
                    np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                ],
                'c': [
                    [
                        np.array([ 'a', 'b', ]),
                        np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                        np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                    ],
                    [
                        np.array([ '1', '2', ]),
                        np.array([ '1', '2', '3', '4' ]),
                    ],
                ],
            },
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                if inner_key != 'c':
                    for i, v in enumerate( inner_item ):
                        for j, v_j in enumerate( v ):
                            try:
                                assert v_j == f[str(key)][inner_key][...][i][j]
                            except AssertionError:
                                assert v_j == f[str(key)][inner_key][...][i][j].decode( 'UTF-8' )
                else:
                    for i, v in enumerate( inner_item ):
                        for j, v_j in enumerate( v ):
                            for k, v_k in enumerate( v_j ):
                                try:
                                    assert v_k == f[str(key)][inner_key][...][i,j,k]
                                except AssertionError:
                                    assert v_k == f[str(key)][inner_key][...][i,j,k].decode( 'UTF-8' )

    ########################################################################

    def test_to_hdf5_jagged_lists_alternate_method( self ):
        '''This method saves the innermost element of each row of the array as
        a separate dataset.
        '''

        # Test data
        d = verdict.Dict( {
            1 : {
                'a': [
                    np.array([ 1, 2, ]),
                    np.array([ 1, 2, 3, 4 ]),
                ],
                'b': [
                    np.array([ 'a', 'b', ]),
                    np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                ],
                'c': [
                    [
                        np.array([ 'a', 'b', ]),
                        np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                        np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                    ],
                    [
                        np.array([ 1, 2, ]),
                        np.array([ 1, 2, 3, 4 ]),
                    ],
                ],
            },
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5(
            self.savefile,
            attributes = attrs,
            handle_jagged_arrs = 'row datasets',
            **self.kwargs
        )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                if inner_key != 'c':
                    for i, v in enumerate( inner_item ):
                        ukey = 'jagged{}'.format( i )
                        for j, v_j in enumerate( v ):
                            try:
                                assert v_j == f[str(key)][inner_key][ukey][...][j]
                            except AssertionError:
                                assert v_j == f[str(key)][inner_key][ukey][...][j].decode( 'UTF-8' )
                else:
                    for i, v in enumerate( inner_item ):
                        ukey = 'jagged{}'.format( i )
                        for j, v_j in enumerate( v ):
                            ukey_j = 'jagged{}'.format( j )
                            for k, v_k in enumerate( v_j ):
                                try:
                                    assert v_k == f[str(key)][inner_key][ukey][ukey_j][...][k]
                                except AssertionError:
                                    assert v_k == f[str(key)][inner_key][ukey][ukey_j][...][k].decode( 'UTF-8' )

    ########################################################################

    # This is a failing test. Do I *want* it to pass? Why did I create it?
    def dont_test_to_hdf5_string_and_tuple_array( self ):

        # Test data
        d = verdict.Dict( {
            1 : {
                'a': np.array([ ( 'aa', 'bb' ), ( 'cc', 'dd' ) ]),
                'b': np.array([ ( 'a', 'b' ), ( 'c', 'd' ) ]),
                'c': 'abcdefg',
                'd': [
                    np.array([ ( 'a', 'b' ), ]),
                    np.array([ ( 'aa', 'bb' ), ( 'cc', 'dd' ) ]),
                ],
            },
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                if inner_key in [ 'a', 'b' ]:
                    for i, arr in enumerate( inner_item ):
                        for j, v in enumerate( arr ):
                            assert v == f[str(key)][inner_key][...][i][j]
                elif inner_key in [ 'd', ]:
                    for i, arr in enumerate( inner_item ):
                        for j, line in enumerate( arr ):
                            for k, v in enumerate( line ):
                                assert v == f[str(key)][inner_key][...][i][j][k]
            else:
                assert inner_item == f[str(key)][inner_key][...]

        # Make sure attributes save
        npt.assert_allclose( f.attrs['x'], attrs['x'] )

    ########################################################################

    def test_from_hdf5( self ):

        # Create test data
        expected = verdict.Dict( {
            1 : verdict.Dict( {
                'a': np.array([ 1., 2. ]),
                'b': np.array([ 3., 4. ]),
                'c': np.array([ 'a', 'b' ]),
                'd': [],
            } ),
            2 : verdict.Dict( {
                'a': np.array([ 5., 6. ]),
                'b': np.array([ 7., 8. ]),
                'c': np.array([ [ 'a', 'b' ], [ 'c', 'd' ] ]),
            } ),
        } )
        attrs = { 'x': 1.5 }
        expected.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Try to load
        actual, actual_attrs = verdict.Dict.from_hdf5(
            self.savefile,
            load_attributes = True,
            **self.kwargs
        )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in actual.items():
            for inner_key, inner_item in item.items():
                try:
                    npt.assert_allclose(
                        inner_item,
                        f[str(key)][inner_key][...],
                    )
                except TypeError:
                    assert len( np.setdiff1d(
                        inner_item,
                        f[key][inner_key][...].astype( str ),
                    ) ) == 0

        # Compare attributes
        self.assertEqual( attrs, actual_attrs )

    ########################################################################

    def test_from_hdf5_condensed( self ):

        # Create test data
        expected = verdict.Dict( {
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
        } )
        expected.to_hdf5( self.savefile, condensed=True, **self.kwargs )

        # Try to load
        actual = verdict.Dict.from_hdf5( self.savefile, unpack=True, **self.kwargs )

        # Compare
        for key, item in expected.items():
            for inner_key, inner_item in item.items():
                for ii_key, ii_item in inner_item.items():
                    npt.assert_allclose(
                        ii_item,
                        actual[key][str(inner_key)][ii_key],
                    )

    ########################################################################

    def test_from_hdf5_condensed_shallow( self ):

        # Create test data
        expected = verdict.Dict( {
            1 : verdict.Dict( {
                'a': 1.,
                'b': 3.,
            } ),
            2 : verdict.Dict( {
                'a': 5.,
                'b': 7.,
            } ),
        } )
        expected.to_hdf5( self.savefile, condensed=True, **self.kwargs )

        # Try to load
        actual = verdict.Dict.from_hdf5( self.savefile, unpack=True, **self.kwargs )

        # Compare
        for key, item in expected.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    actual[str(key)][inner_key],
                )

    ########################################################################

    def test_from_hdf5_one_entry_dict( self ):

        # Test data
        base_arr = np.array([
            [
                [ 1., 2., ],
                [ 3., 4., ],
            ],
            [
                [ 5., 6., ],
                [ 7., 8., ],
            ],
        ])
        d = verdict.Dict( {
            'A' : {
                '1' : {
                    'a': base_arr,
                    'b': 2. * base_arr,
                },
                '2' : {
                    'c': 3. * base_arr,
                    'd': 4. * base_arr,
                },
            }
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Try to load
        actual, attrs = verdict.Dict.from_hdf5( self.savefile, **self.kwargs )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in d['A'].items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    actual['A'][str(key)][inner_key][...],
                )

        # Make sure attributes save
        npt.assert_allclose( f.attrs['x'], attrs['x'] )

    ########################################################################

    def test_from_hdf5_create_nonexistent( self ):

        d = verdict.Dict.from_hdf5( self.savefile, create_nonexistent=True )

        assert len( d ) == 0

    ########################################################################

    def test_from_hdf5_jagged_arr( self ):

        # Test data
        d = verdict.Dict( {
            1 : {
                'a': [
                    np.array([ 1, 2, ]),
                    np.array([ 1, 2, 3, 4 ]),
                ],
                'b': [
                    np.array([ 'a', 'b', ]),
                    np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                ],
                'c': [
                    [
                        np.array([ 3, 3, ]),
                        np.array([ 3, 3, 3,  3 ]),
                        np.array([ 3, 3, 3,  3 ]),
                    ],
                    [
                        np.array([ 1, 2, ]),
                        np.array([ 1, 2, 3, 4 ]),
                    ],
                ],
            },
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5( self.savefile, attributes=attrs, **self.kwargs )

        # Try to load
        actual, attrs = verdict.Dict.from_hdf5( self.savefile, **self.kwargs )

        # Compare
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                if inner_key != 'c':
                    for i, v in enumerate( inner_item ):
                        for j, v_j in enumerate( v ):
                            assert v_j == actual[str(key)][inner_key][i][j]
                else:
                    for i, v in enumerate( inner_item ):
                        for j, v_j in enumerate( v ):
                            npt.assert_allclose(
                                v_j,
                                actual[str(key)]['c'][i][j]
                            )

    ########################################################################

    def test_from_hdf5_jagged_arr_row_datasets( self ):

        # Test data
        d = verdict.Dict( {
            1 : {
                'a': [
                    np.array([ 1, 2, ]),
                    np.array([ 1, 2, 3, 4 ]),
                ],
                'b': [
                    np.array([ 'a', 'b', ]),
                    np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                ],
                'c': [
                    [
                        np.array([ 'a', 'b', ]),
                        np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                        np.array([ 'aa', 'bb', 'cc', 'dd' ]),
                    ],
                    [
                        np.array([ 1, 2, ]),
                        np.array([ 1, 2, 3, 4 ]),
                    ],
                ],
            },
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_hdf5(
            self.savefile,
            attributes = attrs,
            handle_jagged_arrs = 'row datasets',
            **self.kwargs
        )

        # Try to load
        actual, attrs = verdict.Dict.from_hdf5( self.savefile, **self.kwargs )

        # Compare
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                if inner_key != 'c':
                    for i, v in enumerate( inner_item ):
                        for j, v_j in enumerate( v ):
                            assert v_j == actual[str(key)][inner_key][i][j]
                else:
                    for i, v in enumerate( inner_item ):
                        for j, v_j in enumerate( v ):
                            for k, v_k in enumerate( v_j ):
                                assert v_k == actual[str(key)][inner_key][i][j][k]

########################################################################
########################################################################

class TestVerDictSparseHDF5( TestVerDictHDF5 ):

    def setUp( self ):

        self.savefile = 'to_hdf5_test.hdf5'
        self.kwargs = { 'sparse': True }

    def tearDown( self ):

        # Delete spurious files
        if os.path.isfile( self.savefile ):
            os.remove( self.savefile )

    ########################################################################

    def test_to_hdf5_sparse_dataset( self ):

        # Test dataset
        sparse_matrix = np.zeros( (15000, 200 ) )
        sparse_matrix[0,0] = 5
        sparse_matrix[1,12] = 5

        # Usual
        d = verdict.Dict( {
            'a': sparse_matrix,
        } )
        d.to_hdf5( self.savefile )
        size = os.path.getsize( self.savefile )

        # Sparse
        d['a'] = ss.csr_matrix( sparse_matrix )
        d.to_hdf5( self.savefile, sparse=True )
        sparse_size = os.path.getsize( self.savefile )

        assert sparse_size < size

    ########################################################################

    def test_from_hdf5_sparse_dataset( self ):

        # Test dataset
        sparse_matrix = np.zeros( (15000, 200 ) )
        sparse_matrix[0,0] = 5
        sparse_matrix[1,12] = 5

        # Save sparse data
        d = verdict.Dict( {
            'a': ss.csr_matrix( sparse_matrix ),
            'b': np.array([ 1., 2., 3. ]),
        } )
        d.to_hdf5( self.savefile, **self.kwargs )

        # Load sparse data
        d2 = verdict.Dict.from_hdf5( self.savefile, **self.kwargs )
        npt.assert_allclose( d['a'].toarray(), d2['a'].toarray() )
        npt.assert_allclose( d['b'], d2['b'] )

    
########################################################################
########################################################################

class TestVerDictJSON( unittest.TestCase ):

    def setUp( self ):

        self.savefile = 'to_json_test.json'
        self.kwargs = {}

    def tearDown( self ):

        # Delete spurious files
        if os.path.isfile( self.savefile ):
            os.remove( self.savefile )

    ########################################################################

    def test_to_json( self ):

        # Test data
        d = verdict.Dict( {
            1 : verdict.Dict( {
                'a': np.array([ 1., 2. ]),
                'b': np.array([ 3., 4. ]),
            } ),
            2 : verdict.Dict( {
                'a': np.array([ 5., 6. ]),
                'b': np.array([ 7., 8. ]),
            } ),
        } )
        attrs = { 'x' : 1.5, }

        # Try to save
        d.to_json( self.savefile, **self.kwargs )

        # Compare
        with open( self.savefile ) as f:
            actual = json.load(f)
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    actual[str(key)][inner_key],
                )

        # Ensure that the dictionary behavior is preserved
        d[1] / 2.

    ########################################################################

    def test_to_json_additional_nesting( self ):

        # Test data
        d = verdict.Dict( {
            'i' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': np.array([ 1., 2. ]),
                    'b': np.array([ 3., 4. ]),
                } ),
                2 : verdict.Dict( {
                    'a': np.array([ 5., 6. ]),
                    'b': np.array([ 7., 8. ]),
                } ),
            } ),
            'ii' : verdict.Dict( {
                1 : verdict.Dict( {
                    'a': np.array([ 10., 20. ]),
                    'b': np.array([ 30., 40. ]),
                } ),
                2 : verdict.Dict( {
                    'a': np.array([ 50., 60. ]),
                    'b': np.array([ 70., 80. ]),
                } ),
            } ),
        } )

        # Try to save
        d.to_json( self.savefile, **self.kwargs )

        # Compare
        with open( self.savefile ) as f:
            actual = json.load(f)
        for key, item in d.items():
            for inner_key, inner_item in item.items():
                for ii_key, ii_item in inner_item.items():
                    npt.assert_allclose(
                        ii_item,
                        actual[key][str(inner_key)][ii_key],
                    )

    ########################################################################

    def test_from_json( self ):

        # Create test data
        expected = verdict.Dict( {
            1 : verdict.Dict( {
                'a': np.array([ 1., 2. ]),
                'b': np.array([ 3., 4. ]),
            } ),
            2 : verdict.Dict( {
                'a': np.array([ 5., 6. ]),
                'b': np.array([ 7., 8. ]),
            } ),
        } )
        attrs = { 'x': 1.5 }
        expected.to_json( self.savefile, **self.kwargs )

        # Try to load
        actual = verdict.Dict.from_json(
            self.savefile,
            **self.kwargs
        )

        # Compare
        with open( self.savefile ) as f:
            actual = json.load(f)
        for key, item in actual.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    actual[str(key)][inner_key],
                )

    ########################################################################

    def test_from_json_create_nonexisting( self ):

        d = verdict.Dict.from_json( self.savefile, create_nonexisting=True )

        assert len( d ) == 0

########################################################################
########################################################################

class TestVerDictSaveLoad( unittest.TestCase ):

    def setUp( self ):

        self.savefiles = [ 'test.json', 'test.hdf5', 'test.h5' ]
        self.kwargs = {}

    def tearDown( self ):

        # Delete spurious files
        for savefile in self.savefiles:
            if os.path.isfile( savefile ):
                os.remove( savefile )

    ########################################################################

    def test_saveload( self ):

        # Test data
        d = verdict.Dict( {
            1 : verdict.Dict( {
                'a': np.array([ 1., 2. ]),
                'b': np.array([ 3., 4. ]),
            } ),
            2 : verdict.Dict( {
                'a': np.array([ 5., 6. ]),
                'b': np.array([ 7., 8. ]),
            } ),
        } )
        attrs = { 'x' : 1.5, }

        def check_values( expected, actual ):
            for key, item in expected.items():
                for inner_key, inner_item in item.items():
                    npt.assert_allclose(
                        inner_item,
                        actual[str(key)][inner_key],
                    )

        for i, savefile in enumerate( self.savefiles ):

            # Check saving
            d.save( savefile )

            # JSON comparison
            if i == 0:
                # Compare
                with open( savefile ) as f:
                    actual = json.load(f)
                check_values( d, actual )

            # HDF5 comparison
            else:
                f = h5py.File( savefile, 'r' )
                check_values( d, f )

            # Check loading
            d2 = verdict.Dict.load( savefile )
            check_values( d, d2 )

########################################################################
########################################################################

class TestDictFromDefaultsAndVariations( unittest.TestCase ):

    def test_default( self ):

        defaults = { 'best cat' : 'Melvulu', }
        variations = {
            'person a' : { 'other best cat' : 'Chellbrat', },
            'person b' : {
                'best cat' : 'A Normal Melville Cat',
                'other best cat' : 'Chellcat',
            },
        }

        actual = verdict.dict_from_defaults_and_variations( defaults, variations )
        expected = {
            'person a' : {
                'best cat' : 'Melvulu',
                'other best cat' : 'Chellbrat',
            },
            'person b' : {
                'best cat' : 'A Normal Melville Cat',
                'other best cat' : 'Chellcat',
            },
        }

        for key in expected.keys():
            assert expected[key] == actual[key]

########################################################################
########################################################################

class TestExternalCompatibility( unittest.TestCase ):

    def test_deepcopy( self ):

        # Setup
        d = { 'a' : 1, 'b' : 2 }
        ex = verdict.Dict( d )

        ac = copy.deepcopy( ex )

        assert ex == ac
