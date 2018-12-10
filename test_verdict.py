'''Testing for verdict.py
'''

from mock import patch
import h5py
import numpy as np
import numpy.testing as npt
import os
import pandas as pd
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
        d.to_hdf5( self.savefile, attributes=attrs )

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
        d.to_hdf5( self.savefile )

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
        d.to_hdf5( self.savefile, condensed=True )

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
                        f[key][inner_key][...],
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
        d.to_hdf5( self.savefile, condensed=True )

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
                    f[key][...],
                ) ) == 0

    ########################################################################

    def test_from_hdf5( self ):

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
        expected.to_hdf5( self.savefile, attributes=attrs )

        # Try to load
        actual, actual_attrs = verdict.Dict.from_hdf5(
            self.savefile,
            load_attributes = True,
        )

        # Compare
        f = h5py.File( self.savefile, 'r' )
        for key, item in actual.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    f[str(key)][inner_key][...],
                )

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
        expected.to_hdf5( self.savefile, condensed=True )

        # Try to load
        actual = verdict.Dict.from_hdf5( self.savefile, unpack=True )

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
        expected.to_hdf5( self.savefile, condensed=True )

        # Try to load
        actual = verdict.Dict.from_hdf5( self.savefile, unpack=True )

        # Compare
        for key, item in expected.items():
            for inner_key, inner_item in item.items():
                npt.assert_allclose(
                    inner_item,
                    actual[str(key)][inner_key],
                )

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

