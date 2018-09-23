'''Testing for verdict.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import shutil
import subprocess
import unittest

import verdict

########################################################################
########################################################################

class TestVerDictStartup( unittest.TestCase ):

    def test_default( self ):

        d = { 'a' : 1, 'b' : 2 }

        smart_dict = verdict.Dict( d )

        self.assertEqual( smart_dict['b'], 2 )
        self.assertEqual( len( smart_dict ), 2 )

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

    def test_multiply_smart_dict( self ):

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
        expected = { 1 : 1, 2 : 2, }
        actual = d/2
        self.assertEqual( expected, actual )

        d = verdict.Dict( { 1 : 2, 2 : 4 } )
        expected = { 1 : 2, 2 : 1, }
        actual = 4/d
        self.assertEqual( expected, actual )

    ########################################################################

    def test_divide_smart_dict( self ):

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

    def test_add_smart_dict( self ):

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

    def test_subtract_smart_dict( self ):

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
            'a' : 1,
            'c' : 2,
            'b' : 3,
        } )

        expected = np.array([ 1., 3., 2. ])

        actual = d.array()

        npt.assert_allclose( expected, actual )

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

