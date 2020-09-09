#!/usr/bin/env python
'''Version of Python dictionary with extended functionality.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import numpy as np
import os
import pandas as pd
import six

try:
    import collections.abc as collections
except ImportError:
    import collections

########################################################################
########################################################################

class Dict( collections.Mapping ):
    '''Replacement for dictionary that allows easier access to the attributes
    and methods of the dictionary components. For example, if one has a smart
    dictionary of TestClassA objects, each of which has a TestClassB attribute,
    which in turn have a foo method, then smart_dict.test_class_b.foo(2) would
    be a dict with foo calculated for each. In other words, it would be
    equivalent to the following code:

    results = {}
    for key in smart_dict.keys():
        results[key] = smart_dict[key].test_class_b.foo( 2 )
    return results
    '''

    def __init__( self, *args, **kwargs ):

        self._storage = dict( *args, **kwargs )
        self.unpack_name = 'name'

        # Convert contained dicts to Dicts
        for key, item in self.items():
            if isinstance( item, dict ):
               self._storage[key] = Dict( item )

    def __iter__( self ):
        return iter( self._storage )

    def __len__( self ):
        return len( self._storage )

    def __getitem__( self, item ):
        try:
            return self._storage[item]
        # Allow accessing a given depth using '/'
        except KeyError:

            keys = item.split( '/' )
            result = self._storage[keys[0]]
            for key in keys[1:]:
                result = result[key]

            return result

    def __setitem__( self, key, item ):
        self._storage[key] = item

    def __delitem__( self, key ):
        del self._storage[key]

    def __repr__( self ):

        out_str = "Dict, {\n"

        for key in self.keys():

            def get_print_obj( obj ):
                try:
                    print_obj = obj.__repr__()
                except:
                    print_obj = obj
                return print_obj

            out_str += "{} : {},\n".format(
                get_print_obj( key ),
                get_print_obj( self._storage[key] ),
            )

        out_str += "}\n"

        return out_str

    def __getattr__( self, attr ):

        results = {}
        for key in self.keys():

            results[key] = getattr( self._storage[key], attr )

        return Dict( results )

    def __getstate__( self ):

        return self.__dict__

    def __setstate__(self, state ):

        self.__dict__ = state

    ########################################################################
    # Operation Methods
    ########################################################################

    def __mul__( self, other ):

        results = {}

        for key in self.keys():
            results[key] = self._storage[key]*other

        return Dict( results )


