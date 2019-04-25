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

        # Convert contained dicts to Dicts
        for key, item in self.items():
            if isinstance( item, dict ):
               self._storage[key] = Dict( item )

    def __iter__( self ):
        return iter( self._storage )

    def __len__( self ):
        return len( self._storage )

    def depth( self, level=1 ):
        '''Depth of the Dict.

        Args:
            level (int):
                A level of N means this Dict is contained in N-1 other Dict.
        '''

        depths = []
        for item in self.values():
            if hasattr( item, 'depth' ):
                depths.append( item.depth( level+1 ) )
            else:
                return level

        return max( depths )

    def __getitem__( self, item ):
        return self._storage[item]

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

    def __call__( self, *args, **kwargs ):

        results = {}
        for key in self.keys():

            results[key] = self._storage[key]( *args, **kwargs )

        return Dict( results )

    def call_custom_kwargs( self, kwargs, default_kwargs={}, verbose=False ):
        '''Perform call, but using custom keyword arguments per dictionary tag.

        Args:
            kwargs (dict) : Custom keyword arguments to pass.
            default_kwargs (dict) : Defaults shared between keyword arguments.

        Returns:
            results (dict) : Dictionary of results.
        '''

        used_kwargs = dict_from_defaults_and_variations( default_kwargs, kwargs )

        results = {}
        for key in self.keys():

            if verbose:
                print( "Calling for {}".format( key ) )

            results[key] = self._storage[key]( **used_kwargs[key] )

        return Dict( results )

    ########################################################################

    def call_iteratively( self, args_list ):

        results = {}
        for key in self.keys():

            inner_results = []
            for args in args_list:

                inner_result = self._storage[key]( args )
                inner_results.append( inner_result )

            results[key] = inner_results

        return results

    ########################################################################
    # For handling when the Smart Dict contains dictionaries
    ########################################################################

    def inner_item( self, item ):
        '''When Dict is a dictionary of dicts themselves,
        this can be used to get an item from those dictionaries.
        '''

        results = {}

        for key in self.keys():

            results[key] = self._storage[key][item]

        return Dict( results )

    def inner_keys( self ):

        results = {}

        for key in self.keys():

            results[key] = self._storage[key].keys()

        return Dict( results )

    def transpose( self ):

        results = {}

        # Populate results dictionary
        for key, item in self.items():
            for inner_key in item.keys():

                try:
                    results[inner_key][key] = item[inner_key]
                except KeyError:
                    results[inner_key] = {}
                    results[inner_key][key] = item[inner_key]

        return Dict( results )

    ########################################################################
    # Operation Methods
    ########################################################################

    def __add__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = self._storage[key] + other[key]

        else:
            for key in self.keys():
                results[key] = self._storage[key] + other

        return Dict( results )

    __radd__ = __add__

    def __sub__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = self._storage[key] - other[key]

        else:
            for key in self.keys():
                results[key] = self._storage[key] - other

        return Dict( results )

    def __rsub__( self, other ):
        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = other[key] - self._storage[key]

        else:
            for key in self.keys():
                results[key] = other - self._storage[key]

        return Dict( results )

    def __mul__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = self._storage[key]*other[key]

        else:
            for key in self.keys():
                results[key] = self._storage[key]*other

        return Dict( results )

    __rmul__ = __mul__

    def __div__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = self._storage[key]/other[key]
        else:
            for key in self.keys():
                results[key] = self._storage[key]/other

        return Dict( results )

    def __truediv__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = self._storage[key]/other[key]
        else:
            for key in self.keys():
                results[key] = self._storage[key]/other

        return Dict( results )

    def __floordiv__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = self._storage[key]//other[key]
        else:
            for key in self.keys():
                results[key] = self._storage[key]//other

        return Dict( results )

    def __rdiv__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = other[key]/self._storage[key]
        else:
            for key in self.keys():
                results[key] = other/self._storage[key]

        return Dict( results )

    def __rtruediv__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = other[key]/self._storage[key]
        else:
            for key in self.keys():
                results[key] = other/self._storage[key]

        return Dict( results )

    def __rfloordiv__( self, other ):

        results = {}

        if isinstance( other, Dict ):
            for key in self.keys():
                results[key] = other[key]//self._storage[key]
        else:
            for key in self.keys():
                results[key] = other//self._storage[key]

        return Dict( results )

    ########################################################################
    # Other operations
    ########################################################################

    def sum_contents( self ):
        '''Get the sum of all the contents inside the Dict.'''

        for i, key in enumerate( self.keys() ):

            # To handle non-standard items
            # (e.g. things that aren't ints or floats )
            if i == 0:
                result = copy.deepcopy( self._storage[key] )

            else:
                result += self._storage[key]

        return result

    def median( self ):

        return np.median( self.array() )

    def nanmedian( self ):

        return np.nanmedian( self.array() )

    def percentile( self, q ):

        return np.percentile( self.array(), q )

    def nanpercentile( self, q ):

        return np.nanpercentile( self.array(), q )

    def keymax( self ):
        
        for key, item in self.items():
            try:
                if item_max < item:
                    item_max = item
                    key_max = key
            except NameError:
                item_max = item
                key_max = key

        return key_max, item_max

    def keymin( self ):
        
        for key, item in self.items():
            try:
                if item_min > item:
                    item_min = item
                    key_min = key
            except NameError:
                item_min = item
                key_min = key

        return key_min, item_min

    def apply( self, fn, *args, **kwargs ):
        '''Apply some function to each item in the smart dictionary, and
        return the results as a Dict.
        '''

        results = {}

        for key, item in self.items():
            results[key] = fn( item, *args, **kwargs )

        return Dict( results )

    def split_by_key_slice( self, sl, str_to_match=None ):
        '''Break the smart dictionary into smaller smart dictionaries according
        to a subset of the key.

        Args:
            sl (slice) :
                Part of the key to use to make subsets.

            str_to_match (str) :
                If True, split into broad categories that match this string
                in the given slice or don't.
        '''

        results = {}

        if str_to_match is not None:
            results[True] = Dict( {} )
            results[False] = Dict( {} )

        for key, item in self.items():

            key_slice = key[sl]

            if str_to_match is not None:
                str_matches = key_slice == str_to_match
                results[str_matches][key] = item
            
            else:
                try:
                    results[key_slice][key] = item
                except KeyError:
                    results[key_slice] = Dict( {} )
                    results[key_slice][key] = item

        return results

    def split_by_dict( self, d, return_list=False ):
        '''Break the smart dictionary into smaller smart dictionaries according
        to their label provided by a dictionary

        Args:
            d (dict):
                Dictionary to use to split into smaller dictionaries

            return_list (bool) :
                If True, return a list of arrays.
        '''

        results = {}
    
        for key, item in self.items():

            try:
                result_subkey = d[key]
                try:
                    results[result_subkey][key] = item
                except KeyError:
                    results[result_subkey] = Dict( {} )
                    results[result_subkey][key] = item
            except KeyError:
                try:
                    results['no label'][key] = item
                except KeyError:
                    results['no label'] = Dict( {} )
                    results['no label'][key] = item
        
        if not return_list:
            return results

        final_results = []
        for key, item in results.items():
            final_results.append( item.array() )

        return final_results

    def log10( self ):
        '''Wrapper for np.log10'''

        return self.apply( np.log10 )

    def remove_empty_items( self ):
        '''Look for empty items and delete them.'''

        keys_to_delete = []
        for key, item in self.items():
            if len( item ) == 0:
                keys_to_delete.append( key )

        for key in keys_to_delete:
            del self._storage[key]

    ########################################################################
    # Methods for converting to and from other formats.
    ########################################################################

    def array( self ):
        '''Returns a np.ndarray of values with unique order (sorted keys )'''

        values = [ x for _,x in sorted(zip( self.keys(), self.values() ) )]

        return np.array( values )

    ########################################################################

    def keys_array( self ):
        '''Returns a np.ndarray of keys with unique order (sorted keys )'''

        return np.array( sorted( self.keys() ) )

    ########################################################################

    def to_df( self ):
        '''Join the innermost Dict classes into a pandas
        DataFrame, where the keys in the innermost dictionaries are the index
        and the keys for the innermost dictionaries are the column headers.
        '''

        right_depth = self.depth() == 2

        if right_depth:
            dfs = []
            for key, item in self.items():

                # Make the dataframe for each dictionary
                data = {
                    'name': list( item.keys() ),
                    key: list( item.values() ),
                }
                df = pd.DataFrame( data )
                df.set_index( 'name', inplace=True )

                dfs.append( df )

            return pd.concat( dfs, axis=1 )

        else:
            result = {}
            for key, item in self.items():
                result[key] = item.to_df()

            return result

    ########################################################################

    def to_hdf5(
        self,
        filepath,
        attributes = None,
        overwrite_existing_file = True,
        condensed = False
    ):
        '''Save the contents as a HDF5 file.

        Args:
            filepath (str):
                Location to save the hdf5 file at.

            attributes (dict):
                Dictionary of attributes to store as attributes for the HDF5
                file.

            overwrite_existing_file (boolean):
                If True and a file already exists at filepath, delete it prior
                to saving.

            condensed (boolean):
                If True, combine the innermost dictionaries into a condensed
                DataFrame/array-like format.
        '''

        if overwrite_existing_file:
            if os.path.isfile( filepath ):
                os.remove( filepath )

        f = h5py.File( filepath, 'w-' )

        # Store attributes
        if attributes is not None:
            for key, item in attributes.items():
                f.attrs[key] = item

        def recursive_save( current_path, key, item ):
            '''Function for recursively saving to an hdf5 file.

            Args:
                current_path (str):
                    Current location in the hdf5 file.

                key (str):
                    Key to save.

                item:
                    Item to save.
            '''

            # Update path
            current_path = '{}/{}'.format( current_path, key ) 

            if isinstance( item, Dict ):

                # Make space for the data set
                f.create_group( current_path )

                # Save in condensed format
                if condensed and item.depth() == 2:
                    df = item.to_df()

                    recursive_save(
                        current_path,
                        df.index.name,
                        df.index.values.astype( str )
                    )
                    for c_name in df.columns:
                        recursive_save(
                            current_path,
                            c_name,
                            df[c_name].values,
                        )
                                        
                else:
                    # Recurse
                    for inner_key, inner_item in item.items():
                        recursive_save( current_path, inner_key, inner_item )

            # Save data if the inner item isn't a Dict
            else:
                try:
                    f.create_dataset( current_path, data=item )
                # Accounts for h5py not recognizing unicode. This is fixed
                # in h5py 2.9.0, with PR #1032.
                # The fix used here is exactly what the PR does.
                except TypeError:
                    data = np.array(
                        item,
                        dtype=h5py.special_dtype( vlen=six.text_type ),
                    )
                    f.create_dataset( current_path, data=data )

        # Shallow dictionary condensed edge case
        shallow_condensed_save = ( self.depth() <= 2 ) and condensed

        # Actual save
        if not shallow_condensed_save:
            for key, item in self.items():
                recursive_save( '', key, item )

        # For relatively shallow dictionaries
        else:
            df = self.to_df()

            recursive_save(
                '',
                df.index.name,
                df.index.values.astype( str )
            )
            for c_name in df.columns:
                recursive_save(
                    '',
                    c_name,
                    df[c_name].values,
                )


        f.close()

    ########################################################################

    @classmethod
    def from_hdf5( cls, filepath, load_attributes=True, unpack=False ):
        '''Load a HDF5 file as a verdict Dict.

        Args:
            filepath (str):
                Location to load the hdf5 file from.

            load_attributes (boolean):
                If True, load attributes stored in the hdf5 file's .attrs keys
                and return as a separate dictionary.

            unpack (boolean):
                If True and the inner-most groups are combined into a condensed
                DataFrame/array-like format, unpack them into a traditional
                structure.
        '''

        f = h5py.File( filepath, 'r' )

        def recursive_retrieve( current_path, key ):
            '''Function for recursively loading from an hdf5 file.

            Args:
                current_path (str):
                    Current location in the hdf5 file.

                key (str):
                    Key to load.
            '''

            # Update path
            current_path = '{}/{}'.format( current_path, key ) 

            item = f[current_path]
        
            if isinstance( item, h5py.Group ):

                group = f[current_path]
                result = {}

                # Sometimes the data is saved in a condensed, DataFrame-like,
                # format. But when we load it we may want it back in the
                # usual format.
                if unpack and 'name' in group.keys():
                    for i_key in group.keys():

                        if i_key == 'name':
                            continue
                    
                        i_result = {}
                        ii_items = zip( group['name'][...], group[i_key][...] )
                        for ii_key, ii_item in ii_items:
                            i_result[ii_key] = ii_item

                        result[i_key] = Dict( i_result )

                    return Dict( result )

                else:
                    for i_key in group.keys():
                        result[i_key] = recursive_retrieve( current_path, i_key )

                    return Dict( result )

            elif isinstance( item, h5py.Dataset ):
                return np.array( item[...] )

        result = {}
        for key in f.keys():
            result[key] = recursive_retrieve( '', key )

        result = Dict( result )

        # For shallow save files
        if unpack and 'name' in result.keys():
            true_result = {}
            
            for i_key in result.keys():

                if i_key == 'name':
                    continue
            
                i_result = {}
                ii_items = zip( result['name'], result[i_key] )
                for ii_key, ii_item in ii_items:
                    i_result[ii_key] = ii_item

                true_result[i_key] = Dict( i_result )

            result = true_result

        # Load (or don't) attributes and return
        if load_attributes and len( f.attrs.keys() ) > 0:
            attrs = {}
            for key in f.attrs.keys():
                attrs[key] = f.attrs[key]
            return result, attrs
        else:
            return result

    ########################################################################

    @classmethod
    def from_class_and_args( cls, contained_cls, args, default_args={}, ):
        '''Alternate constructor. Creates a Dict of contained_cls objects,
        with arguments passed to it from the dictionary created by
        defaults and variations.

        Args:
            contained_cls (type of object/constructor):
                What class should the smart dict consist of?

            args (dict/other):
                Arguments that should be passed to contained_cls. If not a
                dict, assumed to be the first and only argument for
                the constructor.

            default_args:
                Default arguments to fill in args

        Returns:
            Dict:
                The constructed instance.
        '''

        kwargs = dict_from_defaults_and_variations( default_args, args )

        storage = {}
        for key in kwargs.keys():
            if isinstance( kwargs[key], dict ):
                storage[key] = contained_cls( **kwargs[key] )
            else:
                storage[key] = contained_cls( kwargs[key] )

        return cls( storage )

########################################################################

def merge_two_dicts( dict_a, dict_b ):
    '''Merges two dictionaries into a shallow copy.

    Args:
        dict_a (dict): First dictionary to merge.
        dict_b (dict): Second dictionary to merge.

    Returns:
        dict:
            Dictionary including elements from both.
            dict_a's entries take priority over dict_b.
    '''

    merged_dict = dict_b.copy()
    merged_dict.update( dict_a )

    return merged_dict

########################################################################

def dict_from_defaults_and_variations( defaults, variations ):
    '''Create a dictionary of dictionaries from a default dictionary
        and variations on it.

    Args:
        defaults (dict):
            Default dictionary. What each individual dictionary should
            default to.

        variations (dict of dicts):
            Each dictionary contains what should be different. The key for
            each dictionary should be a label for it.

    Returns:
        dict of dicts:
            The results is basically variations, where each child dict
            is merged with defaults.
    '''

    if len( defaults ) == 0:
        return variations

    result = {}
    for key in variations.keys():

        defaults_copy = defaults.copy()

        defaults_copy.update( variations[key] )

        result[key] = defaults_copy

    return result

