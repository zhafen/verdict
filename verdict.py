#!/usr/bin/env python
'''Version of Python dictionary with extended functionality.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import json
import h5py
import h5sparse
import numpy as np
import os
import pandas as pd
import scipy.sparse
import six
import tqdm

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

    def keys( self ):
        return self._storage.keys()

    def values( self ):
        return self._storage.values()

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

    def setitem( self, key, item, *more_keys ):
        '''Set a value, allowing for nested setting.'''

        if len( more_keys ) == 0:
            self[key] = item
        else:
            # Try to save
            try:
                self[key]
            except KeyError:
                self[key] = Dict({})

            self[key].setitem( more_keys[0], item, *more_keys[1:] )

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

    def __getstate__( self ):

        return self.__dict__

    def __setstate__(self, state ):

        self.__dict__ = state

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
                    self.unpack_name: list( item.keys() ),
                    key: list( item.values() ),
                }
                df = pd.DataFrame( data )
                df.set_index( self.unpack_name, inplace=True )

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
        condensed = False,
        handle_jagged_arrs = 'filled array',
        jagged_flag = 'jagged',
        sparse = False,
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

            handle_jagged_arrs (str):
                How to handle jagged arrays. Options:
                    filled array:
                        Create a uniform filled array capable of holding
                        any jagged arrays.
                    row datasets:
                        Save each row of a jagged array as a separate dataset.
                    assume_none:
                        Don't handle jagged arrays. Assume there are none.

            jagged_flag (str):
                Flag to indicate that this part of the hdf5 file contains part of
                an jagged array-like.

            sparse (boolean):
                If True use h5sparse instead of h5py, allowing saving of sparse matrices
                with reduced file size.
        '''

        # If using sparse matrices
        if sparse:
            hdf5_module = h5sparse
        else:
            hdf5_module = h5py

        # Make sure all contained dictionaries are verdict Dicts
        self = Dict( self )

        if overwrite_existing_file:
            if os.path.isfile( filepath ):
                os.remove( filepath )

        # Make sure the path exists
        os.makedirs(
            os.path.dirname( os.path.abspath( filepath ) ),
            exist_ok = True
        )

        with hdf5_module.File( filepath, 'w-' ) as f:

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

                # Convert missed dictionaries into verdict versions
                if isinstance( item, dict ):
                    item = Dict( item )

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

                    # Save a jagged array
                    if handle_jagged_arrs != 'assume_none':
                        if check_if_jagged_arr( item ):

                            create_dataset_jagged_arr(
                                f,
                                current_path,
                                item,
                                method = handle_jagged_arrs,
                                jagged_flag = jagged_flag,
                                hdf5_module = hdf5_module,
                            )
                        else:
                            create_dataset_fixed( f, current_path, item, hdf5_module=hdf5_module )
                    else:
                        create_dataset_fixed( f, current_path, item, hdf5_module=hdf5_module )

            # Shallow dictionary condensed edge case
            shallow_condensed_save = ( self.depth() <= 2 ) and condensed

            # Actual save
            if not shallow_condensed_save:
                for key, item in tqdm.tqdm( self.items() ):
                    recursive_save( '', key, item )

            # For relatively shallow dictionaries
            else:
                df = self.to_df()

                recursive_save(
                    '',
                    df.index.name,
                    df.index.values.astype( str )
                )
                for c_name in tqdm.tqdm( df.columns ):
                    recursive_save(
                        '',
                        c_name,
                        df[c_name].values,
                    )

    ########################################################################

    @classmethod
    def from_hdf5(
        cls,
        filepath,
        load_attributes = True,
        unpack = False,
        unpack_name = 'name',
        look_for_saved_jagged_arrs = True,
        jagged_flag = 'jagged',
        sparse = False,
        create_nonexistent = False,
    ):
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

            unpack_name (str):
                What name is used for the array containing the columns.

            look_for_saved_jagged_arrs (boolean):
                If True, check dataset names to see if the array is a jagged array.

            jagged_flag (str):
                Flag to indicate that this part of the hdf5 file contains part of
                an jagged array-like.

            sparse (boolean):
                If True use h5sparse instead of h5py, allowing reading of sparse matrices
                with reduced file size.

            create_nonexistent (boolean):
                If True if the file does not exist create it.
        '''

        # If using sparse matrices
        if sparse:
            hdf5_module = h5sparse
        else:
            hdf5_module = h5py

        if not os.path.exists( filepath ) and create_nonexistent:
            return Dict({})

        with hdf5_module.File( filepath, 'r' ) as f:

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

                if isinstance( item, hdf5_module.Group ):

                    group = f[current_path]
                    result = {}

                    # Sometimes the data is saved in a condensed, DataFrame-like,
                    # format. But when we load it we may want it back in the
                    # usual format.
                    if unpack and unpack_name in group.keys():
                        for i_key in group.keys():

                            if i_key == unpack_name:
                                continue

                            i_result = {}
                            ii_items = zip( group[unpack_name][...].astype( str ), group[i_key][...] )
                            for ii_key, ii_item in ii_items:
                                ii_item = if_byte_then_to_str( ii_item )
                                i_result[ii_key] = ii_item

                            result[i_key] = Dict( i_result )

                        return Dict( result )

                    else:
                        for i_key in group.keys():
                            result[i_key] = recursive_retrieve( current_path, i_key )

                        if look_for_saved_jagged_arrs:
                            # Look for saved jagged arrays
                            if i_key[:len(jagged_flag)] == jagged_flag:
                                unpacked = []
                                for i, result_i in enumerate( result ):
                                    unpacked_i = result['{}{}'.format( jagged_flag, i )]
                                    unpacked_i = if_byte_then_to_str( unpacked_i )
                                    unpacked.append( unpacked_i )
                                return unpacked

                        return Dict( result )

                elif isinstance( item, hdf5_module.Dataset ) or isinstance( item, h5py.Dataset ):
                    try:
                        arr = np.array( item[...] )
                    except NotImplementedError:
                        if sparse:
                            arr = item[()]
                        else:
                            raise Exception( 'Not sure how you got here. If sparse is not turned on the above functionality *should* be implemented.' )

                    if look_for_saved_jagged_arrs:
                        if 'jagged saved as filled' in item.attrs:
                            arr = filled_arr_to_jagged_arr(
                                item[...],
                                item.attrs['fill value'],
                            )
                            return arr

                    # Handle 0-length arrays
                    if arr.shape == ():
                        arr = arr[()]

                    arr = if_byte_then_to_str( arr )

                    return arr

            result = {}
            for key in f.keys():
                result[key] = recursive_retrieve( '', key )

            result = Dict( result )
            result.unpack_name = unpack_name

            # For shallow save files
            if unpack and unpack_name in result.keys():
                true_result = {}

                for i_key in result.keys():

                    if i_key == unpack_name:
                        continue

                    i_result = {}
                    ii_items = zip( result[unpack_name].astype( str ), result[i_key] )
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
    
    def to_json(
        self,
        filepath,
        overwrite_existing_file = True,
    ):
        '''Save the contents as a JSON file.

        Args:
            filepath (str):
                Location to save the hdf5 file at.

            overwrite_existing_file (boolean):
                If True and a file already exists at filepath, delete it prior
                to saving.

            condensed (boolean):
                If True, combine the innermost dictionaries into a condensed
                DataFrame/array-like format.
        '''

        # Check for existing
        if os.path.isfile( filepath ):
            if overwrite_existing_file:
                os.remove( filepath )
            else:
                raise IOError( 'File {} already exists'.format( filepath ) )

        # Format data
        def recursive_format( data ):

            # Turn to regular dictionary
            if isinstance( data, Dict ):
                formatted = data._storage
            elif isinstance( data, np.ndarray ):
                formatted = list( data )
            elif isinstance( data, set ):
                formatted = list( data )
            else:
                formatted = data

            # Repeat
            if isinstance( formatted, dict ):
                for key, item in formatted.items():
                    formatted[key] = recursive_format( item )
            if isinstance( formatted, list ):
                for i, item in enumerate( formatted ):
                    formatted[i] = recursive_format( item )

            return formatted

        data_to_save = recursive_format( self._storage )

        # Save
        with open( filepath, 'w', encoding='utf-8') as f:
            json.dump( data_to_save, f, ensure_ascii=False, indent=4)

    ########################################################################

    @classmethod
    def from_json(
        cls,
        filepath,
        create_nonexisting = False,
    ):
        '''Load a HDF5 file as a verdict Dict.

        Args:
            filepath (str):
                Location to load the hdf5 file from.

            create_nonexisting (bool):
                If True and no data to load, return an empty dictionary.
        '''

        if not os.path.exists( filepath ) and create_nonexisting:
            return Dict({})

        with open( filepath ) as f:
            data = json.load( f )

        return Dict( data )

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

########################################################################

def create_dataset_fixed( f, path, data, hdf5_module, attrs=None ):
    '''Accounts for h5py not recognizing unicode. This is fixed
    in h5py 2.9.0, with PR #1032 (not merged at the time of writing).
    The fix used here is exactly what the PR does.'''
    
    try:
        special_dtype = hdf5_module.special_dtype( vlen=six.text_type )
    except AttributeError:
        special_dtype = h5py.special_dtype( vlen=six.text_type )

    try:
        f.create_dataset( path, data=data )
    except TypeError:
        data = np.array(
            data,
            dtype=special_dtype,
        )
        f.create_dataset( path, data=data )

    if attrs is not None:
        for key, item in attrs.items():
            try:
                f[path].attrs[key] = item
            except TypeError:
                item = np.array(
                    item,
                    dtype=special_dtype,
                )
                f[path].attrs[key] = item

########################################################################

def check_if_jagged_arr( arr ):
    '''Check if an array-like object is contains arrays of
    different sizes.

    Args:
        arr: Object to check.

    Returns:
        bool:
            True if an array-like object with arrays of different sizes.
    '''

    # Check if an array-like
    try:
        len_arr = len( arr )
    except TypeError:
        return False

    if len_arr > 1:

        for i, arr_i in enumerate( arr ):

            # Check if an array
            if pd.api.types.is_list_like( arr_i ):
                l_current = len( arr_i )
            else:
                l_current = 0

            # Check if jagged
            if i != 0:
                if l_current != l_prev:
                    return True
            l_prev = copy.copy( l_current )

            # Recurse
            if check_if_jagged_arr( arr_i ): 
                return True

        # If got to this point, then it's even
        return False

########################################################################

def create_dataset_jagged_arr(
        f,
        current_path,
        arr,
        method = 'filled array',
        fill_value = None,
        jagged_flag = 'jagged',
        hdf5_module = h5py,
    ):
    '''Create a dataset for saving an jagged array.

    Args:
        f (open hdf5 file):
            File to create the dataset on.

        current_path (str):
            Location in the file to save the array to.

        arr (array-like):
            Uneven array to save.

        fill_value :
            Fill value to use when using the filled arr method.

        jagged_flag (str):
            Flag to indicate that this part of the hdf5 file contains part of
            an jagged array-like.
    '''

    if method == 'filled array':

        filled_arr, fill_value = jagged_arr_to_filled_arr( arr, fill_value )

        attrs = {
            'jagged saved as filled': True,
            'fill value': fill_value
        }
        create_dataset_fixed( f, current_path, filled_arr, hdf5_module, attrs=attrs )

    elif method == 'row datasets':
        for i, v in enumerate( arr ):

            used_path = '{}/{}{}'.format( current_path, jagged_flag, i )

            # If v is an jagged array, recurse
            if check_if_jagged_arr( v ):
                create_dataset_jagged_arr(
                    f,
                    used_path,
                    v,
                    method = method,
                    jagged_flag = jagged_flag,
                    hdf5_module = hdf5_module,
                )
            else:
                create_dataset_fixed( f, used_path, v, hdf5_module )

    else:
        raise ValueError( 'Unrecognized jagged arr dataset method, {}'.format( method ) )

########################################################################

def jagged_arr_to_filled_arr( arr, fill_value=None, dtype=None, ):
    '''Convert a jagged array to a uniform filled array of minimum size
    needed to contain all the data.

    Args:
        arr (array-like):
            Jagged array to convert to a filled array.

        fill_value:
            Fill value to use. Defaults to -9999 for integers, NaN otherwise.

        dtype:
            Datatype. Defaults to the datatype of arr, if arr consists of data
            of one type.

    Returns:
        np n-dim array:
            A filled array with minimum dimensions needed to contain arr.
    '''

    def arr_depth( a, level=1 ):
        '''Get the array depth, even for a jagged array.'''

        if len( a ) == 0:
            return level

        depths = []
        for v in a:
            if pd.api.types.is_list_like( v ):
                depths.append( arr_depth( v, level+1 ) )
            else:
                depths.append( level )

        return max( depths )

    def recursive_array_shape( a, s=[], dtypes=[], level=0 ):
        '''Loop through and get max dimensions and datatypes of jagged array'''

        # Get length at current depth
        len_depth = np.array( a ).shape[0]

        # Compare to s
        if len( s ) > level:
            s[level] = max( s[level], len_depth )
        else:
            s.append( len_depth )

        # Recurse
        for v in a:
            if not pd.api.types.is_list_like( v ):
                if type( v ) not in dtypes:
                    if isinstance( v, str ):
                        dtype = len( v )
                    else:
                        dtype = type( v )
                    dtypes.append( dtype )
                continue
            s, dtypes = recursive_array_shape( v, s, dtypes, level+1 )

        return s, dtypes
            
    shape, dtypes = recursive_array_shape( arr )

    # The two sections below are somewhat complicated as a result of these:
    # a) The full length of all strings must be saved.
    # b) Passing dtype to np.full can result in unexpected behavior.
    # c) np.dtype( int ).type( np.nan ) is overly long and complicated.

    # Choose array type
    str_dtype = not False in [ isinstance( _, int ) for _ in dtypes ]
    if str_dtype:
        # The + [3,] gives a minimum length
        dtype = '<U{}'.format( max( dtypes+[ 3, ] ) )
    elif len( dtypes ) > 1 and dtype is None:
        raise TypeError( 'Must choose dtype for jagged arrays with multiple data types.' )
    else:
        dtype = dtypes[0]

    # Choose automatic fill value
    if fill_value is None:
        if str_dtype:
            arr_dtype = dtype
            fill_value = np.nan
        else:
            arr_dtype = np.float64
            try:
                fill_value = dtype( np.nan )
            except ValueError:
                fill_value = dtype( -9999 )
    else:
        arr_dtype = type( fill_value )

    new_arr = np.full( shape, fill_value, dtype=arr_dtype )

    def store_jagged_to_masked( a, m_a, ):
        '''Actually store the jagged array to the masked array.'''

        if arr_depth( a ) == 2:
            for i, v in enumerate( a ):
                m_a[i,:len(v)] = v
            return m_a
        else:
            for i, v in enumerate( a ):
                m_a[i] = store_jagged_to_masked( v, m_a[i] )

        return m_a

    new_arr = store_jagged_to_masked( arr, new_arr )

    return new_arr, fill_value

########################################################################

def filled_arr_to_jagged_arr( arr, fill_value ):
    '''Convert a filled array to a jagged array

    Args:
        arr (np.ndarray):
            Array to convert to a jagged set of lists.

        fill_value:
            Value to skip over.

    Returns:
        A list of lists instead of a filled array.
    '''

    result = []
    for v in arr:
        if pd.api.types.is_list_like( v ):
            result.append( filled_arr_to_jagged_arr( v, fill_value ) )
        else:

            v = if_byte_then_to_str( v )

            if isinstance( v, str ):
                if v == fill_value:
                    continue
            else:
                if np.isclose( v, fill_value ):
                    continue
            result.append( v )

    return result

########################################################################

def if_byte_then_to_str( a ):

    # For arrays
    if pd.api.types.is_list_like( a ):

        # Don't bother for sparse arrays
        if scipy.sparse.issparse( a ):
            return a

        # Don't bother for empty arrays
        if len( a ) == 0:
            return a

        # Assume that if the first element is a byte-type then all are bytes, and vice versa.
        try:
            first_element_index = [ np.array([ 0, ]) for _ in range( len( a.shape ) ) ]
            if hasattr( a[first_element_index][0], 'decode' ):
                    a = a.astype( 'str' )
        # Slower, simpler recursive option when necessary
        except (UnicodeDecodeError, AttributeError):
            a = [ if_byte_then_to_str( _ ) for _ in a ]

    # For single values
    else:
        try:
            a = a.decode( 'UTF-8' )
        except (UnicodeDecodeError, AttributeError):
            pass

    return a
