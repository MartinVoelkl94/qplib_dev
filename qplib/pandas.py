
import pandas as pd
import numpy as np
import copy

from .types import _dict, _date
from .util import (
    log,
    _arg_to_list,
    )


def get_df():
    """
    Returns a small sample dataframe containing very messy fake medical data.
    """
    df = pd.DataFrame({
        'ID': [
            10001,
            10002,
            10003,
            20001,
            20002,
            20003,
            30001,
            30002,
            30003,
            30004,
            30005,
            ],
        'name': [
            'John Doe',
            'Jane Smith',
            'Alice Johnson',
            'Bob Brown',
            'eva white',
            'Frank miller',
            'Grace TAYLOR',
            'Harry Clark',
            'IVY GREEN',
            'JAck Williams',
            'john Doe',
            ],
        'date of birth': [
            '1995-01-02',
            '1990/09/14',
            '1985.08.23',
            '19800406',
            '05-11-2007',
            '06-30-1983',
            '28-05-1975',
            '1960Mar08',
            '1955-Jan-09',
            '1950 Sep 10',
            '1945 October 11',
            ],
        'age': [
            -25,
            '30',
            np.nan,
            None,
            '40.0',
            'forty-five',
            'nan',
            'unk',
            '',
            'unknown',
            35,
            ],
        'gender': [
            'M',
            'F',
            'Female',
            'Male',
            'Other',
            'm',
            'ff',
            'NaN',
            None,
            'Mal',
            'female',
            ],
        'height': [
            170,
            '175.5cm',
            None,
            '280',
            'NaN',
            '185',
            '1',
            '6ft 1in',
            -10,
            '',
            200,
            ],
        'weight': [
            70.2,
            '68',
            '72.5lb',
            'na',
            '',
            '75kg',
            None,
            '80.3',
            '130lbs',
            '82',
            -65,
            ],
        'bp systole': [
            '20',
            130,
            'NaN',
            '140',
            '135mmhg',
            '125',
            'NAN',
            '122',
            '',
            130,
            '45',
            ],
        'bp diastole': [
            80,
            '85',
            'nan',
            '90mmHg',
            np.nan,
            '75',
            'NaN',
            None,
            '95',
            '0',
            'NaN',
            ],
        'cholesterol': [
            'Normal',
            'Highe',
            'NaN',
            'GOOD',
            'n.a.',
            'High',
            'Normal',
            'n/a',
            'high',
            '',
            'Normal',
            ],
        'diabetes': [
            'No',
            'yes',
            'N/A',
            'No',
            'Y',
            'Yes',
            'NO',
            None,
            'NaN',
            'n',
            'Yes',
            ],
        'dose': [
            '10kg',
            'NaN',
            '15 mg once a day',
            '20mg',
            '20 Mg',
            '25g',
            'NaN',
            None,
            '30 MG',
            '35',
            '40ml',
            ],
        })
    return df


def get_dfs():
    """
    Returns 2 small, somewhat similar sample dataframes. Mostly for testing qp.diff().
    """

    df_old = pd.DataFrame(columns=['a', 'b', 'c'], index=['x', 'y', 'z', ])

    df_old.insert(0, 'uid', df_old.index)

    df_old.loc['x', 'a'] = 1
    df_old.loc['x', 'b'] = 1
    df_old.loc['x', 'c'] = 1

    df_old.loc['y', 'a'] = 2
    df_old.loc['y', 'b'] = 2
    df_old.loc['y', 'c'] = 2

    df_old.loc['z', 'a'] = 3
    df_old.loc['z', 'b'] = None
    df_old.loc['z', 'c'] = 3


    df_new = pd.DataFrame(columns=['d', 'b', 'a'], index=['y', 'x2', 'z', ])

    df_new.insert(0, 'uid', df_new.index)

    df_new.loc['y', 'd'] = 2
    df_new.loc['y', 'b'] = 2
    df_new.loc['y', 'a'] = 0

    df_new.loc['x2', 'd'] = 1
    df_new.loc['x2', 'b'] = 1
    df_new.loc['x2', 'a'] = 1

    df_new.loc['z', 'd'] = 3
    df_new.loc['z', 'b'] = 3
    df_new.loc['z', 'a'] = np.nan

    return df_old, df_new



@pd.api.extensions.register_index_accessor('qp')
class indexQpExtension(_dict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, index):
        self._og = index


@pd.api.extensions.register_series_accessor('qp')
class seriesQpExtension(_dict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, series):
        self._og = series


@pd.api.extensions.register_dataframe_accessor('qp')
class dfQpExtension(_dict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, df):
        self._og = df




@pd.api.extensions.register_dataframe_accessor('format')
class dfFormatExtension:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, fix_headers=True, add_metadata=True, verbosity=3):
        df = _format_df(
            self.df,
            fix_headers=fix_headers,
            add_metadata=add_metadata,
            verbosity=verbosity,
            )
        return df



def _format_df(
        df,
        fix_headers=True,
        add_metadata=True,
        verbosity=3,
        ):
    """
    Formats dataframe to ensure compatibility with the query language used by df.q().
    """

    qp_old = df.qp
    df = copy.deepcopy(df)

    if fix_headers is True:
        log('info: striping column headers of leading and trailing whitespace',
            'df.format()', verbosity)
        df.columns = df.columns.str.strip()


    if 'meta' in df.columns:
        log('debug: ensuring column "meta" is string', 'df.format()', verbosity)
        df.loc[:, 'meta'] = df.loc[:, 'meta'].astype(str).replace('nan', '')
    elif add_metadata is True:
        log('info: adding column "meta" at position 0', 'df.format()', verbosity)
        metadata_col = pd.Series('', index=df.index, name='meta')
        if df.empty:
            df.insert(0, 'meta', '')
        else:
            df = pd.concat([metadata_col, df], axis=1)

    df.qp = qp_old
    return df




def _to_lines(
        x,
        line_start='#',
        line_stop=' ;\n',
        ):
    if not isinstance(x, list):
        return x
    if len(x) == 1:
        return x[0]
    else:
        x_str = ''
        for i, item in enumerate(x):
            x_str += f'{line_start}{i + 1}: {item}{line_stop}'
    return x_str



def merge(
        left,
        right,
        on='uid',
        include=None,
        exclude=None,
        flatten=None,
        duplicates=True,
        prefix=None,
        line_start='#',
        line_stop=' ;\n',
        verbosity=3,
        ):
    r"""
    Performs a modified left join on two dataframes.
    If the right df has multiple values for the same key,
    they are aggregated into a string with multiple lines:
        *"#1: item1 ;\n#2: item2 ;\n#3: item3 ;"*


    ## Requirements:
    - key-column specified by arg *"on"* in left df should be unique
    - key-column specified by arg *"on"* in right df can be non-unique


    ## Args:

    #### left, right:
    - dfs to join

    #### on:
    - key column(s) to join on

    #### include:
    - all columns specified in include are kept from right df
    - if include is None, all columns from right df are kept

    #### exclude:
    - all columns specified in exclude are removed from right df
    - if exclude is None, no columns are removed from right df

    #### flatten:
    - all columns specified in flatten are not aggregated into a string when non-unique
    - instead a new column for each value is created and merged into the result df

    #### duplicates:
    - True: all columns from right df are kept
    - False: only columns from right df that are not in left df are kept

    #### prefix:
    - None: a sequential integer prefix is generated automatically
    - any string: the prefix is added to all columns from right df


    ## Notes:
    for nice excel formatting use qp.format_excel() or:
    - open resulting excel file
    - select all
    - make all cols very wide
    - click "Wrap Text"
    - auto fit column width
    - align text to top
    """

    #process args
    left = left.copy().fillna('')
    right = right.copy().fillna('')
    include = _arg_to_list(include)
    exclude = _arg_to_list(exclude)
    flatten = _arg_to_list(flatten)

    if on not in left.columns:
        log(f'Error: "{on}" is not in left dataframe', 'qp.merge', verbosity=verbosity)
    elif not left[on].is_unique:
        log(f'warning: column "{on}" is not unique in left dataframe',
            'qp.merge', verbosity=verbosity)
    if on not in right.columns:
        log(f'Error: "{on}" is not in right dataframe', 'qp.merge', verbosity=verbosity)


    #get relevant columns from right df
    if include:
        cols_right = [on]
        cols_right += [col for col in right.columns if col in include and col != on]

    else:
        cols_right = right.columns

    if exclude:
        cols_right = [col for col in cols_right if col not in exclude]

    right = right[cols_right]


    #aggregate repeating rows into lists
    right_compact = right.groupby(on).agg(list)


    #remove duplicate cols (if duplicates is False)
    if duplicates:
        cols = right_compact.columns
    else:
        cols_diff = right_compact.columns.difference(left.columns)
        cols = [col for col in right_compact.columns if col in cols_diff]
        right_compact = right_compact[cols]


    #create new columns from lists (for cols in flatten)
    cols_new = []
    for col in right_compact.columns:
        cols_new.append(col)
        if col in flatten:
            n_max = right_compact[col].apply(lambda x: len(x)).max()
            cols_flat = [f'{col}_{i + 1}' for i in range(n_max)]
            split = pd.DataFrame(right_compact[col].to_list(), columns=cols_flat)
            split.index = right_compact.index
            right_compact = right_compact.merge(
                split,
                left_index=True,
                right_index=True,
                how='left',
                )
            cols_new += cols_flat

    right_compact = right_compact[cols_new]


    #transform lists into multi line strings
    right_compact = right_compact.map(lambda x: _to_lines(x, line_start, line_stop))


    #add prefix to columns
    if prefix is None:
        i = 1
        prefix = '1_'
        while i < 1000:
            for col in left.columns:
                if col.startswith(prefix):
                    i += 1
                    prefix = f'{i}_'
            else:
                break

    right_compact.columns = [f'{prefix}{col}' for col in right_compact.columns]


    #merge dataframes
    result = pd.merge(left, right_compact, how='left', on=on).fillna('')
    return result



def embed(
        df_dest,
        key_dest,
        df_src,
        key_src='uid',
        include=None,
        exclude=None,
        verbosity=3,
        ):
    """
    Embeds rows from a source DataFrame into the destination DataFrame based on a key.

    ## Args:

    #### df_dest, df_src:
    - destination and source dfs

    #### key_dest, key_src:
    - key columns to identify which source rows to embed into which destination values

    #### include:
    - all columns specified in include are kept from the source df
    - if include is None, all columns from the source df are kept

    #### exclude:
    - all columns specified in exclude are removed from the source df
    - if exclude is None, no columns are removed from the source df
    """

    include = _arg_to_list(include)
    exclude = _arg_to_list(exclude)


    if key_dest not in df_dest.columns:
        log(f'error: key "{key_dest}" not found in df_dest', 'qp.embed', verbosity)
        return df_dest
    if key_src not in df_src.columns:
        log(f'error: key "{key_src}" not found in df_src', 'qp.embed', verbosity)
        return df_src

    if not df_src[key_src].is_unique:
        log(f'error: "{key_src}" in df_src is not unique.',
            'qp.embed', verbosity)
        return df_dest

    vals_dest = set(df_dest[key_dest].dropna().unique())
    vals_src = set(df_src[key_src].dropna().unique())
    only_in_dest = vals_dest - vals_src
    if len(only_in_dest) > 0:
        msg = (
            f'warning: {len(only_in_dest)} value(s) in "{key_dest}"'
            f' of df_dest not found in df_src: {only_in_dest}'
            )
        log(msg, 'qp.embed', verbosity)


    if include:
        cols_src = [col for col in df_src.columns if col in include]
    else:
        cols_src = df_src.columns
    if exclude:
        cols_src = [col for col in cols_src if col not in exclude]


    df_merged = pd.DataFrame({
        key_src: df_src[key_src],
        'merged': df_src[key_src].astype(str),
        })
    for col in cols_src:
        df_merged.loc[:, 'merged'] = (
            df_merged['merged']
            + f'\n{col}: '
            + df_src[col].astype(str).fillna('')
            + ' ;'
            )

    result = df_dest.copy()
    temp = df_dest.merge(
        df_merged,
        how='left',
        left_on=key_dest,
        right_on=key_src,
        )
    temp.index = df_dest.index
    result.loc[:, key_dest] = temp['merged']

    return result



def days_between(
        df,
        cols,
        reference_date=None,
        reference_col=None,
        verbosity=3,
        ):
    """
    Calculates the number of days between a reference date
    or column and the specified columns in a DataFrame.
    """

    cols = _arg_to_list(cols)
    df = df.copy()

    if reference_date is None and reference_col is None:
        log('ERROR: no reference date or column provided',
            'qp.days_between', verbosity)
        return df

    if reference_date is not None and reference_col is not None:
        log('ERROR: both reference date and column provided',
            'qp.days_between', verbosity)
        return df

    if 'reference_date' in df.columns:
        log('WARNING: column "reference_col" already exists, overwriting',
            'qp.days_between', verbosity)
    if reference_date is not None:
        df['reference_date'] = _date(reference_date)
        reference_col = str(reference_date)
    else:
        df['reference_date'] = df[reference_col].apply(_date)

    for col in cols:
        if col not in df.columns:
            log(f'ERROR: column {col} not found', 'qp.days_between', verbosity)
            continue

        name = f'days_between_{reference_col}_and_{col}'
        if name in df.columns:
            log(f'WARNING: column "{name}" already exists, overwriting',
                'qp.days_between', verbosity)

        #difference only works with pd.NA, but x.days only works with pd.NaT
        def fix_nas(x):
            return pd.NA if x is pd.NaT else x
        col_formatted = df[col].apply(_date).apply(fix_nas)
        col_diff = col_formatted - df['reference_date']
        df[name] = col_diff.fillna(pd.NaT).apply(lambda x: x.days)

    return df



def deduplicate(obj, name='object', verbosity=3):
    """
    Deduplicate entries in object which can be converted
    to pandas Series by appending consecutive numbers.
    Note that the entries are converted to strings in the process.
    """

    obj = copy.deepcopy(obj)
    class_orig = obj.__class__
    obj = _to_series(obj)

    #Values can only be deduplicated if index is unique
    if not obj.index.is_unique:
        msg = (
            f'info: duplicates found in index of {name}.'
            ' deduplicating by appending consecutive numbers.'
            )
        log(msg, 'qp.diff()', verbosity)
        obj.index = _deduplicate(_to_series(obj.index))

    rounds = 0
    while not obj.is_unique:
        if rounds == 0:
            msg = (
                f'debug: duplicates found in {name}.'
                ' deduplicating by appending consecutive numbers.'
                )
        else:
            msg = (
                f'debug: duplicates still found in {name} after'
                f' {rounds} deduplication rounds.'
                ' deduplicating again by appending consecutive numbers.'
                )
        log(msg, 'qp.diff()', verbosity)
        obj = _deduplicate(obj)
        rounds += 1

    if not isinstance(obj, class_orig):
        try:
            obj = class_orig(obj)
        except Exception as e:
            msg = (
                'Error: could not convert deduplicated Series back to'
                f' original type {class_orig}: {e}'
                )
            log(msg, 'qp.diff()', verbosity)

    return obj


def _to_series(obj, verbosity=3):
    try:
        obj = pd.Series(obj, dtype=str)
    except Exception as e:
        msg = (
            'Error: could not convert input of type'
            f' {type(obj)} to Series: {e}'
            )
        log(msg, 'qp.diff()', verbosity)
    return obj


def _deduplicate(series):
    cumulative_count = series.groupby(series).cumcount()
    duplicates_mask = series.index[cumulative_count > 0]
    duplicates = series[duplicates_mask]
    duplicates_new = (
        duplicates.astype(str)
        + '_'
        + cumulative_count[duplicates_mask].astype(str)
        )
    series[duplicates_mask] = duplicates_new
    return series
