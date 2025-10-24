
import pandas as pd
import numpy as np
import copy

from .types import _dict, _date
from .excel import hide, format_excel
from .util import (
    log,
    _arg_to_list,
    GREEN,
    RED,
    GREEN_LIGHT,
    ORANGE_LIGHT,
    RED_LIGHT,
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



def deduplicate(obj, name='object', verbosity=0):
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



class Diff:
    """
    Calculates differences between dataframes,
    csv or excel files and returns a diff object.


    Parameters
    ----------

    old, new : pd.DataFrame of filepath to CSV or Excel file

    uid : which column to use as a unique identifier to specify
        which rows correspond to each other in old and new data.
        If None, the function will try to find a suitable column automatically.
        If no suitable column is found, the index will be used.
        If False, the index will be used.
        when comparing multiple sheets from excel files, a dictionary
        with sheet names as keys and uid column names as values can be provided.

    ignore : column name or list of column names to ignore for comparison

    rename : dictionary to rename columns before comparison.
        Note that renaming is done before uid and columns to ignore
        are determined, meaning that those must use the new column names.
        This can also be used to fix situations where corrresponding columns
        in old and new data have different names (see examples)


    Examples
    --------

    basic usage:

    >>> import qplib as qp
    >>> diff = qp.diff(df_old, df_new)
    >>> diff.show()  #returns df with highlighted differences
    >>> diff.summary()  #returns df with summary stats
    >>> diff.str()  #string version of summary stats
    >>> diff.print()  #prints the string version
    >>> diff.to_excel('diff.xlsx')  #writes summary and dfs to file


    align inconsistently named columns before comparison:

    >>> corrections = {
            'year of birth': 'yob',
            'birthyear': 'yob',
            },
    >>> qp.diff(
            df_old,
            df_new,
            rename=corrections,
            )
    """

    def __init__(
            self,
            old: pd.DataFrame | str,
            new: pd.DataFrame | str,
            uid=None,
            ignore=None,  #col(s) to ignore for comparison
            rename=None,  #rename cols before comparison
            verbosity=3,
            ):

        #original args
        self._old = old
        self._new = new
        self._uid = uid
        self._ignore = _arg_to_list(ignore)
        self._rename = rename
        self._verbosity = verbosity

        #processed data
        self.dfs_old = []
        self.dfs_new = []
        self.sheets = []
        self.sheet_in_both_files = []
        self.uid_cols = []
        self.cols_renamed_old = []
        self.cols_renamed_new = []
        self.cols_ignored_old = []
        self.cols_ignored_new = []
        self.cols_ignore = []
        self.cols_added = []
        self.cols_removed = []
        self.cols_shared = []
        self.rows_added = []
        self.rows_removed = []
        self.rows_shared = []
        self.vals_added = []
        self.vals_removed = []
        self.vals_changed = []
        self.dtypes_changed = []

        #process inputs
        self._get_dfs()
        self._format_dfs()
        self._rename_cols()
        self._get_uids()
        self._ignore_cols()
        self._get_diff_stats()


    def _get_dfs(self):
        """
        Load and prepare DataFrames from input sources.

        Handles loading from various sources (DataFrames, CSV files, Excel files)
        and determines whether to compare single sheets or multiple Excel sheets.
        Populates self.dfs_old, self.dfs_new, self.sheets, and self.sheet_in_both_files.
        """

        #when both inputs are excel files, they might
        #contain multiple sheets to be compared
        conditions_excel_comp = (
            isinstance(self._old, str)
            and isinstance(self._new, str)
            and self._old.endswith('.xlsx')
            and self._new.endswith('.xlsx')
            )
        if conditions_excel_comp:
            msg = 'debug: comparing all sheets from 2 excel files'
            log(msg, 'qp.diff()', self._verbosity)
            self._read_excel_sheets()


        #only 2 dfs need to be compared if input is
        #2 dfs, 2 csvs or 1 df and 1 csv/excel file
        else:

            if isinstance(self._new, str):
                if self._new.endswith('.csv'):
                    df_new = pd.read_csv(self._new)
                elif self._new.endswith('.xlsx'):
                    df_new = pd.read_excel(self._new)
                else:
                    msg = f'error: unknown file extension: {self._new}'
                    log(msg, 'qp.diff()', self._verbosity)
            elif isinstance(self._new, pd.DataFrame):
                df_new = self._new

            else:
                msg = 'error: incompatible type for new df'
                log(msg, 'qp.diff()', self._verbosity)

            if isinstance(self._old, str):
                if self._old.endswith('.csv'):
                    df_old = pd.read_csv(self._old)
                elif self._old.endswith('.xlsx'):
                    df_old = pd.read_excel(self._old)
                else:
                    msg = f'error: unknown file extension: {self._old}'
                    log(msg, 'qp.diff()', self._verbosity)
            elif isinstance(self._old, pd.DataFrame):
                df_old = self._old
            else:
                msg = 'error: incompatible type for old df'
                log(msg, 'qp.diff()', self._verbosity)

            self.dfs_old.append(df_old)
            self.dfs_new.append(df_new)
            self.sheets.append(None)
            self.sheet_in_both_files.append(None)


    def _read_excel_sheets(self):
        """
        Read all sheets from two Excel files.

        Identifies sheets present in both files, old file only, or new file only.
        Loads DataFrames for each sheet and tracks their comparison status.
        Populates sheet-specific data in the instance lists.
        """

        sheets_old = pd.ExcelFile(self._old).sheet_names
        sheets_new = pd.ExcelFile(self._new).sheet_names
        sheets_all = list(dict.fromkeys(sheets_new + sheets_old))  #preserves order

        for i, sheet in enumerate(sheets_all):

            if sheet in sheets_old and sheet in sheets_new:
                msg = f'debug: found sheet "{sheet}" in both files'
                log(msg, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'yes'
                df_old = pd.read_excel(self._old, sheet_name=sheet)
                df_new = pd.read_excel(self._new, sheet_name=sheet)

            elif sheet in sheets_old:
                msg = f'debug: sheet "{sheet}" is only in old file.'
                log(msg, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'only in old file'
                df_old = pd.read_excel(self._old, sheet_name=sheet)
                df_new = pd.DataFrame()

            elif sheet in sheets_new:
                msg = f'debug: sheet "{sheet}" is only in new file.'
                log(msg, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'only in new file'
                df_old = pd.DataFrame()
                df_new = pd.read_excel(self._new, sheet_name=sheet)

            self.dfs_old.append(df_old)
            self.dfs_new.append(df_new)
            self.sheets.append(sheet)
            self.sheet_in_both_files.append(sheet_in_both_files)


    def _format_dfs(self):
        """
        Apply consistent formatting to all loaded DataFrames.

        Standardizes DataFrame structure by fixing headers and adding metadata columns
        to ensure consistent comparison.
        """
        for i in range(len(self.sheets)):
            self.dfs_old[i] = _format_df(
                self.dfs_old[i],
                fix_headers=False,
                add_metadata=True,
                verbosity=2,
                )
            self.dfs_new[i] = _format_df(
                self.dfs_new[i],
                fix_headers=False,
                add_metadata=True,
                verbosity=2,
                )


    def _rename_cols(self):
        """
        Apply column renaming to DataFrames before comparison.

        Uses the rename dictionary to standardize column names between
        old and new DataFrames. Tracks which columns were renamed in each
        DataFrame and warns about potential issues with renaming metadata columns.
        """
        for i in range(len(self.sheets)):
            df_old = self.dfs_old[i]
            df_new = self.dfs_new[i]
            rename = self._rename
            rename_old = rename_new = ''

            if rename is None:
                pass
            elif isinstance(rename, dict):
                if 'meta' in rename:
                    msg = 'warning: it is not advised to rename the "meta" column'
                    log(msg, 'qp.diff()', self._verbosity)
                rename_old = {
                    old: new
                    for old, new
                    in rename.items()
                    if old in df_old.columns
                    }
                rename_new = {
                    old: new
                    for old, new
                    in rename.items()
                    if old in df_new.columns
                    }
                df_old.rename(columns=rename_old, inplace=True)
                df_new.rename(columns=rename_new, inplace=True)
            else:
                msg = 'error: rename argument must be a dictionary'
                log(msg, 'qp.diff()', self._verbosity)

            self.cols_renamed_old.append(rename_old)
            self.cols_renamed_new.append(rename_new)


    def _get_uids(self):
        """
        Determine unique identifier columns for each comparison.

        For each pair of dataframes, attempts to use the specified uid column or
        automatically finds the best alternative based on uniqueness and overlap
        between dataframes. Sets dataframe indices based on the selected uid column.
        Falls back to using the existing index if no suitable uid column is found.
        """

        for i, sheet in enumerate(self.sheets):

            df_old = self.dfs_old[i]
            df_new = self.dfs_new[i]

            if isinstance(self._uid, dict) and sheet in self._uid:
                uid = self._uid[sheet]
            elif self._uid in df_old.columns and self._uid in df_new.columns:
                uid = self._uid
            elif self._uid is False:
                uid = None
            else:
                if self._uid is None:
                    msg = 'debug: searching for alternative uid column'
                elif sheet is None:
                    msg = 'warning: no valid uid column specified for comparison'
                else:
                    msg = f'warning: no valid uid column specified for sheet "{sheet}"'
                log(msg, 'qp.diff()', self._verbosity)

                uids_potential = df_new.columns.intersection(df_old.columns)
                uids_by_uniqueness = {}
                for uid in uids_potential:
                    unique_in_old = pd.Index(df_old[uid].dropna()).unique()
                    unique_in_new = pd.Index(df_new[uid].dropna()).unique()
                    unique_shared = unique_in_new.intersection(unique_in_old)
                    uids_by_uniqueness[uid] = len(unique_shared)

                if len(uids_by_uniqueness) > 0:
                    uid = sorted(
                        uids_by_uniqueness.items(),
                        key=lambda item: item[1],
                        reverse=True,
                        )[0][0]
                    msg = f'debug: found alternative uid "{uid}" for comparison'
                    log(msg, 'qp.diff()', self._verbosity)
                else:
                    uid = None
                    msg = 'info: no alternative uid found. using index for comparison'
                    log(msg, 'qp.diff()', self._verbosity)

            if uid is not None:
                df_old.index = deduplicate(
                    df_old[uid],
                    name=f'{uid} in old df',
                    verbosity=self._verbosity,
                    )
                df_new.index = deduplicate(
                    df_new[uid],
                    name=f'{uid} in new df',
                    verbosity=self._verbosity,
                    )
            self.uid_cols.append(uid)


    def _ignore_cols(self):
        """
        Identify and track columns to ignore during comparison.

        Builds lists of columns to exclude from difference calculations,
        including user-specified ignore columns, metadata columns, and
        uid columns. Tracks which ignore columns are present in each DataFrame.
        """

        for i in range(len(self.sheets)):

            cols_ignore = self._ignore.copy()

            self.cols_ignored_old.append((
                self.dfs_old[i]
                .columns
                .intersection(cols_ignore)
                .to_list()
                ))

            self.cols_ignored_new.append((
                self.dfs_new[i]
                .columns
                .intersection(cols_ignore)
                .to_list()
                ))

            cols_ignore.append('meta')
            if self.uid_cols[i] is not None:
                cols_ignore.append(self.uid_cols[i])
            self.cols_ignore.append(cols_ignore)


    def _get_diff_stats(self):
        """
        Calculate comprehensive difference statistics for each sheet.

        Analyzes DataFrames to identify added/removed columns and rows,
        shared elements, and data type changes. This method populates
        the stats lists used by .summary(), .str() and .print().
        """

        for i in range(len(self.sheets)):
            df_old = self.dfs_old[i]
            df_new = self.dfs_new[i]
            cols_ignore = self.cols_ignore[i]

            self.cols_added.append((
                df_new
                .columns
                .difference(df_old.columns)
                .difference(cols_ignore)
                ))
            self.cols_removed.append((
                df_old
                .columns
                .difference(df_new.columns)
                .difference(cols_ignore)
                ))
            self.cols_shared.append((
                df_new
                .columns
                .intersection(df_old.columns)
                .difference(cols_ignore)
                ))

            self.rows_added.append((
                df_new
                .index
                .difference(df_old.index)
                ))
            self.rows_removed.append((
                df_old
                .index
                .difference(df_new.index)
                ))
            self.rows_shared.append((
                df_new
                .index
                .intersection(df_old.index)
                ))

            different_dtypes = {}
            for col in self.cols_shared[i]:
                if df_old[col].dtype != df_new[col].dtype:
                    changed = {
                        'old': df_old[col].dtype,
                        'new': df_new[col].dtype
                        }
                    different_dtypes[col] = changed
            if len(different_dtypes) > 0:
                self.dtypes_changed.append(different_dtypes)
            else:
                self.dtypes_changed.append(None)


    def show(
            self,
            mode='mix',
            sheet=0,
            prefix_old='old: ',
            ):
        """
        Generate a styled DataFrame showing differences between datasets.
        Differences are highlighted with color-coded styles,
        supporting multiple visualization modes.


        Parameters
        ----------

        mode : str, default 'mix'
            Display mode for differences:
            - 'new': Show new DataFrame with added and changed elements highlighted
            - 'new+': Show new DataFrame with old values in additional (hidden) columns
            - 'old': Show old DataFrame with removed elements highlighted
            - 'mix': Combine both DataFrames showing all changes

        sheet : int or str, default 0
            Sheet index or name to display in case multiple
            sheets from excel files were compared

        prefix_old : str, default 'old: '
            Prefix for columns showing old values (used in 'new+' mode)


        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame with color-coded differences, or None if sheet not found
        """

        if sheet in self.sheets:
            ind = self.sheets.index(sheet)
        elif isinstance(sheet, int) and sheet < len(self.sheets):
            ind = sheet
        else:
            log(f'error: sheet "{sheet}" not found', 'qp.diff()', self._verbosity)
            return None

        df_old = self.dfs_old[ind]
        df_new = self.dfs_new[ind]
        uid_col = self.uid_cols[ind]
        cols_added = self.cols_added[ind]
        cols_removed = self.cols_removed[ind]
        cols_shared = self.cols_shared[ind]
        rows_added = self.rows_added[ind]
        rows_removed = self.rows_removed[ind]
        rows_shared = self.rows_shared[ind]


        if df_new.empty:
            df_diff = copy.deepcopy(df_old)
            df_diff_style = pd.DataFrame(
                f'background-color: {RED}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            df_diff['meta'] += 'removed sheet'


        elif df_old.empty:
            df_diff = copy.deepcopy(df_new)
            df_diff_style = pd.DataFrame(
                f'background-color: {GREEN}',
                index=df_diff.index,
                columns=df_diff.columns,
                )
            df_diff['meta'] += 'added sheet'


        elif mode in ['new', 'new+']:
            df_diff = copy.deepcopy(df_new)
            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            #add metadata columns
            if mode == 'new+':
                cols_new = ['meta']
                cols_add = []
                for col in df_diff.columns:
                    if not col.startswith(prefix_old) and col != 'meta':
                        cols_new.append(col)
                        if col != uid_col:
                            cols_new.append(prefix_old + col)
                            if prefix_old + col not in df_diff.columns:
                                cols_add.append(prefix_old + col)

                df_diff = pd.concat(
                    [
                        df_diff,
                        pd.DataFrame('', index=df_diff.index, columns=cols_add)
                    ],
                    axis=1,
                    )
                df_diff_style = pd.concat(
                    [
                        df_diff_style,
                        pd.DataFrame(
                            'font-style: italic',
                            index=df_diff.index,
                            columns=cols_add
                            )
                    ],
                    axis=1,
                    )
                df_diff = df_diff[cols_new]
                df_diff_style = df_diff_style[cols_new]


            df_diff_style.loc[:, cols_added] = f'background-color: {GREEN}'
            df_diff_style.loc[rows_added, :] = f'background-color: {GREEN}'

            df_diff.loc[rows_added, 'meta'] += 'added row'


        elif mode == 'old':
            df_diff = copy.deepcopy(df_old)
            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            df_diff.loc[rows_removed, 'meta'] += 'removed row'

        elif mode == 'mix':
            inds_old = df_old.index.difference(df_new.index)
            cols_old = df_old.columns.difference(df_new.columns)

            df_diff = pd.concat([df_new, df_old.loc[:, cols_old]], axis=1)
            df_diff.loc[inds_old, :] = df_old.loc[inds_old, :]

            df_diff_style = pd.DataFrame(
                '',
                index=df_diff.index,
                columns=df_diff.columns,
                )

            df_diff_style.loc[:, cols_added] = f'background-color: {GREEN}'
            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_added, :] = f'background-color: {GREEN}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            df_diff.loc[rows_added, 'meta'] += 'added row'
            df_diff.loc[rows_removed, 'meta'] += 'removed row'

        else:
            log(f'error: unknown mode: {mode}', 'qp.diff()', self._verbosity)


        #highlight values in shared columns
        #column 0 contains metadata and is skipped
        cols_shared_no_metadata = [
            col for col
            in cols_shared
            if not col.startswith(prefix_old) and col != 'meta'
            ]

        df_old_isna = df_old.loc[rows_shared, cols_shared_no_metadata].isna()
        df_new_isna = df_new.loc[rows_shared, cols_shared_no_metadata].isna()
        df_new_equals_old = (
            df_new.loc[rows_shared, cols_shared_no_metadata]
            == df_old.loc[rows_shared, cols_shared_no_metadata]
            )

        df_added = df_old_isna & ~df_new_isna
        df_removed = df_new_isna & ~df_old_isna
        df_changed = (~df_new_isna & ~df_old_isna & ~df_new_equals_old).astype(bool)
        #the previous comparison can result in dtype "boolean" instead of "bool"
        #"boolean" masks cannot be used to set values as str

        df_diff_style.loc[rows_shared, cols_shared_no_metadata] += (
            df_added
            .mask(df_added, f'background-color: {GREEN_LIGHT}')
            .where(df_added, '')
            )

        df_diff_style.loc[rows_shared, cols_shared_no_metadata] += (
            df_removed
            .mask(df_removed, f'background-color: {RED_LIGHT}')
            .where(df_removed, '')
            )

        df_diff_style.loc[rows_shared, cols_shared_no_metadata] += (
            df_changed
            .mask(df_changed, f'background-color: {ORANGE_LIGHT}')
            .where(df_changed, '')
            )



        df_added_sum = df_added.sum(axis=1)
        df_removed_sum = df_removed.sum(axis=1)
        df_changed_sum = df_changed.sum(axis=1)

        self.vals_added.append(int(df_added_sum.sum()))
        self.vals_removed.append(int(df_removed_sum.sum()))
        self.vals_changed.append(int(df_changed_sum.sum()))

        df_diff.loc[rows_shared, 'meta'] += (
            df_added_sum
            .apply(lambda x: f'<br>vals added: {x}' if x > 0 else '')
            )
        df_diff.loc[rows_shared, 'meta'] += (
            df_removed_sum
            .apply(lambda x: f'<br>vals removed: {x}' if x > 0 else '')
            )
        df_diff.loc[rows_shared, 'meta'] += (
            df_changed_sum
            .apply(lambda x: f'<br>vals changed: {x}' if x > 0 else '')
            )


        if mode == 'new+':
            cols_shared_metadata = [prefix_old + col for col in cols_shared_no_metadata]
            df_all_modifications = (df_added | df_removed | df_changed)
            df_old_changed = (
                df_old
                .loc[rows_shared, cols_shared_no_metadata]
                .where(df_all_modifications, '')
                )
            df_diff.loc[rows_shared, cols_shared_metadata] = df_old_changed.values


        if len(df_diff.columns) * len(df_diff.index) > 100_000:
            msg = (
                'warning: more than 100 000 cells are being formatted.'
                'while this might not cause performance issues for formatting,'
                'the result might be slow to render, especially in jupyter notebooks.'
                )
            log(msg, 'qp.diff()', self._verbosity)

        #replace "<" and ">" with html entities to prevent interpretation as html tags
        cols_no_metadata = [
            col for col
            in df_diff.columns
            if not col.startswith(prefix_old) and col != 'meta'
            ]

        if pd.__version__ >= '2.1.0':
            df_diff.loc[:, cols_no_metadata] = (
                df_diff
                .loc[:, cols_no_metadata]
                .map(lambda x: _try_replace_gt_lt(x))
                )
        else:
            df_diff.loc[:, cols_no_metadata] = (
                df_diff
                .loc[:, cols_no_metadata]
                .applymap(lambda x: _try_replace_gt_lt(x))
                )


        diff_styled = df_diff.style.apply(lambda x: df_diff_style, axis=None)
        return diff_styled


    def summary(self, linebreak='<br>'):
        """
        Generate a comprehensive summary dataframe of differences between datasets.
        Dataframe contains statistics about added/removed columns and rows,
        shared elements, data type changes, and other metadata.

        Parameters
        ----------
        linebreak : str, default '&lt;br&gt;'
            String to use for line breaks in the summary output.
            Use '&lt;br&gt;' for HTML display or '\\n' for plain text.

        Returns
        -------
        pandas.io.formats.style.Styler
            Styled DataFrame containing summary statistics with left-aligned,
            pre-wrapped text formatting
        """
        summary = pd.DataFrame()

        if len(self.sheets) > 1 or self.sheets[0] is not None:
            summary['sheets'] = self.sheets
            summary['in both files'] = self.sheet_in_both_files

        summary['uid col'] = _cols_to_str(self.uid_cols)
        summary['cols shared'] = [len(x) for x in self.cols_shared]
        summary['rows shared'] = [len(x) for x in self.rows_shared]
        summary['cols added'] = _iters_to_str(self.cols_added, linebreak)
        summary['cols removed'] = _iters_to_str(self.cols_removed, linebreak)
        summary['rows added'] = _iters_to_str(self.rows_added, linebreak)
        summary['rows removed'] = _iters_to_str(self.rows_removed, linebreak)

        #these 3 are calculated as part of show(), so only include them
        #if all sheets have been processed with show()
        if (
            len(self.vals_added)
            == len(self.vals_removed)
            == len(self.vals_changed)
            == len(self.sheets)
                ):
            summary['vals added'] = self.vals_added
            summary['vals removed'] = self.vals_removed
            summary['vals changed'] = self.vals_changed

        summary['dtypes changed'] = _nested_dicts_to_str(self.dtypes_changed, linebreak)
        summary['cols renamed in new'] = _dicts_to_str(self.cols_renamed_new, linebreak)
        summary['cols renamed in old'] = _dicts_to_str(self.cols_renamed_old, linebreak)
        summary['cols ignored in new'] = _iters_to_str(self.cols_ignored_new, linebreak)
        summary['cols ignored in old'] = _iters_to_str(self.cols_ignored_old, linebreak)

        summary = summary.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'pre-wrap',
            })
        return summary


    def to_excel(
            self,
            path,
            mode='mix',
            index=False,
            prefix_old='old: ',
            linebreak='\n',
            ):
        """
        Export diff results to an Excel file with formatting.

        Creates an Excel file containing a summary sheet and individual
        sheets for each comparison with highlighted differences. Applies
        Excel-specific formatting and hides old value columns.


        Parameters
        ----------
        path : str
            File path for the output Excel file
        mode : str, default 'mix'
            Display mode for differences (see show() method for options)
        index : bool, default False
            Whether to include row indices in the Excel output
        prefix_old : str, default 'old: '
            Prefix for columns showing old values (used in 'new+' mode)
        linebreak : str, default '\\n'
            String to use for line breaks in Excel cells
        """

        with pd.ExcelWriter(path) as writer:

            #placeholder. summary is updated based on the results
            #of the individual sheet comparisons, but should still
            #be at the beginning of the file.
            pd.DataFrame().to_excel(
                writer,
                sheet_name='diff_summary',
                )

            for i, sheet in enumerate(self.sheets):

                if sheet is None:
                    sheet = 'diff'

                if sheet == 'diff_summary':
                    msg = (
                        'warning: comparison for sheet "diff_summary" will not'
                        ' be written to file since this name is reserved'
                        )
                    log(msg, 'qp.diff()', self._verbosity)

                else:

                    result = self.show(
                        mode=mode,
                        sheet=i,
                        prefix_old=prefix_old,
                        )

                    result.data['meta'] = result.data['meta'].str.replace('<br>', '\n')

                    result.to_excel(
                        writer,
                        sheet_name=sheet,
                        index=index,
                        )

            self.summary(linebreak).to_excel(
                writer,
                sheet_name='diff_summary',
                index=index,
                )

        if mode == 'new+':
            hide(
                path,
                axis='col',
                patterns=f'{prefix_old}.*',
                hide=True,
                verbosity=self._verbosity,
                )
        format_excel(path)
        log(f'info: differences saved to "{path}"', 'qp.diff()', self._verbosity)


    def print(self):
        """
        Print the string representation of differences to console.

        Convenience method that prints the output of str() method,
        providing a readable summary of all differences found.
        """
        print(self.str())


    def str(self):
        """
        Get string representation of differences.

        Convenience method that calls __str__() to return a formatted
        string summary of all differences found between datasets.
        """
        return str(self)


    def __str__(self):
        """
        Generate a human-readable string summary of all differences.

        Creates a detailed text summary showing differences for single
        DataFrame comparisons or multi-sheet Excel file comparisons.
        Handles cases where datasets are identical.
        """
        summary = self.summary()
        if len(self.sheets) == 1 and self.sheets[0] is None:
            if self.dfs_new[0].equals(self.dfs_old[0]):
                string = 'both dataframes are identical'
            else:
                string = 'Diff between 2 dataframes\n'
                string += _sheet_to_str(summary, 0)
        else:
            string = f'Diff between 2 excel files with {len(self.sheets)} sheets\n'
            for i, sheet in enumerate(self.sheets):
                if self.dfs_new[i].equals(self.dfs_old[i]):
                    string += f'\nSheet "{sheet}" is identical in both files\n'
                else:
                    string += f'\nSheet: {sheet}\n'
                    string += _sheet_to_str(summary, i)
        return string


def _sheet_to_str(summary, row):
    string = ''
    one_liners = [
        'in both files',
        'uid col',
        'cols shared',
        'rows shared',
        ]
    for col in summary.data.columns:
        if col == 'sheets':
            continue
        elif col in one_liners:
            string += f'  {col}: {summary.data.loc[row, col]}\n'
        else:
            string += f'  {col}:\n    {summary.data.loc[row, col]}\n'
    return string.replace('<br>', '\n    ')


def _cols_to_str(iter):
    iter_new = []
    for col in iter:
        if col is None:
            iter_new.append('')
        else:
            iter_new.append(str(col))
    return iter_new


def _dicts_to_str(iter, linebreak='<br>'):
    iter_new = []
    for dictionary in iter:
        if dictionary:
            iter_new.append(
                f';{linebreak}'.join(
                    [f'{k} -> {v}' for k, v in dictionary.items()]
                    )
                )
        else:
            iter_new.append('')
    return iter_new


def _iters_to_str(iters, linebreak='<br>'):
    iters_new = []
    for iter in iters:
        iters_new.append(f';{linebreak}'.join([str(x) for x in iter]))
    return iters_new


def _nested_dicts_to_str(iter, linebreak='<br>'):
    iter_new = []
    for dictionary in iter:
        string = ''
        if dictionary:
            for k, v in dictionary.items():
                string += f'{k}: {v["old"]} -> {v["new"]}{linebreak}'
        iter_new.append(string)
    return iter_new


def _try_replace_gt_lt(x):
    if isinstance(x, str):
        return x.replace('<', '&lt;').replace('>', '&gt;')
    elif isinstance(x, type):
        return str(x).replace('<', '&lt;').replace('>', '&gt;')
    else:
        return x
