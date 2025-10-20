
import pandas as pd
import numpy as np
import copy

from .types import _dict, _date
from .xlsx import hide, format_excel
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

    return df_new, df_old



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

    if df.empty:
        log('debug: dataframe is empty, skipping formatting', 'df.format()', verbosity)
        return df

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
        message = (
            f'warning: {len(only_in_dest)} value(s) in "{key_dest}"'
            f' of df_dest not found in df_src: {only_in_dest}'
            )
        log(message, 'qp.embed', verbosity)


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




class Diff:
    def __init__(
        self,
        new: pd.DataFrame | str,
        old: pd.DataFrame | str,
        uid=None,
        ignore=None,  #col(s) to ignore for comparison
        rename=None,  #rename cols before comparison
        verbosity=3,
        ):

        #original args
        self._new = new
        self._old = old
        self._uid = uid
        self._ignore = _arg_to_list(ignore)
        self._rename = rename
        self._verbosity = verbosity

        #processed data
        self.dfs_new = []
        self.dfs_old = []
        self.dfs_diff = []
        self.dfs_diff_styled = []
        self.sheets = []
        self.sheet_in_both_files = []
        self.uid_cols = []
        self.renamed_cols_new = []
        self.renamed_cols_old = []
        self.ignored_cols_new = []
        self.ignored_cols_old = []
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

        #process inputs
        self._get_dfs()
        self._format_dfs()
        self._rename_cols()
        self._get_uids()
        self._ignore_cols()
        self._get_diff_stats()


    def _get_dfs(self):
            
        #when both inputs are excel files, they might
        #contain multiple sheets to be compared
        conditions_excel_comp = (
            isinstance(self._new, str)
            and isinstance(self._old, str)
            and self._new.endswith('.xlsx')
            and self._old.endswith('.xlsx')
            )
        if conditions_excel_comp:
            log('debug: comparing all sheets from 2 excel files', 'qp.diff()', self._verbosity)
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
                    log(f'error: unknown file extension: {self._new}', 'qp.diff()', self._verbosity)
            elif isinstance(self._new, pd.DataFrame):
                df_new = self._new

            else:
                message = f'error: incompatible type for new df'
                log(message, 'qp.diff()', self._verbosity)

            if isinstance(self._old, str):
                if self._old.endswith('.csv'):
                    df_old = pd.read_csv(self._old)
                elif self._old.endswith('.xlsx'):
                    df_old = pd.read_excel(self._old)
                else:
                    log(f'error: unknown file extension: {self._old}', 'qp.diff()', self._verbosity)
            elif isinstance(self._old, pd.DataFrame):
                df_old = self._old
            else:
                message = 'error: incompatible type for old df'
                log(message, 'qp.diff()', self._verbosity)

            self.dfs_new.append(df_new)
            self.dfs_old.append(df_old)
            self.sheets.append(None)
            self.sheet_in_both_files.append(None)


    def _read_excel_sheets(self):

        sheets_new = pd.ExcelFile(self._new).sheet_names
        sheets_old = pd.ExcelFile(self._old).sheet_names
        sheets_all = list(dict.fromkeys(sheets_new + sheets_old))  #preserves order

        for i, sheet in enumerate(sheets_all):

            if sheet in sheets_new and sheet in sheets_old:
                message = f'debug: found sheet "{sheet}" in both files'
                log(message, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'yes'
                df_new = pd.read_excel(self._new, sheet_name=sheet)
                df_old = pd.read_excel(self._old, sheet_name=sheet)

            elif sheet in sheets_new:
                message = f'warning: sheet "{sheet}" is only in new file. cannot compare'
                log(message, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'only in new file'
                df_new = pd.read_excel(self._new, sheet_name=sheet)
                df_old = pd.DataFrame()

            elif sheet in sheets_old:
                message = f'warning: sheet "{sheet}" is only in old file. cannot compare'
                log(message, 'qp.diff()', self._verbosity)
                sheet_in_both_files = 'only in old file'
                df_new = pd.DataFrame()
                df_old = pd.read_excel(self._old, sheet_name=sheet)

            self.dfs_new.append(df_new)
            self.dfs_old.append(df_old)
            self.sheets.append(sheet)
            self.sheet_in_both_files.append(sheet_in_both_files)


    def _format_dfs(self):
        for i, sheet in enumerate(self.sheets):
            self.dfs_new[i] = _format_df(
                self.dfs_new[i],
                fix_headers=False,
                add_metadata=True,
                verbosity=2,
                )
            self.dfs_old[i] = _format_df(
                self.dfs_old[i],
                fix_headers=False,
                add_metadata=True,
                verbosity=2,
                )


    def _rename_cols(self):
            for i, sheet in enumerate(self.sheets):
                df_new = self.dfs_new[i]
                df_old = self.dfs_old[i]
                rename = self._rename
                rename_new = rename_old = ''

                if rename is None:
                    pass
                elif isinstance(rename, dict):
                    if 'meta' in rename:
                        message = 'warning: it is not advised to rename the "meta" column'
                        log(message, 'qp.diff()', self._verbosity)
                    rename_new = {old: new for old, new in rename.items() if old in df_new.columns}
                    rename_old = {old: new for old, new in rename.items() if old in df_old.columns}
                    df_new.rename(columns=rename_new, inplace=True)
                    df_old.rename(columns=rename_old, inplace=True)
                else:
                    log('error: rename argument must be a dictionary', 'qp.diff()', self._verbosity)

                self.renamed_cols_new.append(rename_new)
                self.renamed_cols_old.append(rename_old)


    def _get_uids(self):

        for i, sheet in enumerate(self.sheets):
            
            df_new = self.dfs_new[i]
            df_old = self.dfs_old[i]

            if isinstance(self._uid, dict) and sheet in self._uid:
                uid = self._uid[sheet]
            elif self._uid in df_new.columns and self._uid in df_old.columns:
                uid = self._uid
            else:
                if sheet is None:
                    message = 'warning: no valid uid column specified for dataframe comparison'
                else:
                    message = f'warning: no valid uid column specified for sheet "{sheet}"'
                log(message, 'qp.diff()', self._verbosity)

                uids_potential = df_new.columns.intersection(df_old.columns)
                uids_by_uniqueness = {}
                for uid in uids_potential:
                    unique_in_new = pd.Index(df_new[uid].dropna().unique())
                    unique_in_old = pd.Index(df_old[uid].dropna().unique())
                    unique_shared = unique_in_new.intersection(unique_in_old)
                    uids_by_uniqueness[uid] = len(unique_shared)

                if len(uids_by_uniqueness) > 0:
                    uid = sorted(
                        uids_by_uniqueness.items(),
                        key=lambda item: item[1],
                        reverse=True,
                        )[0][0]
                    log(f'info: found alternative uid "{uid}" for comparison', 'qp.diff', self._verbosity)
                else:
                    uid = None
                    log('warning: no alternative uid found. using index for comparison',
                        'qp.diff', self._verbosity)
            
            if uid is not None:
                df_new.index = df_new[uid]
                df_old.index = df_old[uid]
            self.uid_cols.append(uid)


    def _ignore_cols(self):
        for i in range(len(self.sheets)):
            cols_ignore = self._ignore.copy()
            self.ignored_cols_new.append((
                self.dfs_new[i]
                .columns
                .intersection(cols_ignore)
                .to_list()
                ))
            self.ignored_cols_old.append((
                self.dfs_old[i]
                .columns
                .intersection(cols_ignore)
                .to_list()
                ))
            cols_ignore.append('meta')
            if self.uid_cols[i] is not None:
                cols_ignore.append(self.uid_cols[i])
            self.cols_ignore.append(cols_ignore)


    def _get_diff_stats(self, mode='mix'):

        for i in range(len(self.sheets)):
            df_new = self.dfs_new[i]
            df_old = self.dfs_old[i]
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


    def show(
            self,
            mode='mix',
            sheet=0,
            prefix_old='old: ',
            ):

        df_new = self.dfs_new[sheet]
        df_old = self.dfs_old[sheet]
        uid_col = self.uid_cols[sheet]
        cols_added = self.cols_added[sheet]
        cols_removed = self.cols_removed[sheet]
        cols_shared = self.cols_shared[sheet]
        rows_added = self.rows_added[sheet]
        rows_removed = self.rows_removed[sheet]
        rows_shared = self.rows_shared[sheet]

        
        if mode in ['new', 'new+']:
            df_diff = copy.deepcopy(df_new)
            df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)

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
            df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)

            df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
            df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

            df_diff.loc[rows_removed, 'meta'] += 'removed row'

        elif mode == 'mix':
            inds_old = df_old.index.difference(df_new.index)
            cols_old = df_old.columns.difference(df_new.columns)

            df_diff = pd.concat([df_new, df_old.loc[:, cols_old]], axis=1)
            df_diff.loc[inds_old, :] = df_old.loc[inds_old, :]

            df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)

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

        df_new_isna = df_new.loc[rows_shared, cols_shared_no_metadata].isna()
        df_old_isna = df_old.loc[rows_shared, cols_shared_no_metadata].isna()
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
            message = (
                'warning: more than 100 000 cells are being formatted.'
                'while this might not cause performance issues for formatting,'
                'the result might be slow to render, especially in jupyter notebooks.'
                )
            log(message, 'qp.diff()', self._verbosity)

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


    def summary(self):
        summary = pd.DataFrame({
            'sheets': self.sheets,
            'in both files': self.sheet_in_both_files,
            'uid col': self.uid_cols,
            'renamed cols in new df': self.renamed_cols_new,
            'renamed cols in old df': self.renamed_cols_old,
            'ignored cols in new df': self.ignored_cols_new,
            'ignored cols in old df': self.ignored_cols_old,
            'cols ignored for comparison': self.cols_ignore,
            'cols added': [len(x) for x in self.cols_added],
            'cols removed': [len(x) for x in self.cols_removed],
            'cols shared': [len(x) for x in self.cols_shared],
            'rows added': [len(x) for x in self.rows_added],
            'rows removed': [len(x) for x in self.rows_removed],
            'rows shared': [len(x) for x in self.rows_shared],
            # 'vals added': self.vals_added,
            # 'vals removed': self.vals_removed,
            # 'vals changed': self.vals_changed,
            })
        return summary


def _try_replace_gt_lt(x):
    if isinstance(x, str):
        return x.replace('<', '&lt;').replace('>', '&gt;')
    elif isinstance(x, type):
        return str(x).replace('<', '&lt;').replace('>', '&gt;')
    else:
        return x
