import pandas as pd
import numpy as np
import copy
import os
import datetime

from IPython.display import display
from ipywidgets import interact, widgets
from pandas.api.extensions import register_dataframe_accessor

from .util import log, GREEN, RED, GREEN_LIGHT, ORANGE_LIGHT, RED_LIGHT
from .types import _dict
from .xlsx import hide, format_excel



def get_df():
    """
    Returns a small sample dataframe containing very messy fake medical data.
    """
    df = pd.DataFrame({
        'ID': [10001, 10002, 10003, 20001, 20002, 20003, 30001, 30002, 30003, 30004, 30005],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'eva white', 'Frank miller', 'Grace TAYLOR', 'Harry Clark', 'IVY GREEN', 'JAck Williams', 'john Doe'],
        'date of birth': ['1995-01-02', '1990/09/14', '1985.08.23', '19800406', '05-11-2007', '06-30-1983', '28-05-1975', '1960Mar08', '1955-Jan-09', '1950 Sep 10', '1945 October 11'],
        'age': [-25, '30', np.nan, None, '40.0', 'forty-five', 'nan', 'unk', '', 'unknown', 35],
        'gender': ['M', 'F', 'Female', 'Male', 'Other', 'm', 'ff', 'NaN', None, 'Mal', 'female'],
        'height': [170, '175.5cm', None, '280', 'NaN', '185', '1', '6ft 1in', -10, '', 200],
        'weight': [70.2, '68', '72.5lb', 'na', '', '75kg', None, '80.3', '130lbs', '82', -65],
        'bp systole': ['20', 130, 'NaN', '140', '135mmhg', '125', 'NAN', '122', '', 130, '45'],
        'bp diastole': [80, '85', 'nan', '90mmHg', np.nan, '75', 'NaN', None, '95', '0', 'NaN'],
        'cholesterol': ['Normal', 'Highe', 'NaN', 'GOOD', 'n.a.', 'High', 'Normal', 'n/a', 'high', '', 'Normal'],
        'diabetes': ['No', 'yes', 'N/A', 'No', 'Y', 'Yes', 'NO', None, 'NaN', 'n', 'Yes'],
        'dose': ['10kg', 'NaN', '15 mg once a day', '20mg', '20 Mg', '25g', 'NaN', None, '30 MG', '35', '40ml']
        })
    return df

def get_dfs():
    """
    Returns 2 small, somewhat similar sample dataframes. Mostly for testing qp.diff().
    """

    df_old = pd.DataFrame(columns=['a', 'b', 'c'], index=['x','y','z'])

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


    df_new = pd.DataFrame(columns=['d', 'b', 'a'], index=['y','x2','z'])

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
        df = _format_df(self.df, fix_headers=fix_headers, add_metadata=add_metadata, verbosity=verbosity)    
        return df



def _format_df(df, fix_headers=True, add_metadata=True, verbosity=3):
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
        df.loc[:, 'meta'] = df.loc[:, 'meta'].astype(str)
    elif add_metadata is True:
        log('info: adding column "meta" at position 0', 'df.format()', verbosity)
        metadata_col = pd.Series('', index=df.index, name='meta')
        df = pd.concat([metadata_col, df], axis=1)

    df.qp = qp_old  
    return df




def _to_lines(x, line_start='#', line_stop=' ;\n'):
    if not isinstance(x, list):
        return x
    if len(x) == 1:
        return x[0]
    else:
        x_str = ''
        for i, item in enumerate(x):
            x_str += f'{line_start}{i+1}: {item}{line_stop}'
    return x_str


def merge(left, right, on='uid', flatten=None, duplicates=True, prefix=None, line_start='#', line_stop=' ;\n', verbosity=3):
    """
    Performs a modified left join on two dataframes:
    - key-column specified by arg "on" in left df should be unique
    - key-column specified by arg "on" in right df can be non-unique
    - if right df has multiple values for the same key:
        - they are aggregated into a string with multiple lines
        - aggregated string format: "#1: item1 ;\n#2: item2 ;\n#3: item3 ;"
    - if flatten is not None, the specified columns are flattened into multiple columns
    - left join on key-column
    - if duplicates=True, all columns from right df are kept
    - if duplicates=False, only columns from right df that are not in left df are kept
    - if prefix is None, a sequential integer prefix is generated automatically
    - if prefix is not None, the prefix is added to all columns from right df
    - if verbosity > 0, log messages are printed

    for nice excel formatting use format_excel() or:
    - open resulting excel file
    - select all
    - make all cols very wide
    - click "Wrap Text"
    - auto fit column width
    - align text to top
    """



    #prepare and validate
    left = left.copy().fillna('')
    right = right.copy().fillna('')
    if flatten is None:
        flatten = []
    if not isinstance(flatten, list):
        flatten = [flatten]
    if on not in left.columns:
        log(f'Error: "{on}" is not in left dataframe', 'qp.merge', verbosity=verbosity)
    elif not left[on].is_unique:
        log(f'warning: column "{on}" is not unique in left dataframe', 'qp.merge', verbosity=verbosity)
    if on not in right.columns:
        log(f'Error: "{on}" is not in right dataframe', 'qp.merge', verbosity=verbosity)


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
            cols_flat = [f'{col}_{i+1}' for i in range(n_max)]
            split = pd.DataFrame(right_compact[col].to_list(), columns=cols_flat)
            split.index = right_compact.index
            right_compact = right_compact.merge(split, left_index=True, right_index=True, how='left')
            cols_new += cols_flat

    right_compact = right_compact[cols_new]


    #transform lists into multi line strings
    right_compact = right_compact.map(lambda x: _to_lines(x, line_start=line_start, line_stop=line_stop))


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



def _diff(
    df_new: pd.DataFrame | str,
    df_old: pd.DataFrame | str,
    mode='mix',
    output='df',  #df, summary, str, all, print, <filename.xlsx>
    uid=None,     #None defaults to index
    ignore=None,  #col(s) to ignore for comparison
    rename=None,  #rename cols before comparison
    max_cols=None,
    max_rows=None,
    prefix_old='old: ',
    verbosity=3,
    ):
    """
    compares two dataframes/csv/excel files and returns differences

    mode (not needed for output="summary" or "str" or "print"):
    - 'new': creates new dataframe with highlighted value additions, removals and changes
    - 'new+': also shows old values in columns next to new values (old values are hidden when saved to excel)
    - 'old': creates old dataframe with highlighted value additions, removals and changes
    - 'mix': creates mixture of new and old dataframe with highlighted value additions, removals and changes
    
    output:
    - 'df': returns dataframe(s) with highlighted differences (using the given mode argument). multiple dfs are returned in a dict.
    - 'summary': returns a df containing the number of added, removed and changed values and columns for each sheet
    - 'str': returns a string containing a summary of the differences
    - 'all': returns a tuple containing the dataframe, the summary df and the string
    - 'print': prints a string containing a summary of the differences and returns None
    - '<filename.xlsx>': saves the differences to an excel file with the given name and returns as with 'all'

    ignore:
    - column(s) to ignore for comparison

    rename:
    - dictionary to rename columns before comparison
    - is applied to both dataframes


    Excel comparison:

    when two excel files are compared, all sheets with the same name in both files are compared, meaning that 'df'
    will return a dictionary containing multiple dfs with highlighted differences. 'summary' will return a df containing
    the summary of differences for each sheet.

    requirements for the excel sheets:
    - only sheets with the same name are compared
    - needs a column with unique entries to use as key for comparison (specified by arg "uid")
    - uid must have the same name in all files and sheets

    if uid=None:
    - uses sequential index instead of any given unique column
    - this guarantees uniqueness
    - only works if all sheets have corresponding records in the same order
    """

    if isinstance(df_new, str) and isinstance(df_old, str) \
        and df_new.endswith('.xlsx') and df_old.endswith('.xlsx'):
        df, summary, string = _diff_excel(
            df_new,
            df_old,
            mode,
            output,
            uid,
            ignore,
            rename,
            max_cols,
            max_rows,
            prefix_old,
            verbosity,
            )
        
    
    else:
        if isinstance(df_new, str):
            if df_new.endswith('.csv'):
                df_new = pd.read_csv(df_new)
            elif df_new.endswith('.xlsx'):
                df_new = pd.read_excel(df_new)
            else:
                log(f'error: unknown file extension: {df_new}', 'qp.diff()', verbosity)
                return None

        if isinstance(df_old, str):
            if df_old.endswith('.csv'):
                df_old = pd.read_csv(df_old)
            elif df_old.endswith('.xlsx'):
                df_old = pd.read_excel(df_old)
            else:
                log(f'error: unknown file extension: {df_old}', 'qp.diff()', verbosity)
                return None

        if df_new.equals(df_old):
            df = df_new
            summary = pd.DataFrame()
            string = 'both dataframes are identical'
        else:
            df, summary = _diff_df(
                df_new,
                df_old,
                mode,
                output,
                uid,
                ignore,
                rename,
                max_cols, max_rows,
                prefix_old,
                name_new='new df',
                name_old='old df',
                verbosity=verbosity
                )
            string = _diff_str(df_new, df_old, uid, ignore, rename, verbosity)


    if output == 'df':
        return df
    if output == 'str':
        return string
    if output == 'summary':
        return summary
    if output == 'all':
        return df, summary, string
    if output == 'print':
        print(string)
        return None
    elif output.endswith('.xlsx') and isinstance(df, dict): 
        with pd.ExcelWriter(output) as writer:
            summary.to_excel(writer, sheet_name='diff_summary', index=False)
            for sheet, df in df.items():
                if sheet == 'diff_summary':
                    log(f'warning: comparison for sheet "diff_summary" will not be written to file since this name is reserved', 'qp.diff()', verbosity)
                    continue
                if hasattr(df, 'data'):
                    df.data['meta'] = df.data['meta'].str.replace('<br>', '\n')
                    df.to_excel(writer, sheet_name=sheet, index=True)
        if mode=='new+':
            hide(output, axis='col', patterns=f'{prefix_old}.*', hide=True, verbosity=verbosity)
        format_excel(output)
        log(f'info: differences saved to "{output}"', 'qp.diff()', verbosity)
        return df, summary, string
    elif output.endswith('.xlsx'):
        df.to_excel(output, index=uid)
        if mode=='new+':
            hide(output, axis='col', patterns=f'{prefix_old}.*', hide=True, verbosity=verbosity)
        log(f'info: differences saved to "{output}"', 'qp.diff()', verbosity)
        return df, summary, string
    else:
        log(f'error: unknown return value: {output}', 'qp.diff()', verbosity)
        return None
   

def _diff_df(
    df_new,
    df_old,
    mode='mix',
    output='df',
    uid=None,
    ignore=None,
    rename=None,
    max_rows=None,
    max_cols=None, 
    prefix_old='old: ',
    name_new='new df',
    name_old='old df',
    verbosity=3,
    ):
    '''
    see _diff() for details
    '''

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    ignore.append('meta')
    if uid:
        if uid not in df_new.columns:
            log(f'error: "{uid}" is not in new dataframe', 'qp.diff', verbosity)
            return pd.DataFrame(), {}
        if uid not in df_old.columns:
            log(f'error: "{uid}" is not in old dataframe', 'qp.diff', verbosity)
            return pd.DataFrame(), {}
        df_new.index = df_new[uid]
        df_old.index = df_old[uid]
        ignore.append(uid)
    if max_rows is None:
        max_rows = 200
    if max_cols is None:
        max_cols = 50
    if output.endswith('.xlsx'):
        newline = '\n'
    else:
        newline = '<br>'


    summary_empty = {
        'cols added': None,
        'cols removed': None,
        'rows added': None,
        'rows removed': None,
        'vals added': None,
        'vals removed': None,
        'vals changed': None,
        }
    if not df_new.index.is_unique:
        log(f'warning: uid of {name_new} is not unique. choose a unique column as uid or set "uid=None" to use sequential numbering', 'qp.diff', verbosity)
        return pd.DataFrame(), summary_empty
    if not df_old.index.is_unique:
        log(f'warning: uid of {name_old} is not unique. choose a unique column as uid or set "uid=None" to use sequential numbering', 'qp.diff', verbosity)
        return pd.DataFrame(), summary_empty



    #prepare dataframes
    if rename is None:
        pass
    elif isinstance(rename, dict):
        if 'meta' in rename:
            log('warning: it is not advised to rename the "meta" column', 'qp.diff()', verbosity)
        if 'uid' in rename:
            log('warning: it is not advised to rename the "uid" column', 'qp.diff()', verbosity)
        df_new = df_new.rename(columns=rename)
        df_old = df_old.rename(columns=rename)
    else:
        log('error: rename argument must be a dictionary', 'qp.diff()', verbosity)

    df_new = _format_df(df_new, fix_headers=False, add_metadata=True, verbosity=2)
    df_old = _format_df(df_old, fix_headers=False, add_metadata=True, verbosity=2)



    cols_added = df_new.columns.difference(df_old.columns).difference(ignore)
    cols_removed = df_old.columns.difference(df_new.columns).difference(ignore)
    cols_shared = df_new.columns.intersection(df_old.columns).difference(ignore)

    rows_added = df_new.index.difference(df_old.index)
    rows_removed = df_old.index.difference(df_new.index)
    rows_shared = df_new.index.intersection(df_old.index)

    summary = {
        'cols added': len(cols_added),
        'cols removed': len(cols_removed),
        'rows added': len(rows_added),
        'rows removed': len(rows_removed),
        'vals added': 0,
        'vals removed': 0,
        'vals changed': 0
        }



    #create dfs showing the highlighted changes dependant on mode argument
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
                    if col != uid:
                        cols_new.append(prefix_old + col)

                        if prefix_old + col not in df_diff.columns:
                            cols_add.append(prefix_old + col)

            df_diff = pd.concat([df_diff, pd.DataFrame('', index=df_diff.index, columns=cols_add)], axis=1)
            df_diff_style = pd.concat([df_diff_style, pd.DataFrame('font-style: italic', index=df_diff.index, columns=cols_add)], axis=1)
        
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
        log(f'error: unknown mode: {mode}', 'qp.diff()', verbosity)


    #highlight values in shared columns
    #column 0 contains metadata and is skipped
    cols_shared_no_metadata = [col for col in cols_shared if not col.startswith(prefix_old) and col != 'meta']

    df_new_isna = df_new.loc[rows_shared, cols_shared_no_metadata].isna()
    df_old_isna = df_old.loc[rows_shared, cols_shared_no_metadata].isna()
    df_new_equals_old = df_new.loc[rows_shared, cols_shared_no_metadata] == df_old.loc[rows_shared, cols_shared_no_metadata]

    df_added = df_old_isna & ~df_new_isna
    df_removed = df_new_isna & ~df_old_isna
    df_changed = (~df_new_isna & ~df_old_isna & ~df_new_equals_old).astype(bool)
    #the previous comparison can result in dtype "boolean" instead of "bool"
    #"boolean" masks cannot be used to set values as str, which is needed for the style df

    df_diff_style.loc[rows_shared, cols_shared_no_metadata] += df_added.mask(df_added, f'background-color: {GREEN_LIGHT}').where(df_added, '')
    df_diff_style.loc[rows_shared, cols_shared_no_metadata] += df_removed.mask(df_removed, f'background-color: {RED_LIGHT}').where(df_removed, '')
    df_diff_style.loc[rows_shared, cols_shared_no_metadata] += df_changed.mask(df_changed, f'background-color: {ORANGE_LIGHT}').where(df_changed, '')



    df_added_sum = df_added.sum(axis=1)
    df_removed_sum = df_removed.sum(axis=1)
    df_changed_sum = df_changed.sum(axis=1)

    summary['vals added'] += int(df_added_sum.sum())
    summary['vals removed'] += int(df_removed_sum.sum())
    summary['vals changed'] += int(df_changed_sum.sum())

    df_diff.loc[rows_shared, 'meta'] += df_added_sum.apply(lambda x: f'{newline}vals added: {x}' if x > 0 else '')
    df_diff.loc[rows_shared, 'meta'] += df_removed_sum.apply(lambda x: f'{newline}vals removed: {x}' if x > 0 else '')
    df_diff.loc[rows_shared, 'meta'] += df_changed_sum.apply(lambda x: f'{newline}vals changed: {x}' if x > 0 else '')


    if mode == 'new+':
        cols_shared_metadata = [prefix_old + col for col in cols_shared_no_metadata]
        df_all_modifications = (df_added | df_removed | df_changed)
        df_old_changed = df_old.loc[rows_shared, cols_shared_no_metadata].where(df_all_modifications, '')
        df_diff.loc[rows_shared, cols_shared_metadata] = df_old_changed.values


    if output.endswith('.xlsx'):
        result = df_diff.style.apply(lambda x: _apply_style(x, df_diff_style), axis=None) 
    else:
        if max_cols is not None and max_cols < len(df_diff.columns):
            log(f'warning: highlighting differences in {max_cols} out of {len(df_diff.columns)} columns. change with "max_cols="', 'qp.diff()', verbosity)
        if max_rows is not None and max_rows < len(df_diff.index):
            log(f'warning: highlighting differences in {max_rows} out of {len(df_diff.index)} rows. change with "max_rows="', 'qp.diff()', verbosity)

        df_diff = df_diff.iloc[:max_rows, :max_cols]
        df_diff_style = df_diff_style.iloc[:max_rows, :max_cols]

        #replace "<" and ">" with html entities to prevent them from being interpreted as html tags
        cols_no_metadata = [col for col in df_diff.columns if not col.startswith(prefix_old) and col != 'meta']
        if pd.__version__ >= '2.1.0':
            df_diff.loc[:, cols_no_metadata] = df_diff.loc[:, cols_no_metadata].map(lambda x: _try_replace_gt_lt(x))
        else:
            df_diff.loc[:, cols_no_metadata] = df_diff.loc[:, cols_no_metadata].applymap(lambda x: _try_replace_gt_lt(x))

        result = df_diff.style.apply(lambda x: _apply_style(x, df_diff_style), axis=None)

    return result, summary
    

def _diff_excel(
    file_new='new.xlsx',
    file_old='old.xlsx',
    mode='new+',
    output='df',
    uid=None,
    ignore=None,
    rename=None,
    max_rows=None,
    max_cols=None,
    prefix_old='old: ',
    verbosity=3,
    ):
    '''
    see _diff() for details
    '''
    filename_new = os.path.basename(file_new)
    filename_old = os.path.basename(file_old)

    summary = pd.DataFrame(columns=[
        'sheet',
        'cols added',
        'cols removed',
        'rows added',
        'rows removed',
        'vals added',
        'vals removed',
        'vals changed',
        f'is in "{filename_new}"',
        f'is in "{filename_old}"',
        f'uid is unique in "{filename_new}"',
        f'uid is unique in "{filename_old}"',
        ])
    results = {}
    

    #get names of all sheets in the excel files
    sheets_new = pd.ExcelFile(file_new).sheet_names
    sheets_old = pd.ExcelFile(file_old).sheet_names
    
    #iterate over all sheets
    for sheet in sheets_new:
        if sheet in sheets_old:
            df_new = pd.read_excel(file_new, sheet_name=sheet)
            df_old = pd.read_excel(file_old, sheet_name=sheet)

            name_new=f'sheet "{sheet}" in file "{file_new}"'
            name_old=f'sheet "{sheet}" in file "{file_old}"'
            result, changes = _diff_df(
                df_new,
                df_old,
                mode,
                output,
                uid,
                ignore,
                rename,
                max_rows,
                max_cols,
                prefix_old,
                name_new,
                name_old,
                verbosity,
                )
            
            #check if df is empty
            if isinstance(result, pd.DataFrame) and result.empty:
                log(f'error: comparison was not possible for sheet "{sheet}"', 'qp.diff', verbosity)
            else:
                log(f'info: compared sheet "{sheet}" from both files', 'qp.diff()', verbosity)
            
            results[sheet] = result
        

            idx = len(summary)
            summary.loc[idx, 'sheet'] = sheet
            summary.loc[idx, f'is in "{filename_new}"'] = True
            summary.loc[idx, f'is in "{filename_old}"'] = True
            summary.loc[idx, f'uid is unique in "{filename_new}"'] = df_new.index.is_unique
            summary.loc[idx, f'uid is unique in "{filename_old}"'] = df_old.index.is_unique
            for key, val in changes.items():
                summary.loc[idx, key] = val
            
        else:
            log(f'warning: sheet "{sheet}" is only in new file. nothing to compare', 'qp.diff()', verbosity)
            if uid is None:
                df_new = pd.read_excel(file_new, sheet_name=sheet)
            else:
                df_new = pd.read_excel(file_new, sheet_name=sheet, index_col=uid)
            idx = len(summary)
            summary.loc[idx, 'sheet'] = sheet
            summary.loc[idx, f'is in "{filename_new}"'] = True
            summary.loc[idx, f'is in "{filename_old}"'] = False
            summary.loc[idx, f'uid is unique in "{filename_new}"'] = df_new.index.is_unique
    
    #wip
    string = 'no string version of differences available when comparing excel files'

    return results, summary, string


def _diff_str(df_new, df_old, uid=None, ignore=None, rename=None, verbosity=3):
    """"
    see _diff() for details
    """

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
        ignore.append('meta')
    if uid:
        df_new.index = df_new[uid]
        df_old.index = df_old[uid]
        ignore.append(uid)

    if df_new.equals(df_old):
        return 'both dataframes are identical'
    
    if isinstance(rename, dict):
        df_new = df_new.rename(columns=rename)
        df_old = df_old.rename(columns=rename)
    
    idx_shared = df_new.index.intersection(df_old.index)
    cols_shared = df_new.columns.intersection(df_old.columns).difference(ignore)


    #different dtypes, headers and indices

    dtypes_new = {}
    dtypes_old = {}
    for col in cols_shared:
        if df_new[col].dtype != df_old[col].dtype:
            dtypes_new[col] = df_new[col].dtype
            dtypes_old[col] = df_old[col].dtype

    result = 'only in df_new:\n'
    result += f'dtypes: {dtypes_new}\n'
    result += f'indices: {df_new.index.difference(df_old.index).values}\n'
    result += f'headers: {df_new.columns.difference(df_old.columns).difference(ignore).values}\n'

    result += 'only in df_old:\n'
    result += f'dtypes: {dtypes_old}\n'
    result += f'indices: {df_old.index.difference(df_new.index).values}\n'
    result += f'headers: {df_old.columns.difference(df_new.columns).difference(ignore).values}\n'


    #different values

    if len(cols_shared) > 20 or len(idx_shared) > 200:
        log('warning: too many shared columns or indices to show different values', 'qp.diff()', verbosity)
        result += 'too many shared columns or indices to show different values'

    else:
        df_new_shared = df_new.loc[idx_shared, cols_shared]
        df_old_shared = df_old.loc[idx_shared, cols_shared]
        diffs = df_new_shared != df_old_shared

        temp1 = df_new_shared[diffs].astype(str).fillna('')
        temp2 = df_old_shared[diffs].astype(str).fillna('')
        
        result += f'\ndifferent values in df_new:\n{temp1}\n'
        result += f'\ndifferent values in df_old:\n{temp2}\n'

    return result


def _try_replace_gt_lt(x):
    if isinstance(x, str):
        return x.replace('<', '&lt;').replace('>', '&gt;')
    elif isinstance(x, type):
        return str(x).replace('<', '&lt;').replace('>', '&gt;')
    else:
        return x
    
def _apply_style(x, df_style):
    return df_style
 
