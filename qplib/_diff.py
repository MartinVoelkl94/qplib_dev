import pandas as pd
import numpy as np
import copy
from .pd_util import _format_df
from .util import log, GREEN, ORANGE, RED, GREEN_LIGHT, ORANGE_LIGHT, RED_LIGHT




def _diff(
    df_new, df_old,
    mode='mix',  returns='df',  #df, summary, str, all, print, <filename.xlsx>
    index_col=0, ignore=None,  #col(s) to ignore for comparison
    max_cols=None, max_rows=None,
    prefix_old='old: ', verbosity=3):
    """
    compares two dataframes/csv/excel files and returns differences

    mode:
    not needed for returns="summary" or "str" or "print"
    - 'new': creates new dataframe with highlighted value additions, removals and changes
    - 'new+': also shows old values in columns next to new values
    - 'old': creates old dataframe with highlighted value additions, removals and changes
    - 'mix': creates mixture of new and old dataframe with highlighted value additions, removals and changes
    
    returns:
    - 'df': returns the dataframe with highlighted differences (using the given mode argument)
    - 'summary': returns a dictionary containing the number of added, removed and changed values and columns for each sheet
    - 'str': returns a string containing a summary of the differences
    - 'all': returns a tuple containing the dataframe, the summary dictionary and the string
    - 'print': prints the string containing a summary of the differences and returns None
    - '<filename.xlsx>': saves the differences to an excel file with the given name and returns as with 'all'

    
    Excel comparison:

    when two excel files are compared, all sheets with the same name in both files are compared, meaning that 'df'
    will return a dictionary containing multiple dfs with highlighted differences. 'summary' will return a df containing
    the summary of differences for each sheet.

    requirements for the excel sheets:
    - only sheets with the same name are compared
    - needs a unique column to use as index, or sequential order of records
    - index must be unique
    - index must correspond to the same "item" in both sheets

    if index_col=None:
    - uses sequential numbers as index instead of any given column
    - uniqueness is guaranteed
    - only works if all sheets have the same "items" in the same rows
    """

    if isinstance(df_new, str) and isinstance(df_old, str) \
        and df_new.endswith('.xlsx') and df_old.endswith('.xlsx'):
        df, summary, string = _diff_excel(
            df_new, df_old,
            mode, returns,
            index_col, ignore,
            max_cols, max_rows,
            prefix_old, verbosity
            )
        
    
    else:
        if isinstance(df_new, str):
            if df_new.endswith('.csv'):
                df_new = pd.read_csv(df_new)
            if df_new.endswith('.xlsx'):
                df_new = pd.read_excel(df_new)

        if isinstance(df_old, str):
            if df_old.endswith('.csv'):
                df_new = pd.read_csv(df_old)
            if df_old.endswith('.xlsx'):
                df_new = pd.read_excel(df_old)

        if df_new.equals(df_old):
            df = pd.DataFrame()
            summary = {}
            string = 'both dataframes are identical'
        else:
            df, summary = _diff_df(
                df_new, df_old,
                mode, returns,
                ignore,
                max_cols, max_rows,
                prefix_old, verbosity
                )
            string = _diff_str(df_new, df_old, ignore, verbosity)


    if returns == 'df':
        return df
    if returns == 'str':
        return string
    if returns == 'summary':
        return summary
    if returns == 'all':
        return df, summary, string
    if returns == 'print':
        print(string)
        return None
    elif returns.endswith('.xlsx') and isinstance(df, dict): 
        with pd.ExcelWriter(returns) as writer:
            summary.to_excel(writer, sheet_name='summary', index=False)
            if index_col:
                index = True
            else:
                index = False

            for sheet, df in df.items():
                df.data['meta'] = df.data['meta'].str.replace('<br>', '\n')
                df.to_excel(writer, sheet_name=sheet, index=index)
        log(f'info: differences saved to "{returns}"', 'qp._diff.diff()', verbosity)
        return df, summary, string
    elif returns.endswith('.xlsx'):
        df.to_excel(returns, index=index_col)
        log(f'info: differences saved to "{returns}"', 'qp._diff.diff()', verbosity)
        return df, summary, string
    else:
        log(f'error: unknown return value: {returns}', 'qp._diff.diff()', verbosity)
        return None
   

def _diff_df(
    df_new, df_old,
    mode='mix', returns='df',
    ignore=None,
    max_rows=None, max_cols=None, 
    prefix_old='old: ',
    verbosity=3):
    '''
    see _diff() for details
    '''
    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    if max_rows is None:
        max_rows = 200
    if max_cols is None:
        max_cols = 20
    if returns.endswith('.xlsx'):
        newline = '\n'
    else:
        newline = '<br>'


    if not df_new.index.is_unique:
        log('error: index of new dataframe is not unique', 'qp.diff()', verbosity)
    if not df_old.index.is_unique:
        log('error: index of old dataframe is not unique', 'qp.diff()', verbosity)



    #prepare dataframes
    df_new = _format_df(df_new, fix_headers=False, add_metadata=True, verbosity=verbosity)
    df_old = _format_df(df_old, fix_headers=False, add_metadata=True, verbosity=verbosity)



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
    df_changed = ~df_new_isna & ~df_old_isna & ~df_new_equals_old

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


    if returns.endswith('.xlsx'):
        result = df_diff.style.apply(lambda x: _apply_style(x, df_diff_style), axis=None) 
    else:
        if max_cols is not None and max_cols < len(df_diff.columns):
            log(f'warning: showing {max_cols} out of {len(df_diff.columns)} columns', 'qp.diff()', verbosity)
        if max_rows is not None and max_rows < len(df_diff.index):
            log(f'warning: showing {max_rows} out of {len(df_diff.index)} rows', 'qp.diff()', verbosity)

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
    file_new='new.xlsx', file_old='old.xlsx',
    mode='new+', returns='df',
    index_col=0, ignore=None,
    max_rows=None, max_cols=None, prefix_old='old: ',
    verbosity=3):
    '''
    see _diff() for details
    '''

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]

    summary = pd.DataFrame(columns=[
        'sheet',
        f'is in new file',
        f'is in old file',
        f'index (first col) is unique in new file',
        f'index (first col) is unique in old file',
        'cols added',
        'cols removed',
        'rows added',
        'rows removed',
        'vals added',
        'vals removed',
        'vals changed',
        ])
    results = {}
    

    #get names of all sheets in the excel files
    sheets_new = pd.ExcelFile(file_new).sheet_names
    sheets_old = pd.ExcelFile(file_old).sheet_names
    
    #iterate over all sheets
    for sheet in sheets_new:
        if sheet in sheets_old:
            if index_col is None:
                df_new = pd.read_excel(file_new, sheet_name=sheet)
                df_old = pd.read_excel(file_old, sheet_name=sheet)
            else:
                df_new = pd.read_excel(file_new, sheet_name=sheet, index_col=index_col)
                df_old = pd.read_excel(file_old, sheet_name=sheet, index_col=index_col)

            result, changes = _diff_df(
                df_new, df_old,
                mode, returns,
                ignore,
                max_rows, max_cols,
                prefix_old, verbosity
                )
            
            results[sheet] = result
        

            idx = len(summary)
            summary.loc[idx, 'sheet'] = sheet
            summary.loc[idx, f'is in new file'] = True
            summary.loc[idx, f'is in old file'] = True
            summary.loc[idx, f'index (first col) is unique in new file'] = df_new.index.is_unique
            summary.loc[idx, f'index (first col) is unique in old file'] = df_old.index.is_unique
            for key, val in changes.items():
                summary.loc[idx, key] = val
            
        else:
            idx = len(summary)
            summary.loc[idx, 'sheet'] = sheet
            summary.loc[idx, f'is in new file'] = True
            summary.loc[idx, f'is in old file'] = False
            summary.loc[idx, f'index (first col) is unique in new file'] = df_new.index.is_unique
    
    #wip
    string = 'no string version of differences available when comparing excel files'

    return results, summary, string


def _diff_str(df_new, df_old, ignore=None, verbosity=3):
    """"
    see _diff() for details
    """

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]

    if df_new.equals(df_old):
        return 'both dataframes are identical'
    
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
        log('warning: too many shared columns or indices to show different values', 'qp._diff.diff()', verbosity)
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
 

