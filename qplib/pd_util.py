import pandas as pd
import numpy as np
import copy
import os
import datetime

from IPython.display import display
from ipywidgets import interact, widgets
from pandas.api.extensions import register_dataframe_accessor

from .util import log, GREEN, ORANGE, RED, GREEN_LIGHT, ORANGE_LIGHT, RED_LIGHT
from .types import _date, _na, qpDict



def get_df(size='small'):
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

    df_old.loc['x', 'a'] = 1.0
    df_old.loc['x', 'b'] = 1.0
    df_old.loc['x', 'c'] = 1.0

    df_old.loc['y', 'a'] = 2.0
    df_old.loc['y', 'b'] = 2.0
    df_old.loc['y', 'c'] = 2.0

    df_old.loc['z', 'a'] = 3.0
    df_old.loc['z', 'b'] = None
    df_old.loc['z', 'c'] = 3.0


    df_new = pd.DataFrame(columns=['d', 'b', 'a'], index=['y','x2','z'])

    df_new.loc['y', 'd'] = 2.0
    df_new.loc['y', 'b'] = 2.0
    df_new.loc['y', 'a'] = 0.0

    df_new.loc['x2', 'd'] = 1.0
    df_new.loc['x2', 'b'] = 1.0
    df_new.loc['x2', 'a'] = 1.0

    df_new.loc['z', 'd'] = 3.0
    df_new.loc['z', 'b'] = 3.0
    df_new.loc['z', 'a'] = np.nan

    return df_new, df_old



@pd.api.extensions.register_index_accessor('qp')
class indexQpExtension(qpDict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, index):
        self._og = index


@pd.api.extensions.register_series_accessor('qp')
class seriesQpExtension(qpDict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, series):
        self._og = series


@pd.api.extensions.register_dataframe_accessor('qp')
class dfQpExtension(qpDict):
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
            'qp.df.format()', verbosity)
        df.columns = df.columns.str.strip()
             

    if 'meta' in df.columns:
        log('debug: ensuring column "meta" is string', 'qp.df.format()', verbosity)
        df.loc[:, 'meta'] = df.loc[:, 'meta'].astype(str)
    elif add_metadata is True:
        log('info: adding column "meta" at position 0', 'df.format()', verbosity)
        metadata_col = pd.Series('', index=df.index, name='meta')
        df = pd.concat([metadata_col, df], axis=1)

    df.qp = qp_old  
    return df





@pd.api.extensions.register_dataframe_accessor('save')
class DataFrameSave:
    """
    saves a dataframe to a sheet in an excel file. If the file/sheet already exists, the data will be overwritten.

    if a folder named "archive" exists at the chosen path, a timestamped copy of the file will be saved there,
    unless archive=False.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __repr__(self):
        return 'docstring of dataframe accessor df.save():' + self.__doc__
    
    def __call__(
        self,
        path='df.xlsx',
        sheet='data1',
        index=True,
        if_sheet_exists='replace',
        archive=True,
        datefmt='%Y_%m_%d',
        diff_before=None,
        diff_mode='new+',
        verbosity=3,
        ):
        save(
            self.df,
            path,
            sheet,
            index,
            if_sheet_exists,
            archive,
            datefmt,
            diff_before,
            diff_mode,
            verbosity,
            )


def save(
    df,
    path='df.xlsx',
    sheet='data1',
    index=True,
    if_sheet_exists='replace',
    archive=True,
    datefmt='%Y_%m_%d',
    diff_before=None,
    diff_mode='new+',
    verbosity=3,
    ):
    """
    see qp.save() for details
    """
    if diff_before is not None:
        df_old = load(path, sheet, before=diff_before)
        df = _diff(
            df,
            df_old,
            mode=diff_mode,
            verbosity=3,
            )
        

    if not path.endswith('.xlsx'):
        path = f'{path}.xlsx'

    if os.path.isfile(path):
        log(f'warning: file "{path}" already exists. data in sheet "{sheet}" will be overwritten', 'df.save()', verbosity)
        with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)
    else:
        log(f'info: saving df to "{path}" in sheet "{sheet}"', 'df.save()', verbosity)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)


    if archive is True:
        folder = os.path.dirname(path)
        if folder == '':
            folder = os.getcwd()
        archive_folder = f'{folder}/archive'

        if not os.path.isdir(f'{archive_folder}'):
            log(f'warning: did not find archive folder "{archive_folder}"', 'df.save()', verbosity)
            return

        today = datetime.datetime.now().strftime(datefmt)
        name = os.path.basename(path).split('.xlsx')[0]
        path_copy = f'{archive_folder}/{name}_{today}.xlsx'

        if os.path.isfile(path_copy):
            log(f'warning: archive file "{path_copy}" already exists. data in sheet "{sheet}" will be overwritten',
                'df.save()', verbosity)
            with pd.ExcelWriter(path_copy, mode='a', engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
                df.to_excel(writer, sheet_name=sheet, index=index)
        else:
            log(f'info: archiving df to "{path_copy}" in sheet "{sheet}"', 'df.save()', verbosity)
            with pd.ExcelWriter(path_copy, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet, index=index)      



def load(path='df', sheet='data1', index_col=0, before='now', return_date=False, verbosity=3, **kwargs):
    """
    loads .xlsx file from before a given date.
    assumes that the filenames end with a timestamp.
    
    before:
    defines recency of the file
    - now: most recent version
    - today: most recent version before today
    - this day: most recent version before today
    - this week: ...
    - this month: ...
    - this year: ...
    - '2024_01_01': most recent version before 2024_01_01 (accepts many date formats)
    """

    if os.path.isfile(path):
        df = pd.read_excel(path, sheet_name=sheet, index_col=index_col, **kwargs)
        if 'meta' in df.columns:
            df.loc[:, 'meta'] = df.loc[:, 'meta'].apply(lambda x: _na(x, errors='ignore', na=''))
        return df
        
    today = datetime.date.today()


    if before == 'now':
        cutoff = today + datetime.timedelta(days=1)
    elif before == 'today':
        cutoff = today
    elif before == 'this day':
        cutoff = today
    elif before == 'this week':
        cutoff = today - datetime.timedelta(days=today.weekday())
    elif before == 'this month':
        cutoff = today - datetime.timedelta(days=today.day-1)
    elif before == 'this year':
        cutoff = pd.to_datetime(f'{today.year}0101').date()
    else:
        cutoff = _date(before)

    name = os.path.basename(path)
    folder = os.path.dirname(path)
    
    if folder == '':
        folder = os.getcwd()

    timestamps = pd.Series([])
    for file in os.listdir(folder):
        if file.startswith(name) and file.endswith('.xlsx'):
            try:
                timestamp_str = file.split(name)[-1].replace('.xlsx', '')
                timestamp = _date(timestamp_str)
                if timestamp < cutoff and file == f'{name}{timestamp_str}.xlsx':
                    timestamps[timestamp] = timestamp_str
            except:
                pass

    if len(timestamps) == 0:
        log(f'warning: no timestamped files starting with "{name}" found in "{folder}" before {cutoff}',
            'qp.load()', verbosity)
        return None
    else:
        timestamps = timestamps.sort_index()
        latest = timestamps.iloc[len(timestamps) - 1]
        path = f'{folder}/{name}{latest}.xlsx'
        log(f'info: loading "{path}"', 'qp.load()', verbosity)
        df = pd.read_excel(path, sheet_name=sheet, index_col=index_col, **kwargs)
        
        if 'meta' in df.columns:
            df.loc[:, 'meta'] = df.loc[:, 'meta'].apply(lambda x: _na(x, errors='ignore', na=''))
 
        if return_date is True:
            return df, latest
        else:
            return df



def _diff(
    df_new,
    df_old,
    mode='mix',
    output='df',  #df, summary, str, all, print, <filename.xlsx>
    ignore=None,  #col(s) to ignore for comparison
    rename=None,  #rename cols before comparison
    index_col=0,  #column to use as index when reading from file
    max_cols=None,
    max_rows=None,
    prefix_old='old: ',
    verbosity=3,
    ):
    """
    compares two dataframes/csv/excel files and returns differences

    mode:
    not needed for output="summary" or "str" or "print"
    - 'new': creates new dataframe with highlighted value additions, removals and changes
    - 'new+': also shows old values in columns next to new values
    - 'old': creates old dataframe with highlighted value additions, removals and changes
    - 'mix': creates mixture of new and old dataframe with highlighted value additions, removals and changes
    
    output:
    - 'df': returns the dataframe with highlighted differences (using the given mode argument)
    - 'summary': returns a dictionary containing the number of added, removed and changed values and columns for each sheet
    - 'str': returns a string containing a summary of the differences
    - 'all': returns a tuple containing the dataframe, the summary dictionary and the string
    - 'print': prints the string containing a summary of the differences and returns None
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
            df_new,
            df_old,
            mode,
            output,
            ignore,
            rename,
            index_col,
            max_cols,
            max_rows,
            prefix_old,
            verbosity,
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
                df_new,
                df_old,
                mode,
                output,
                ignore,
                rename,
                max_cols, max_rows,
                prefix_old,
                name_new='new df',
                name_old='old df',
                verbosity=verbosity
                )
            string = _diff_str(df_new, df_old, ignore, verbosity)


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
                    log(f'warning: comparison for sheet "diff_summary" will not be written to file since this name is reserved', 'qp._diff.diff()', verbosity)
                    continue
                if hasattr(df, 'data'):
                    df.data['meta'] = df.data['meta'].str.replace('<br>', '\n')
                    df.to_excel(writer, sheet_name=sheet, index=True)
        log(f'info: differences saved to "{output}"', 'qp._diff.diff()', verbosity)
        return df, summary, string
    elif output.endswith('.xlsx'):
        df.to_excel(output, index=index_col)
        log(f'info: differences saved to "{output}"', 'qp._diff.diff()', verbosity)
        return df, summary, string
    else:
        log(f'error: unknown return value: {output}', 'qp._diff.diff()', verbosity)
        return None
   

def _diff_df(
    df_new,
    df_old,
    mode='mix',
    output='df',
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
    if max_rows is None:
        max_rows = 200
    if max_cols is None:
        max_cols = 20
    if output.endswith('.xlsx'):
        newline = '\n'
    else:
        newline = '<br>'

    flag_return = False
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
        log(f'warning: index of {name_new} is not unique. "index_col=None" to use sequential numbering as index', 'qp.diff()', verbosity)
        flag_return = True
    if not df_old.index.is_unique:
        log(f'warning: index of {name_old} is not unique. "index_col=None" to use sequential numbering as index', 'qp.diff()', verbosity)
        flag_return = True
    if flag_return:
        return pd.DataFrame(), summary_empty


    #prepare dataframes
    if rename is None:
        pass
    elif isinstance(rename, dict):
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
    ignore=None,
    rename=None,
    index_col=0,
    max_rows=None,
    max_cols=None,
    prefix_old='old: ',
    verbosity=3,
    ):
    '''
    see _diff() for details
    '''

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

            name_new=f'sheet "{sheet}" in file "{file_new}"'
            name_old=f'sheet "{sheet}" in file "{file_old}"'
            result, changes = _diff_df(
                df_new,
                df_old,
                mode,
                output,
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
            summary.loc[idx, f'is in new file'] = True
            summary.loc[idx, f'is in old file'] = True
            summary.loc[idx, f'index (first col) is unique in new file'] = df_new.index.is_unique
            summary.loc[idx, f'index (first col) is unique in old file'] = df_old.index.is_unique
            for key, val in changes.items():
                summary.loc[idx, key] = val
            
        else:
            log(f'warning: sheet "{sheet}" is only in new file. nothing to compare', 'qp.diff()', verbosity)
            if index_col is None:
                df_new = pd.read_excel(file_new, sheet_name=sheet)
            else:
                df_new = pd.read_excel(file_new, sheet_name=sheet, index_col=index_col)
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
 
