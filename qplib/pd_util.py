import pandas as pd
import numpy as np
import copy
import os
import datetime

from IPython.display import display
from ipywidgets import interact, widgets
from pandas.api.extensions import register_dataframe_accessor

from .util import log, GREEN, ORANGE, RED, GREEN_LIGHT, ORANGE_LIGHT, RED_LIGHT
from .types import qp_date, qp_na, qpDict




def get_df(size='small'):
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
        self._input = None
        self._operators = None
        self._operators_binary = None
        self._modifiers = None
        self._modifiers_binary = None


@pd.api.extensions.register_series_accessor('qp')
class seriesQpExtension(qpDict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, series):
        self._og = series
        self._input = None
        self._operators = None
        self._operators_binary = None
        self._modifiers = None
        self._modifiers_binary = None


@pd.api.extensions.register_dataframe_accessor('qp')
class dfQpExtension(qpDict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, df):
        self._og = df
        self._input = None
        self._operators = None
        self._operators_binary = None
        self._modifiers = None
        self._modifiers_binary = None



@pd.api.extensions.register_dataframe_accessor('check')
class dfCheckExtension:
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __call__(self, verbosity=3):
        _check_df(self.df, verbosity=verbosity)
        return self.df

@pd.api.extensions.register_dataframe_accessor('format')
class dfFormatExtension:
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __call__(self, fix_headers=True, add_metadata=True, verbosity=3):
        df = _format_df(self.df, fix_headers=fix_headers, add_metadata=add_metadata, verbosity=verbosity)    
        return df

def _check_df(df, verbosity=3):
    """
    df.q() uses '&&', '//' and '´' for expression syntax.
    these should not be used in colnames.
    """
    problems_found = False

    if len(df.index) != len(df.index.unique()):
        log('error: index is not unique', 'qp.pd_util._check_df', verbosity)
        problems_found = True

    if len(df.columns) != len(df.columns.unique()):
        log('error: columns are not unique', 'qp.pd_util._check_df', verbosity)
        problems_found = True

    problems = {
        '"&&"': [],
        '"//"': [],
        '"´"': [],
        'leading whitespace': [],
        'trailing whitespace': [],
        }
    metadata = []

    for col in df.columns:
        if isinstance(col, str):
            if '&&' in col:
                problems['"&&"'].append(col)
            if '//' in col:
                problems['"//"'].append(col)
            if '´' in col:
                problems['"´"'].append(col)

            if col.startswith(' '):
                problems['leading whitespace'].append(col)
            if col.endswith(' '):
                problems['trailing whitespace'].append(col)


    for problem, cols in problems.items():
        if len(cols) > 0:
            log(f'warning: the following column headers contain {problem}, use df = df.format(): {cols}',
                'qp.pd_util._check_df', verbosity)
            problems_found = True

    if problems_found is False:
        log('info: df was checked. no problems found', 'qp.pd_util._check_df', verbosity)

def _format_df(df, fix_headers=True, add_metadata=True, verbosity=3):
    qp_old = df.qp
    df = copy.deepcopy(df)

    if fix_headers is True:
        log('info: striping column headers of leading and trailing whitespace, replacing "//" with "/ /", "&&" with "& &" and "´" with "`"',
            'qp.df.format()', verbosity)
        df.columns = df.columns\
            .str.replace('&&', '& &')\
            .str.replace('//', '/ /')\
            .str.replace('´', '`')\
            .str.strip()
        

    if 'meta' in df.columns:
        log('debug: ensuring column "meta" is string', 'qp.df.format()', verbosity)
        df.loc[:, 'meta'] = df.loc[:, 'meta'].astype(str)
    elif add_metadata is True:
        log('info: adding column "meta" at position 0', 'df.format()', verbosity)
        metadata_col = pd.Series('', index=df.index, name='meta')
        df = pd.concat([metadata_col, df], axis=1)

    df.qp = qp_old  
    return df




def _show_differences(
    df_new, df_old, show='mix',
    summary='print',  #print, return, None
    max_cols=200, max_rows=20,
    newline='<br>', prefix_new='', prefix_old='old: ',
    verbosity=3,):
    '''
    shows differences between dataframes
    '''

    if not df_new.index.is_unique:
        log('error: index of new dataframe is not unique', 'qp.diff()', verbosity)
    if not df_old.index.is_unique:
        log('error: index of old dataframe is not unique', 'qp.diff()', verbosity)



    #prepare dataframes
    df_new = _format_df(df_new, fix_headers=False, add_metadata=True, verbosity=verbosity)
    df_old = _format_df(df_old, fix_headers=False, add_metadata=True, verbosity=verbosity)



    cols_added = df_new.columns.difference(df_old.columns)
    cols_removed = df_old.columns.difference(df_new.columns)
    cols_shared = df_new.columns.intersection(df_old.columns)

    rows_added = df_new.index.difference(df_old.index)
    rows_removed = df_old.index.difference(df_new.index)
    rows_shared = df_new.index.intersection(df_old.index)

    changes_all = {
        'cols added': len(cols_added),
        'cols removed': len(cols_removed),
        'rows added': len(rows_added),
        'rows removed': len(rows_removed),
        'vals added': 0,
        'vals removed': 0,
        'vals changed': 0
        }



    #create dfs showing the highlighted changes dependant on "show" settings
    if show in ['new', 'new+']:
        df_diff = copy.deepcopy(df_new)
        df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)

        #add metadata columns
        if show == 'new+':
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



    elif show == 'old':
        df_diff = copy.deepcopy(df_old)
        df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)
        
        df_diff_style.loc[:, cols_removed] = f'background-color: {RED}'
        df_diff_style.loc[rows_removed, :] = f'background-color: {RED}'

        df_diff.loc[rows_removed, 'meta'] += 'removed row'

    elif show == 'mix':
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
        log(f'error: unknown show mode: {show}', 'qp.diff()', verbosity)


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

    changes_all['vals added'] += int(df_added_sum.sum())
    changes_all['vals removed'] += int(df_removed_sum.sum())
    changes_all['vals changed'] += int(df_changed_sum.sum())

    df_diff.loc[rows_shared, 'meta'] += df_added_sum.apply(lambda x: f'{newline}vals added: {x}' if x > 0 else '')
    df_diff.loc[rows_shared, 'meta'] += df_removed_sum.apply(lambda x: f'{newline}vals removed: {x}' if x > 0 else '')
    df_diff.loc[rows_shared, 'meta'] += df_changed_sum.apply(lambda x: f'{newline}vals changed: {x}' if x > 0 else '')


    if show == 'new+':
        cols_shared_metadata = [prefix_old + col for col in cols_shared_no_metadata]
        df_all_modifications = (df_added | df_removed | df_changed)
        df_old_changed = df_old.loc[rows_shared, cols_shared_no_metadata].where(df_all_modifications, '')
        df_diff.loc[rows_shared, cols_shared_metadata] = df_old_changed.values




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
    changes_truncated = {key: val for key,val in changes_all.items() if val > 0}

    if summary == 'print':
        display(changes_truncated)
        return result
    elif summary == 'return':
        return result, changes_truncated
    else:
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


def excel_diff(file_new='new', file_old='old', diff='new+',
    ignore_index=True, to_excel=True,
    max_cols=None, max_rows=None, verbosity=3):
    '''
    shows differences between two excel files.

    specs and requs:
    - only sheets with the same name are compared
    - the first column of each sheet is assumed to be an index
    - index must be unique
    - index must correspond to the same "item" in both sheets

    if ignore_index=True:
    - uses sequential numbers as index instead of any given column
    - uniqueness is guaranteed
    - only works if the all sheets have the same "items" in the same rows
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
        'values added',
        'values removed',
        'values changed',
        ])
    results = {}
    

    #get names of all sheets in the excel files
    sheets_new = pd.ExcelFile(file_new).sheet_names
    sheets_old = pd.ExcelFile(file_old).sheet_names
    
    #iterate over all sheets
    for sheet in sheets_new:
        if sheet in sheets_old:
            if ignore_index:
                df_new = pd.read_excel(file_new, sheet_name=sheet)
                df_old = pd.read_excel(file_old, sheet_name=sheet)
            else:
                df_new = pd.read_excel(file_new, sheet_name=sheet, index_col=0)
                df_old = pd.read_excel(file_old, sheet_name=sheet, index_col=0)

            result, changes = _show_differences(
                df_new, df_old, show=diff, summary='return',
                max_cols=max_cols, max_rows=max_rows, verbosity=verbosity
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

    if to_excel:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        file_out = f'diff_{timestamp}.xlsx'
        with pd.ExcelWriter(file_out) as writer:
            summary.to_excel(writer, sheet_name='summary')
            for sheet, result in results.items():
                result.to_excel(writer, sheet_name=sheet)
        log(f'info: differences saved to "{file_out}"', 'qp.excel_diff()', verbosity)
        
    return summary, results




@pd.api.extensions.register_dataframe_accessor('save')
class DataFrameSave:
    """
    saves the dataframe to a sheet in an excel file. If the file/sheet already exists, the data will be overwritten.

    "archive" controls if and where a timestamped copy of the file is saved:
    False: do not save a copy
    'source': save copy in an archive folder located in current working directory
    'destination': save copy in an archive folder located in the same directory as the file to be saved
    'both': save copy in both locations
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __repr__(self):
        return 'docstring of dataframe accessor df.save():' + self.__doc__
    
    def __call__(
        self,
        path='df.xlsx', sheet='data1', index=True, if_sheet_exists='replace',
        diff_before=None, diff_show='new+',
        archive=True, datefmt='%Y_%m_%d',
        ):
        save(self.df, path=path, sheet=sheet, index=index, if_sheet_exists=if_sheet_exists, archive=archive, datefmt=datefmt, diff_before=diff_before, diff_show=diff_show)

def save(
    df,
    path='df.xlsx', sheet='data1', index=True, if_sheet_exists='replace',
    diff_before=None, diff_show='new+',
    archive=True, datefmt='%Y_%m_%d',
    verbosity=3,
    ):
    """
    saves a dataframe to a sheet in an excel file. If the file/sheet already exists, the data will be overwritten.

    if a folder named "archive" exists at the chosen path, a timestamped copy of the file will be saved there,
    unless archive=False.
    """
    if diff_before is not None:
        df_old, date_old = load(path, sheet, before=diff_before, return_date=True)
        df = _show_differences(df, df_old, show=diff_show, verbose=False, newline='\n', note=f'changes compared to {date_old}')
        


    if not path.endswith('.xlsx'):
        path = f'{path}.xlsx'


    if os.path.isfile(path):
        log(f'warning: file "{path}" already exists. data in sheet "{sheet}" will be overwritten', 'df.save()', verbosity)
        with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)
    else:
        log(f'warning: saving df to "{path}" in sheet "{sheet}"', 'df.save()', verbosity)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)


    #archiving
    folder = os.path.dirname(path)
    if folder == '':
        folder = os.getcwd()
    archive_folder = f'{folder}/archive'

    if archive is True:
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

def load(path='df', sheet='data1', index=0, before='now', return_date=False, verbosity=3, **kwargs):
    """
    loads .xlsx file from before a given date.
    assumes that the filenames end with a timestamp.

    
    "before" defines recency of the file:

    now: most recent version
    today: most recent version before today
    this day: most recent version before today
    this week: ...
    this month: ...
    this year: ...

    '2024_01_01': most recent version before 2024_01_01
    """
    if os.path.isfile(path):
        df = pd.read_excel(path, sheet_name=sheet, index_col=index, **kwargs)
        if 'meta' in df.columns:
            df.loc[:, 'meta'] = df.loc[:, 'meta'].apply(lambda x: qp_na(x, errors='ignore', na=''))
        return df
        
    today = datetime.date.today()

    match before:
        case 'now':
            cutoff = today + datetime.timedelta(days=1)
        case 'today':
            cutoff = today
        case 'this day':
            cutoff = today
        case 'this week':
            cutoff = today - datetime.timedelta(days=today.weekday())
        case 'this month':
            cutoff = today - datetime.timedelta(days=today.day-1)
        case 'this year':
            cutoff = pd.to_datetime(f'{today.year}0101').date()
        case _:
            cutoff = qp_date(before)

    name = os.path.basename(path)
    folder = os.path.dirname(path)
    
    if folder == '':
        folder = os.getcwd()
    if os.path.isdir(f'{folder}/archive'):
        folder = f'{folder}/archive'
    else:
        log(f'info: no archive folder found. looking for most recent file in "{folder}" instead',
            'df.load()', verbosity)

    timestamps = pd.Series([])
    for file in os.listdir(folder):
        if file.startswith(name) and file.endswith('.xlsx'):
            try:
                timestamp_str = file.split(name)[-1].replace('.xlsx', '')
                timestamp = qp_date(timestamp_str)
                if timestamp < cutoff and file == f'{name}{timestamp_str}.xlsx':
                    timestamps[timestamp] = timestamp_str
            except:
                pass

    if len(timestamps) == 0:
        log(f'warning: no timestamped files starting with "{name}" found in "{folder}" before {cutoff}',
            'df.load()', verbosity)
        return None
    else:
        timestamps = timestamps.sort_index()
        latest = timestamps.iloc[len(timestamps) - 1]
        path = f'{folder}/{name}{latest}.xlsx'
        log(f'info: loading "{path}"', 'df.load()', verbosity)
        if return_date is True:
            df = pd.read_excel(path, sheet_name=sheet, index_col=index, **kwargs), latest
        else:
            df = pd.read_excel(path, sheet_name=sheet, index_col=index, **kwargs)
        
        if 'meta' in df.columns:
            df.loc[:, 'meta'] = df.loc[:, 'meta'].apply(lambda x: qp_na(x, errors='ignore', na=''))

        return df


