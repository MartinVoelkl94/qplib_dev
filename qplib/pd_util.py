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
    Returns 2 small, slightly similar sample dataframes. Mostly for testing qp.diff().
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
        df = _diff(df, df_old, show=diff_show, verbose=False, newline='\n')
        


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
                timestamp = _date(timestamp_str)
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
            df.loc[:, 'meta'] = df.loc[:, 'meta'].apply(lambda x: _na(x, errors='ignore', na=''))

        return df


