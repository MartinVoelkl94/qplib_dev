import pandas as pd
import numpy as np
import copy
import os
import datetime

from IPython.display import display
from ipywidgets import interact, widgets
from pandas.api.extensions import register_dataframe_accessor

from .util import log, qpDict
from .types import qp_date
from .types import qp_na


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
        self._annotated = None


@pd.api.extensions.register_series_accessor('qp')
class seriesQpExtension(qpDict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, series):
        self._og = series
        self._input = None
        self._operators = None
        self._annotated = None


@pd.api.extensions.register_dataframe_accessor('qp')
class dfQpExtension(qpDict):
    """
    stores parameters and data for custom extensions.
    """
    def __init__(self, df):
        self._og = df
        self._input = None
        self._operators = None
        self._modifiers = None
        self._annotated = None
        self._metadata_added = False



@pd.api.extensions.register_dataframe_accessor('format')
class dfFormatExtension:
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __call__(self, fix_headers=True, add_metadata=True, **kwargs):
        df = copy.deepcopy(self.df)

        if fix_headers is True:
            log('striping column headers of leading and trailing whitespace, replacing "//" with "/ /", "&&" with "& &" and ">>" with "> >"', level='info', source='qp.df.format()')
            df.columns = df.columns\
                .str.replace('&&', '& &')\
                .str.replace('//', '/ /')\
                .str.replace('>>', '> >')\
                .str.strip()
            
        if add_metadata is True:
            log('adding metadata row and column', level='info', source='qp.df.format()')
            df = _prepare_df(df)
            
        return df


def _check_df(df):
    """
    df.q() uses '&&', '//' and '>>' for expression syntax.
    these should not be used in colnames.
    """
    problems = {
        '"&&"': [],
        '"//"': [],
        '">>"': [],
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
            if '>>' in col:
                problems['">>"'].append(col)

            if col.startswith(' '):
                problems['leading whitespace'].append(col)
            if col.endswith(' '):
                problems['trailing whitespace'].append(col)

            if col.startswith('#'):
                metadata.append(col)

    for problem, cols in problems.items():
        if len(cols) > 0:
            log(f'the following column headers contain {problem}, use df = df.format(): {cols}', level='warning', source='qp.pd_util._check_df')

    if df.qp._metadata_added is False and len(metadata) > 0:
        log('the following column headers starting with "#" will be treated as metadata and ignored by most operations', level='warning', source='qp.pd_util._check_df')
  



def diff(df_new, df_old, show='mix', verbosity=3,
    max_cols=200, max_rows=20,
    newline='<br>', note=''):
    '''
    shows differences between dataframes
    '''

    #prepare dataframes
    df_new = _prepare_df(df_new)
    df_old = _prepare_df(df_old)



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
            cols_new = ['#']
            cols_add = []
            for col in df_diff.columns:
                if not col.startswith('#'):
                    cols_new.append(col)
                    cols_new.append(f'#{col}')

                    if f'#{col}' not in df_diff.columns:
                        cols_add.append(f'#{col}')

            df_diff = pd.concat([df_diff, pd.DataFrame('', index=df_diff.index, columns=cols_add)], axis=1)
            df_diff_style = pd.concat([df_diff_style, pd.DataFrame('font-style: italic', index=df_diff.index, columns=cols_add)], axis=1)
        
            df_diff = df_diff[cols_new]
            df_diff_style = df_diff_style[cols_new]


        df_diff_style.loc[:, cols_added] = 'background-color: #6dae51'  #green
        df_diff_style.loc[rows_added, :] = 'background-color: #6dae51'  #green

        df_diff.loc['#', cols_added] += 'added col'
        df_diff.loc[rows_added, '#'] += 'added row'



    elif show == 'old':
        df_diff = copy.deepcopy(df_old)
        df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)
        
        df_diff_style.loc[:, cols_removed] = 'background-color: #db2727'  #red
        df_diff_style.loc[rows_removed, :] = 'background-color: #db2727'  #red

        df_diff.loc['#', cols_removed] += 'removed col'
        df_diff.loc[rows_removed, '#'] += 'removed row'

    elif show == 'mix':
        inds_old = df_old.index.difference(df_new.index)
        cols_old = df_old.columns.difference(df_new.columns)

        df_diff = pd.concat([df_new, df_old.loc[:, cols_old]], axis=1)
        df_diff.loc[inds_old, :] = df_old.loc[inds_old, :]

        df_diff_style = pd.DataFrame('', index=df_diff.index, columns=df_diff.columns)

        df_diff_style.loc[:, cols_added] = 'background-color: #6dae51'  #green
        df_diff_style.loc[:, cols_removed] = 'background-color: #db2727'  #red
        df_diff_style.loc[rows_added, :] = 'background-color: #6dae51'  #green
        df_diff_style.loc[rows_removed, :] = 'background-color: #db2727'  #red

        df_diff.loc['#', cols_added] += 'added col'
        df_diff.loc['#', cols_removed] += 'removed col'
        df_diff.loc[rows_added, '#'] += 'added row'
        df_diff.loc[rows_removed, '#'] += 'removed row'

    else:
        log(f'unknown show mode: {show}', 'error', source='qp.diff()')


    #highlight values in shared columns
    #column 0 contains metadata and is skipped
    cols_shared_no_metadata = [col for col in cols_shared if not col.startswith('#')]
    rows_shared_no_metadata = [row for row in rows_shared if row != '#']

    df_new_isna = df_new.loc[rows_shared_no_metadata, cols_shared_no_metadata].isna()
    df_old_isna = df_old.loc[rows_shared_no_metadata, cols_shared_no_metadata].isna()
    df_new_equals_old = df_new.loc[rows_shared_no_metadata, cols_shared_no_metadata] == df_old.loc[rows_shared_no_metadata, cols_shared_no_metadata]

    df_added = df_old_isna & ~df_new_isna
    df_removed = df_new_isna & ~df_old_isna
    df_changed = ~df_new_isna & ~df_old_isna & ~df_new_equals_old

    df_diff_style.loc[rows_shared_no_metadata, cols_shared_no_metadata] += df_added.mask(df_added, 'background-color: #c0e7b0').where(df_added, '')  #light green
    df_diff_style.loc[rows_shared_no_metadata, cols_shared_no_metadata] += df_removed.mask(df_removed, 'background-color: #f39191').where(df_removed, '')  #light red
    df_diff_style.loc[rows_shared_no_metadata, cols_shared_no_metadata] += df_changed.mask(df_changed, 'background-color: #ffd480').where(df_changed, '')  #light orange



    df_added_sum = df_added.sum(axis=1)
    df_removed_sum = df_removed.sum(axis=1)
    df_changed_sum = df_changed.sum(axis=1)

    changes_all['vals added'] += df_added_sum.sum()
    changes_all['vals removed'] += df_removed_sum.sum()
    changes_all['vals changed'] += df_changed_sum.sum()

    df_diff.loc[rows_shared_no_metadata, '#'] += df_added_sum.apply(lambda x: f'{newline}vals added: {x}' if x > 0 else '')
    df_diff.loc[rows_shared_no_metadata, '#'] += df_removed_sum.apply(lambda x: f'{newline}vals removed: {x}' if x > 0 else '')
    df_diff.loc[rows_shared_no_metadata, '#'] += df_changed_sum.apply(lambda x: f'{newline}vals changed: {x}' if x > 0 else '')


    df_diff.loc['#', cols_shared_no_metadata] += df_added.sum().apply(lambda x: f'{newline}vals added: {x}' if x > 0 else '')
    df_diff.loc['#', cols_shared_no_metadata] += df_removed.sum().apply(lambda x: f'{newline}vals removed: {x}' if x > 0 else '')
    df_diff.loc['#', cols_shared_no_metadata] += df_changed.sum().apply(lambda x: f'{newline}vals changed: {x}' if x > 0 else '')


    if show == 'new+':
        cols_shared_metadata = [f'#{col}' for col in cols_shared_no_metadata]
        df_all_modifications = (df_added | df_removed | df_changed)#.add_prefix('#', axis='columns')
    
        df_old_filtered_values = df_old.loc[rows_shared_no_metadata, cols_shared_no_metadata].where(df_all_modifications, '')
        df_old_filtered_text = df_old_filtered_values.mask(df_all_modifications, f'{newline}old: ')
        df_old_filtered = df_old_filtered_text + df_old_filtered_values.astype(str)

        df_diff.loc[rows_shared_no_metadata, cols_shared_metadata] += df_old_filtered.add_prefix('#', axis='columns').values


    if note != '':
        df_diff.loc['#', '#'] += note

    if verbosity >= 2:
        if max_cols is not None and max_cols < len(df_diff.columns):
            log(f'showing {max_cols} out of {len(df_diff.columns)} columns', level='warning', source='qp.diff()')
        if max_rows is not None and max_rows < len(df_diff.index):
            log(f'showing {max_rows} out of {len(df_diff.index)} rows', level='warning', source='qp.diff()')

    if verbosity >= 3:
        changes_truncated = {key: val for key,val in changes_all.items() if val > 0}
        display(changes_truncated)

    df_diff = df_diff.iloc[:max_rows, :max_cols]
    df_diff_style = df_diff_style.iloc[:max_rows, :max_cols]

    #replace "<" and ">" with html entities to prevent them from being interpreted as html tags
    rows_no_metadata = [row for row in df_diff.index if row != '#']
    cols_no_metadata = [col for col in df_diff.columns if not col.startswith('#')]
    if pd.__version__ >= '2.1.0':
        df_diff.loc[rows_no_metadata, cols_no_metadata] = df_diff.loc[rows_no_metadata, cols_no_metadata].map(lambda x: _try_replace_gt_lt(x))
    else:
        df_diff.loc[rows_no_metadata, cols_no_metadata] = df_diff.loc[rows_no_metadata, cols_no_metadata].applymap(lambda x: _try_replace_gt_lt(x))


    result = df_diff.style.apply(lambda x: _apply_style(x, df_diff_style), axis=None)
    return result


def _prepare_df(df):
    df = df.copy()
    if len(df.index) != len(df.index.unique()):
        log('index is not unique', 'error', source='qp.diff()')

    if len(df.columns) != len(df.columns.unique()):
        log('columns are not unique', 'error', source='qp.diff()')

    #metadata is stored in the first row and column
    if '#' in df.index:
        df.loc['#', :] = df.loc['#', :].apply(lambda x: qp_na(x, errors='ignore', na=''))
    else:
        #inserting metadata row at the top, sadly a bit hacky
        #because there does not seem to be an inplace function for that
        df_temp = df.copy()
        index_temp = df.index
        index_changed = pd.Index(['#', *index_temp])
        
        df.loc['#'] = ''
        df.set_index(index_changed, inplace=True)
        df.loc[index_temp, :] = df_temp
        df.loc['#'] = ''

    if '#' in df.columns:
        df.loc[:, '#'] = df.loc[:, '#'].apply(lambda x: qp_na(x, errors='ignore', na=''))
    else:
        #inserting metadata column at the start
        metadata_col = pd.Series('', index=df.index, name='#')
        df = pd.concat([metadata_col, df], axis=1)

    df.qp._metadata_added = True

    return df

def _try_replace_gt_lt(x):
    if isinstance(x, str):
        return x.replace('<', '&lt;').replace('>', '&gt;')
    elif isinstance(x, type):
        return str(x).replace('<', '&lt;').replace('>', '&gt;')
    else:
        return x
    
def _apply_style(x, df_style):
    return df_style



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
    ):
    """
    saves a dataframe to a sheet in an excel file. If the file/sheet already exists, the data will be overwritten.

    if a folder named "archive" exists at the chosen path, a timestamped copy of the file will be saved there,
    unless archive=False.
    """
    if diff_before is not None:
        df_old, date_old = load(path, sheet, before=diff_before, return_date=True)
        df = diff(df, df_old, show=diff_show, verbose=False, newline='\n', note=f'changes compared to {date_old}')
        


    if not path.endswith('.xlsx'):
        path = f'{path}.xlsx'


    if os.path.isfile(path):
        log(f'file "{path}" already exists. data in sheet "{sheet}" will be overwritten', level='warning', source='df.save()')
        with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)
    else:
        log(f'saving df to "{path}" in sheet "{sheet}"', level='info', source='df.save()')
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=index)


    #archiving
    folder = os.path.dirname(path)
    if folder == '':
        folder = os.getcwd()
    archive_folder = f'{folder}/archive'

    if archive is True:
        if not os.path.isdir(f'{archive_folder}'):
            log(f'did not find archive folder "{archive_folder}"', level='warning', source='df.save()')
            return

        today = datetime.datetime.now().strftime(datefmt)
        name = os.path.basename(path).split('.xlsx')[0]
        path_copy = f'{archive_folder}/{name}_{today}.xlsx'

        if os.path.isfile(path_copy):
            log(f'archive file "{path_copy}" already exists. data in sheet "{sheet}" will be overwritten', level='warning', source='df.save()')
            with pd.ExcelWriter(path_copy, mode='a', engine='openpyxl', if_sheet_exists=if_sheet_exists) as writer:
                df.to_excel(writer, sheet_name=sheet, index=index)
        else:
            log(f'archiving df to "{path_copy}" in sheet "{sheet}"', level='info', source='df.save()')
            with pd.ExcelWriter(path_copy, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet, index=index)      


def load(path='df', sheet='data1', index=0, before='now', return_date=False, **kwargs):
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
        if '#' in df.index:
            df.loc['#', :] = df.loc['#', :].apply(lambda x: qp_na(x, errors='ignore', na=''))
        if '#' in df.columns:
            df.loc[:, '#'] = df.loc[:, '#'].apply(lambda x: qp_na(x, errors='ignore', na=''))
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
        log(f'no archive folder found. looking for most recent file in "{folder}" instead', level='info', source='df.load()')

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
        log(f'no timestamped files starting with "{name}" found in "{folder}" before {cutoff}', level='warning', source='df.load()')
        return None
    else:
        timestamps = timestamps.sort_index()
        latest = timestamps.iloc[len(timestamps) - 1]
        path = f'{folder}/{name}{latest}.xlsx'
        log(f'loading "{path}"', level='info', source='df.load()')
        if return_date is True:
            df = pd.read_excel(path, sheet_name=sheet, index_col=index, **kwargs), latest
        else:
            df = pd.read_excel(path, sheet_name=sheet, index_col=index, **kwargs)
        
        if '#' in df.index:
            df.loc['#', :] = df.loc['#', :].apply(lambda x: qp_na(x, errors='ignore', na=''))
        if '#' in df.columns:
            df.loc[:, '#'] = df.loc[:, '#'].apply(lambda x: qp_na(x, errors='ignore', na=''))

        return df


