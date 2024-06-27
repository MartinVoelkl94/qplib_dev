
import numpy as np
import pandas as pd
import copy
import os
import sys
import shutil
import datetime
from IPython.display import display
from .types import qp_num
from .types import qp_na
from .types import qp_nk



class qpDict(dict):
    """
    dictionary where key:value pairs and attributes are interchangeable.
    """

    def __init__(self, iterable={}):
        super().__init__((key,val) for key,val in iterable.items())
        for key, val in iterable.items():
            if key not in self.__dict__.keys():
                setattr(self, key, val)
            else:
                log('key is already used as dictionary attribute.', level='error', source='qpDict.__init__', input=key)

    def __setattr__(self, key, val):
        super().__setattr__(key, val)
        super().__setitem__(key, val)

    def __setitem__(self, key, val):
        super().__setitem__(key, val)
        super().__setattr__(key, val)

    def __delattr__(self, key):
        super().__delattr__(key)
        super().__delitem__(key)

    def __delitem__(self, key):
        super().__delitem__(key)
        super().__delattr__(key)

    def values_flat(self):
        values_flat = []
        for val in self.values():
            if isinstance(val, dict):
                values_flat.extend(val.values())
            elif isinstance(val, qpDict):
                values_flat.extend(val.values_flat())
            elif hasattr(val, '__iter__') and not isinstance(val, str):
                values_flat.extend(val)
            else:
                values_flat.append(val)
        return values_flat




def log(message=None, level='info', source='', input='', clear=False):
    """
    A very basic "logger".
    For more extensive logging purposes use a logging module.
    This is mostly meant to be used as a replacement for print() statements. 

    usage if qplib is imported as qp:
        qp.log('message', level='info') or qp.logs('message'): add info log entry
        qp.log('message', level='warning'): add warning log entry
        qp.log('message', level='error'): add error log entry
        qp.log(clear=True): clear all log entries
        qp.log(): return dataframe of log entries
        qp.util.logs: location of dataframe containing log entries
    
    """

    if 'logs' not in globals().keys():
        globals()['logs'] = pd.DataFrame(columns=['level', 'message', 'source', 'input', 'time'])
        print('created dataframe qp.util.logs for tracking log entries')
        print('use qp.log(message, level, source) or qp.log(message) to add log entries')
        print('logs are saved in qp.util.logs')

    if clear:
        globals()['logs'] = pd.DataFrame(columns=['level', 'message', 'source', 'input', 'time'])
        print('cleared all logs in qp.util.logs.')
        return
    
    if message is None:
        return globals()['logs']
    
    idx = len(globals()['logs'])

    match level.lower():
        case 'debug':
            globals()['logs'].loc[idx, 'level'] = 'debug'
        case 'info':
            globals()['logs'].loc[idx, 'level'] = 'info'
        case 'warning':
            globals()['logs'].loc[idx, 'level'] = 'Warning'
        case 'error':
            globals()['logs'].loc[idx, 'level'] = 'ERROR'

    globals()['logs'].loc[idx, 'message'] = message
    globals()['logs'].loc[idx, 'source'] = source
    globals()['logs'].loc[idx, 'input'] = input
    globals()['logs'].loc[idx, 'time'] = pd.Timestamp.now()

    if level.lower() != 'debug':
        # df2.style.hide(axis=1)
        display(globals()['logs'].tail(1).style.hide(axis=1).apply(
            lambda x: [
                'background-color: orange' if x['level'] == 'Warning'
                else 'background-color: red' if x['level'] == 'ERROR'
                else 'background-color: #59A859'
                for i in x],
            axis=1))


def header(word='header', width=42, filler=' '):
    """
    Prints a header with a word in the middle and a border around it.
    """
    
    if len(word) > width - 2:
        width = len(word) + 2
    if len(word) % 2 == 1:
        width += 1
    
    border = '#' * width
    filler = ' ' * int(((len(border) - len(word) - 2)/2))

    text = border + '\n'\
        + '#' + filler + word + filler + '#' + '\n'\
        + border
    
    print(text)


def now(fmt='%Y_%m_%d'):
    """
    wrapper for the datetime.datetime.now() function
    common formats:
    '%Y_%m_%d_%Hh%Mm%Ss': standard
    %Y_%b_%d:  3 letter month
    """
    return datetime.datetime.now().strftime(fmt)



#"bashlike" wrappers and aliases

def ls(path_or_object='', out='df', recursive=False):
    """
    """

    if isinstance(path_or_object, str):
        return _list_files(path_or_object, out, recursive)
    else:
        if out != 'df':
            log(f'qp.ls() always returns dataframes when used on python objects, ignoring out={out}')
        layers = []
        result = pd.DataFrame()
        result = _ls_object(path_or_object, recursive, result, layers)
        return result#.fillna('')

def lsr(path_or_object='', out='df', recursive=True):
    """
    """
    return ls(path_or_object, out, recursive)

def _list_files(path, out, recursive):
    """
    list files and folders
    """

    if path == '':
        path = os.getcwd()

    if recursive is True:
        filepaths = []
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                filepaths.append(f'{root}\\{filename}')
    else:
        filepaths = [f'{path}\\{filename}' for filename in os.listdir(path)]

    if out == 'list':
        return filepaths
    elif out == 'series':
        return pd.Series(filepaths)
    elif out == 'df':
        files = pd.DataFrame()

        files['_path'] = filepaths
        files['name'] = files['_path'].apply(lambda x: os.path.basename(x))
        files['size'] = files['_path'].apply(lambda x: os.path.getsize(x) if os.path.isfile(x) else None)
        files['created'] = files['_path'].apply(
            lambda x: datetime.datetime.fromtimestamp(os.path.getctime(x)).strftime('%Y-%m-%d %H:%M:%S'))
        files['last modified'] = files['_path'].apply(
            lambda x: datetime.datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S'))
        files['last accessed'] = files['_path'].apply(
            lambda x: datetime.datetime.fromtimestamp(os.path.getatime(x)).strftime('%Y-%m-%d %H:%M:%S'))
        files['permissions'] = files['_path'].apply(lambda x: oct(os.stat(x).st_mode)[-3:] if os.path.isfile(x) else None)
        files['path'] = files['_path']
        files['folder'] = files['_path'].apply(lambda x: os.path.dirname(x))
        files['type'] = files['_path'].apply(lambda x: 'dir' if os.path.isdir(x) else os.path.splitext(x)[1])

        files.drop(columns='_path', inplace=True)
        # display(files)

        return files

def _ls_object(py_object, recursive, result, layers):
    """
    used by ls() to check the type of python py_objects and open them
    """
    ind = len(result)

    for layer,name in enumerate(layers):
        if f'layer{layer+1}' not in result.columns:
            result.insert(len(layers)-1, f'layer{layer+1}', '')
        result.loc[ind, f'layer{layer+1}'] = name
    


    if isinstance(py_object, int):
        result.loc[ind, 'type'] = 'int'
        result.loc[ind, 'value'] = py_object
        result.loc[ind, 'size'] = f'{len(str(py_object))}'
    elif isinstance(py_object, float):
        result.loc[ind, 'type'] = 'float'
        result.loc[ind, 'value'] = py_object
        result.loc[ind, 'size'] = f'{len(str(py_object))-1}'
    elif isinstance(py_object, str):
        result.loc[ind, 'type'] ='str'
        result.loc[ind, 'value'] = py_object
        result.loc[ind, 'size'] = f'{len(py_object)}'
    elif isinstance(py_object, list):
        result.loc[ind, 'type'] = 'list'
        result.loc[ind, 'value'] = None
        result.loc[ind,'size'] = f'{len(py_object)} elements'
    elif isinstance(py_object, tuple):
        result.loc[ind, 'type'] = 'tuple'
        result.loc[ind, 'value'] = None
        result.loc[ind,'size'] = f'{len(py_object)} elements'
    elif isinstance(py_object, set):
        result.loc[ind, 'type'] ='set'
        result.loc[ind, 'value'] = None
        result.loc[ind,'size'] = f'{len(py_object)} elements'


    elif isinstance(py_object, dict):
        result.loc[ind, 'type'] ='dict'
        result.loc[ind, 'value'] = None
        result.loc[ind,'size'] = f'{len(py_object)} key,val pairs'
        if recursive is True:
            for key, value in py_object.items():
                result = _ls_object(value, recursive, result, layers + [f'["{key}"]'])

    elif isinstance(py_object, pd.core.series.Series):
        result.loc[ind, 'type'] ='series'
        result.loc[ind, 'value'] = None
        result.loc[ind,'size'] = f'{len(py_object)} rows'

    elif isinstance(py_object, np.ndarray):
        result.loc[ind, 'type'] = 'ndarray'
        result.loc[ind, 'value'] = None
        result.loc[ind,'size'] = f'{py_object.shape}'

    elif isinstance(py_object, pd.core.frame.DataFrame):
        df = py_object
        if len(layers) > 0:
            result.loc[ind, 'type'] = 'df'
            result.loc[ind,'size'] = f'{len(df)} rows, {len(df.columns)} cols'
        else:
            result.loc[ind, 'contents'] = f'{len(df)} rows, {len(df.columns)} cols'
            result.loc[ind, 'na'] = df.applymap(lambda x: qp_na(x, errors=0, na=1)).sum().sum()
            result.loc[ind, 'nk'] = df.applymap(lambda x: qp_nk(x, errors=0, nk=1)).sum().sum()

            ind1 = ind
            rows = len(df.index)
            for col in df.columns:
                ind1 += 1
                result.loc[ind1, 'contents'] = col
                result.loc[ind1, 'na'] = df[col].apply(lambda x: qp_na(x, errors=0, na=1)).sum()
                result.loc[ind1, 'nk'] = df[col].apply(lambda x: qp_nk(x, errors=0, nk=1)).sum()
                result.loc[ind1, 'min'] = df[col].apply(lambda x: qp_num(x, errors=None)).min()
                result.loc[ind1, 'max'] = df[col].apply(lambda x: qp_num(x, errors=None)).max()
                result.loc[ind1, 'median'] = df[col].apply(lambda x: qp_num(x, errors=None)).median()
                result.loc[ind1, 'mean'] = df[col].apply(lambda x: qp_num(x, errors=None)).mean()

    return result.fillna('').replace(0, '')


def pwd():
    """
    print working directory
    """
    return os.getcwd()


def cd(path=None):
    """
    change directory
    """

    if path in [None, '']:
        path = os.getcwd()
    elif path == '..':
        path = os.path.dirname(os.getcwd())

    dir_old = os.getcwd()
    if dir_old.endswith(path):
        log(f'already in {path}', level='info', source=f'qp.cd("{path}")')
        return
    
    os.chdir(path)
    dir_new = os.getcwd()
    log(f'moved from<br>{dir_old}<br>to<br>{dir_new}', level='info', source=f'qp.cd("{path}")')
    return


def cp(src, dest):
    """
    copy file or directory
    """
    if os.path.isdir(src):
        shutil.copytree(src, dest)
    else:
        shutil.copy(src, dest)
    log(f'copied<br>{src}<br>to<br>{dest}', level='info', source=f'qp.cp(source="{src}", destination="{dest}")')
    return


def mkdir(name):
    """
    create directory
    """
    if os.path.isdir(name):
        log(f'directory "{name}" already exists', level='info', source=f'qp.mkdir("{name}")')
    else:
        os.mkdir(name)
        log(f'created directory "{name}"', level='info', source=f'qp.mkdir("{name}")')
    return


def isdir(name):
    """
    check if directory exists
    """
    if os.path.isdir(name):
        return True
    else:
        return False
  
    
def isfile(name):
    """
    check if file exists
    """
    if os.path.isfile(name):
        return True
    else:
        return False


def ispath(name):
    """
    check if path exists
    """
    if os.path.exists(name):
        return True
    else:
        return False









