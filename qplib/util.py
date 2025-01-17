
import numpy as np
import pandas as pd
import copy
import os
import sys
import shutil
import datetime
from IPython.display import display
from IPython import get_ipython
from .types import _num
from .types import _na
from .types import _nk


GREEN = '#6dae51'
ORANGE = 'orange'
RED = '#f73434'

GREY_LIGHT = 'lightgrey'
BLUE_LIGHT = 'skyblue'
GREEN_LIGHT = '#c0e7b0'
ORANGE_LIGHT = '#f7d67c'
RED_LIGHT = '#f7746a'


logs = pd.DataFrame(columns=['level', 'text', 'context', 'time'])
def log(text=None, context='', verbosity=None, clear=False):
    """
    A very basic "logger" meant to be used in place of print() statements in jupyter notebooks. 
    For more extensive logging purposes use a logging module.
    

    examples:

    from qplib import log

    log('trace: this is a trace entry which will be highlighted grey')
    log('debug: this is a debug entry which will be highlighted blue')
    log('info: this is a info entry which will be highlighted green')
    log('warning: this is a warning entry which will be highlighted orange')
    log('error: this is a error entry which will be highlighted red')

    log(clear=True)  #clear all log entries
    log()  #return dataframe of log entries
    qplib.util.logs  #location of dataframe containing log entries

    """

    time = pd.Timestamp.now()
    global logs
    idx = len(logs)
    
    if clear:
        logs = pd.DataFrame(columns=['text', 'context', 'level', 'time'])
        if verbosity in (None, 3, 4, 5):
            print('cleared all logs in qp.util.logs.')
        return
    
    if text is None:
        return logs

    
    levels = {'trace': 5, 'debug': 4, 'info': 3, 'warning': 2, 'error': 1}
    level = 'info'
    level_int = 3

    for level_temp in levels.keys():
        if text.lower().startswith(level_temp):
            level = level_temp
            level_int = levels[level]
            text = text[len(level_temp):].strip()
            if text[0] in [':', '-', ' ']:
                text = text[1:].strip()

    if verbosity is None:
        verbosity = level_int
    elif verbosity < 1:
        return
    

    if level == 'trace':
        logs.loc[idx, 'level'] = 'trace'
        color = GREY_LIGHT
    elif level == 'debug':
        logs.loc[idx, 'level'] = 'debug'
        color = BLUE_LIGHT
    elif level == 'info':
        logs.loc[idx, 'level'] = 'info'
        color = GREEN_LIGHT
    elif level == 'warning':
        logs.loc[idx, 'level'] = 'Warning'
        color = ORANGE_LIGHT
    elif level == 'error':
        logs.loc[idx, 'level'] = 'ERROR'
        color = RED_LIGHT

    logs.loc[idx, 'text'] = text
    logs.loc[idx, 'context'] = context
    logs.loc[idx, 'time'] = time

    if level_int <= verbosity:
        last = logs.tail(1)

        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            last.loc[:, 'text'] = last.loc[:, 'text'].str.replace('\n', '<br>')
            last.loc[:, 'text'] = last.loc[:, 'text'].str.replace('\t', '&emsp;')
            last.loc[:, 'context'] = last.loc[:, 'context'].str.replace('\n', '<br>')
            last.loc[:, 'context'] = last.loc[:, 'context'].str.replace('\t', '&emsp;')

            display(last.style.hide(axis=1).apply(
                    lambda x: [f'background-color: {color}' for i in x], axis=1
                    ).set_properties(**{'text-align': 'left'})
                )
        else:
            print(last)



class Args:
    def __init__(self, verbosity=3, overwrite=True, **kwargs):
        self.verbosity = verbosity
        self.overwrite = overwrite
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __setattr__(self, name, value):
        if hasattr(self, name):
            if self.overwrite:
                log(f'warning: {name} already exists and will be overwritten',
                    'qp.util.Args.__setattr__', self.verbosity)
                super().__setattr__(name, value)
            else:
                log(f'error: {name} already exists and cannot be overwritten',
                    'qp.util.Args.__setattr__', self.verbosity)
        else:
            super().__setattr__(name, value)



def header(word='header', slim=True, width=None, filler=' '):
    """
    Creates text headers for code sections or plain text.

    slim header:
    ######################     Header     ######################

    normal header:
    ##########################################
    #                 Header                 #
    ##########################################
    """

    if slim is True:
        if width is None:
            width = 60
        
        if len(word) > width - 10:
            width = len(word) + 10
        if len(word) % 2 == 1:
            width += 1

        border = int((width - len(word) - 10) / 2)
        text = '#' * border + '     ' + word + '     ' + '#' * border
    
    else:
        if width is None:
            width = 42

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
    alias for datetime.datetime.now().strftime(format_str)

    common format_str options:
    '%Y_%m_%d_%Hh%Mm%Ss': standard
    %Y_%b_%d:  3 letter month
    """
    return datetime.datetime.now().strftime(fmt)



#"bashlike" wrappers and aliases

def ls(path_or_object='', out='df', recursive=False, verbosity=3):
    """
    """

    if isinstance(path_or_object, str):
        return _list_files(path_or_object, out, recursive)
    else:
        if out != 'df':
            log(f'warning: qp.ls() always returns dataframes when used on python objects, ignoring out={out}',
                f'qp.ls({path_or_object=}, {out=}, {recursive=})', verbosity)
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
            result.loc[ind, 'na'] = df.applymap(lambda x: _na(x, errors=0, na=1)).sum().sum()
            result.loc[ind, 'nk'] = df.applymap(lambda x: _nk(x, errors=0, nk=1)).sum().sum()

            ind1 = ind
            rows = len(df.index)
            for col in df.columns:
                ind1 += 1
                result.loc[ind1, 'contents'] = col
                result.loc[ind1, 'na'] = df[col].apply(lambda x: _na(x, errors=0, na=1)).sum()
                result.loc[ind1, 'nk'] = df[col].apply(lambda x: _nk(x, errors=0, nk=1)).sum()
                result.loc[ind1, 'min'] = df[col].apply(lambda x: _num(x, errors=None)).min()
                result.loc[ind1, 'max'] = df[col].apply(lambda x: _num(x, errors=None)).max()
                result.loc[ind1, 'median'] = df[col].apply(lambda x: _num(x, errors=None)).median()
                result.loc[ind1, 'mean'] = df[col].apply(lambda x: _num(x, errors=None)).mean()

    return result.fillna('').replace(0, '')


def pwd():
    """
    print working directory
    """
    return os.getcwd()


def cd(path=None, verbosity=3):
    """
    change directory
    """

    if path in [None, '']:
        path = os.getcwd()
    elif path == '..':
        path = os.path.dirname(os.getcwd())

    dir_old = os.getcwd()
    if dir_old.endswith(path):
        log(f'info: already in {path}', f'qp.cd("{path}")', verbosity)
        return
    
    os.chdir(path)
    dir_new = os.getcwd()
    log(f'info: moved from<br>{dir_old}<br>to<br>{dir_new}', f'qp.cd("{path}")', verbosity)
    return


def cp(src, dest, verbosity=3):
    """
    copy file or directory
    """
    if os.path.isdir(src):
        shutil.copytree(src, dest)
    else:
        shutil.copy(src, dest)
    log(f'info: copied<br>{src}<br>to<br>{dest}', f'qp.cp({src=}, {dest=})', verbosity)
    return


def mkdir(name, verbosity=3):
    """
    create directory
    """
    if os.path.isdir(name):
        log(f'info: directory "{name}" already exists', f'qp.mkdir("{name}")', verbosity)
    else:
        os.mkdir(name)
        log(f'info: created directory "{name}"', f'qp.mkdir("{name}")', verbosity)
    return


def isdir(name, verbosity=3):
    """
    check if directory exists
    """
    if os.path.isdir(name):
        return True
    else:
        return False
  
    
def isfile(name, verbosity=3):
    """
    check if file exists
    """
    if os.path.isfile(name):
        return True
    else:
        return False


def ispath(name, verbosity=3):
    """
    check if path exists
    """
    if os.path.exists(name):
        return True
    else:
        return False









