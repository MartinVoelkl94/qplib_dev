
import numpy as np
import pandas as pd
import re
import os
import shutil
import datetime
import random
import string as str_module
from IPython import get_ipython
from IPython.display import display
from .types import (
    _num,
    _na,
    _nk,
    _date,
    _datetime,
    )


TYPES_ITERABLE = (
    list,
    tuple,
    set,
    pd.Index,
    pd.Series,
    )

GREEN = '#6dae51'
RED = '#f73434'

GREY_LIGHT = '#d3d3d3'
BLUE_LIGHT = '#87ceeb'
GREEN_LIGHT = '#c0e7b0'
ORANGE_LIGHT = '#f7d67c'
RED_LIGHT = '#f7746a'


logs = []
def log(
        text=None,
        context='',
        verbosity=None,
        clear=False,
        ):
    """
    A very basic "logger" meant to be used in place
    of print() statements in jupyter notebooks.
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
    qplib.util.logs  #location of list containing log entries

    """
    if verbosity == 0:
        return

    time = datetime.datetime.now()
    global logs

    if clear:
        logs.clear()
        if verbosity in (None, 3, 4, 5):
            print('cleared all logs in qp.util.logs.')
        return

    if text is None:
        return pd.DataFrame(logs)


    levels = {
        'TRACE': 5,
        'DEBUG': 4,
        'INFO': 3,
        'WARNING': 2,
        'ERROR': 1,
        }
    colors = {
        'TRACE': GREY_LIGHT,
        'DEBUG': BLUE_LIGHT,
        'INFO': GREEN_LIGHT,
        'WARNING': ORANGE_LIGHT,
        'ERROR': RED_LIGHT,
        }

    color = GREEN_LIGHT
    level = 'INFO'
    level_int = 3
    text_temp = text.upper()

    #detect logging level
    for level_temp in levels.keys():
        if text_temp.startswith(level_temp):
            level = level_temp
            level_int = levels[level]
            color = colors[level]
            text = text[len(level_temp):].strip()
            if text and text[0] in [':', '-', ' ']:
                text = text[1:].strip()
            break

    if verbosity is None:
        verbosity = level_int

    if len(logs) == 0:
        delta_ms = 0.0
    else:
        delta_ms = datetime.datetime.now() - logs[-1]['time']
        delta_ms = delta_ms.total_seconds() * 1000
    message = {
        'level': level,
        'text': text,
        'context': context,
        'time': time,
        'delta_ms': delta_ms,
        }

    if level_int <= verbosity:
        logs.append(message)

        #for jupyter
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':  #pragma: no cover

            #make html friendly
            message['text'] = (
                message['text']
                .replace('\n', '<br>')
                .replace('\t', '&emsp;')
                )
            message['context'] = (
                message['context']
                .replace('\n', '<br>')
                .replace('\t', '&emsp;')
                )
            message_df = pd.DataFrame(message, index=[len(logs)])

            display(
                message_df
                .style
                .hide(axis=1)
                .apply(lambda x: [f'background-color: {color}' for i in x], axis=1)
                .set_properties(**{'text-align': 'left'})
                )

        #everywhere else
        else:
            context_formatted = context.replace('\n', '\n    ')
            text_formatted = text.replace('\n', '\n    ')
            string = (
                f'{level} log message:\n'
                f'  time: {time}\n'
                f'  time since last log: {delta_ms:.2f} ms\n'
                f'  context:{context_formatted}\n'
                f'  text:\n    """{text_formatted}"""\n'
                )
            print(string)



def _arg_to_list(arg):
    """
    converts a non interable argument to a list
    """
    if arg is None:
        return []
    elif isinstance(arg, TYPES_ITERABLE):
        return list(arg)
    else:
        return [arg]


def fetch(path, before='now', verbosity=3):
    """
    returns the path to the most recent version of a file
    assuming that a date is part of the filename.

    "before" defines recency of the file:
    - now: most recent version
    - today: most recent version before today
    - this day: most recent version before today
    - this week: ...
    - this month: ...
    - this year: ...
    - '2024_01_01': most recent version before 2024_01_01 (accepts many date formats)
    """

    if os.path.isfile(path):
        log(f'info: found file "{path}"', 'qp.fetch()', verbosity)
        return path

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
        cutoff = today - datetime.timedelta(days=today.day - 1)
    elif before == 'this year':
        cutoff = pd.to_datetime(f'{today.year}0101').date()
    else:
        cutoff = _date(before)


    name = os.path.basename(path)
    folder = os.path.dirname(path)
    extension = ''


    if folder == '':
        folder = os.getcwd()

    timestamps = pd.Series([])
    for file in os.listdir(folder):
        #check if file starts with name and is a file
        if os.path.isfile(f'{folder}/{file}') and file.startswith(name):
            try:
                timestamp_str_full = file.split(name)[-1]
                extension = '.' + timestamp_str_full.split('.')[-1]
                timestamp_str = timestamp_str_full.replace(f'{extension}', '')
                timestamp = _datetime(timestamp_str)
                if timestamp < _datetime(cutoff):
                    timestamps[timestamp] = (timestamp_str, extension)
            except Exception:  #pragma: no cover
                pass
    if len(timestamps) == 0:
        text = (
            'error: no timestamped files starting with'
            f' "{name}" found in "{folder}" before {cutoff}'
            )
        log(text, 'qp.fetch()', verbosity)
        raise FileNotFoundError(text)
    else:
        timestamps = timestamps.sort_index()
        latest = timestamps.iloc[len(timestamps) - 1][0]
        extension = timestamps.iloc[len(timestamps) - 1][1]
        path = f'{folder}/{name}{latest}{extension}'
        log(f'info: found file "{path}"', 'qp.fetch()', verbosity)
        return path


def match(patterns, value, regex=True):

    patterns = _arg_to_list(patterns)

    if regex and isinstance(value, str):
        for pattern in patterns:
            if re.fullmatch(pattern, value):
                return True
        return False
    else:
        return value in patterns


def ensure_unique_string(
        string,
        taken,
        strategy='increment',
        ) -> str:
    """
    Ensure that a string is unique within a set of taken strings.

    Parameters
    ----------
    string : The original string to be made unique.
    taken : An iterable of strings that are already taken.
    strategy : The strategy to use for making the string unique. Default is 'increment'.
        * "increment": use incrementing numbers appended to the string.
        * "random": use random characters appended to the string.
        * "timestamp": use a timestamp appended to the string.
        * "datestamp": append the current date to the string.
            If date is already taken, incrementing numbers are appended.

    """

    base_string = string

    if strategy == 'increment':
        counter = 1
        while string in taken:
            string = f"{base_string}{counter}"
            counter += 1
    elif strategy == 'random':
        while string in taken:
            chars = str_module.ascii_letters + str_module.digits
            rand_str = ''.join(random.choices(chars, k=6))
            string = f"{base_string}_{rand_str}"
    elif strategy == 'timestamp':
        while string in taken:
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%Hh%Mm%Ss')
            string = f"{base_string}_{timestamp}"
    elif strategy == 'datestamp':
        datestamp = datetime.datetime.now().strftime('%Y_%m_%d')
        string_temp = f"{base_string}_{datestamp}"
        counter = 1
        while string_temp in taken:
            string_temp = f"{base_string}_{datestamp}_v{counter}"
            counter += 1
        string = string_temp
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    return string


def header(
        word='header',
        slim=True,
        width=None,
        filler=' ',
        ):  #pragma: no cover
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
        filler = ' ' * int(((len(border) - len(word) - 2) / 2))
        text = border + '\n'\
            + '#' + filler + word + filler + '#' + '\n'\
            + border

    print(text)


def now(fmt='%Y_%m_%d'):  #pragma: no cover
    """
    alias for datetime.datetime.now().strftime(format_str)

    common format_str options:
    '%Y_%m_%d_%Hh%Mm%Ss': standard
    %Y_%b_%d:  3 letter month
    """
    return datetime.datetime.now().strftime(fmt)



#"bashlike" wrappers and aliases

def ls(
        path_or_object='',
        out='df',
        recursive=False,
        verbosity=3,
        ):
    """
    when path is passed: list files and folders in path.
    when python object is passed: list contents of the object.
    """
    if isinstance(path_or_object, str):
        return _list_files(path_or_object, out, recursive)
    else:
        if out != 'df':
            text = (
                'warning: qp.ls() always returns dataframes'
                f' when used on python objects, ignoring out={out}'
                )
            log(text, f'qp.ls({path_or_object=}, {out=}, {recursive=})', verbosity)
        layers = []
        result = pd.DataFrame()
        result = _ls_object(path_or_object, recursive, result, layers)
        return result


def lsr(
        path_or_object='',
        out='df',
        recursive=True,
        ):
    """
    alias for ls() with recursive=True
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
        files['size'] = files['_path'].apply(
            lambda x: (
                os.path.getsize(x)
                if os.path.isfile(x)
                else None
                )
            )
        files['created'] = files['_path'].apply(
            lambda x: (
                datetime
                .datetime
                .fromtimestamp(os.path.getctime(x))
                .strftime('%Y-%m-%d %H:%M:%S')
                )
            )
        files['last modified'] = files['_path'].apply(
            lambda x: (
                datetime
                .datetime
                .fromtimestamp(os.path.getmtime(x))
                .strftime('%Y-%m-%d %H:%M:%S')
                )
            )
        files['last accessed'] = files['_path'].apply(
            lambda x: (
                datetime
                .datetime
                .fromtimestamp(os.path.getatime(x))
                .strftime('%Y-%m-%d %H:%M:%S')
                )
            )
        files['permissions'] = files['_path'].apply(
            lambda x: (
                oct(os.stat(x).st_mode)[-3:]
                if os.path.isfile(x)
                else None
                )
            )
        files['path'] = files['_path']
        files['folder'] = files['_path'].apply(lambda x: os.path.dirname(x))
        files['type'] = files['_path'].apply(
            lambda x: (
                'dir'
                if os.path.isdir(x)
                else os.path.splitext(x)[1]
                )
            )

        files.drop(columns='_path', inplace=True)

        return files


def _ls_object(py_object, recursive, result, layers):
    """
    used by ls() to check the type of python py_objects and open them
    """
    ind = len(result)

    for layer, name in enumerate(layers):
        if f'layer{layer + 1}' not in result.columns:
            result.insert(len(layers) - 1, f'layer{layer + 1}', '')
        result.loc[ind, f'layer{layer + 1}'] = name


    if isinstance(py_object, int):
        result.loc[ind, 'type'] = 'int'
        result.loc[ind, 'value'] = py_object
        result.loc[ind, 'size'] = f'{len(str(py_object))}'
    elif isinstance(py_object, float):
        result.loc[ind, 'type'] = 'float'
        result.loc[ind, 'value'] = py_object
        result.loc[ind, 'size'] = f'{len(str(py_object)) - 1}'
    elif isinstance(py_object, str):
        result.loc[ind, 'type'] = 'str'
        result.loc[ind, 'value'] = py_object
        result.loc[ind, 'size'] = f'{len(py_object)}'
    elif isinstance(py_object, list):
        result.loc[ind, 'type'] = 'list'
        result.loc[ind, 'value'] = None
        result.loc[ind, 'size'] = f'{len(py_object)} elements'
    elif isinstance(py_object, tuple):
        result.loc[ind, 'type'] = 'tuple'
        result.loc[ind, 'value'] = None
        result.loc[ind, 'size'] = f'{len(py_object)} elements'
    elif isinstance(py_object, set):
        result.loc[ind, 'type'] = 'set'
        result.loc[ind, 'value'] = None
        result.loc[ind, 'size'] = f'{len(py_object)} elements'


    elif isinstance(py_object, dict):
        result.loc[ind, 'type'] = 'dict'
        result.loc[ind, 'value'] = None
        result.loc[ind, 'size'] = f'{len(py_object)} key,val pairs'
        if recursive is True:
            for key, value in py_object.items():
                result = _ls_object(value, recursive, result, layers + [f'["{key}"]'])

    elif isinstance(py_object, pd.core.series.Series):
        result.loc[ind, 'type'] = 'series'
        result.loc[ind, 'value'] = None
        result.loc[ind, 'size'] = f'{len(py_object)} rows'

    elif isinstance(py_object, np.ndarray):
        result.loc[ind, 'type'] = 'ndarray'
        result.loc[ind, 'value'] = None
        result.loc[ind, 'size'] = f'{py_object.shape}'

    elif isinstance(py_object, pd.core.frame.DataFrame):
        df = py_object
        if len(layers) > 0:
            result.loc[ind, 'type'] = 'df'
            result.loc[ind, 'size'] = f'{len(df)} rows, {len(df.columns)} cols'
        else:
            result.loc[ind, 'contents'] = f'{len(df)} rows, {len(df.columns)} cols'
            result.loc[ind, 'na'] = (
                df
                .applymap(lambda x: _na(x, errors=0, na=1))
                .sum()
                .sum()
                )
            result.loc[ind, 'nk'] = (
                df
                .applymap(lambda x: _nk(x, errors=0, nk=1))
                .sum()
                .sum()
                )

            ind1 = ind
            for col in df.columns:
                ind1 += 1
                result.loc[ind1, 'contents'] = col
                result.loc[ind1, 'na'] = (
                    df[col]
                    .apply(lambda x: _na(x, errors=0, na=1))
                    .sum()
                    )
                result.loc[ind1, 'nk'] = (
                    df[col]
                    .apply(lambda x: _nk(x, errors=0, nk=1))
                    .sum()
                    )
                result.loc[ind1, 'min'] = (
                    df[col]
                    .apply(lambda x: _num(x, errors=None))
                    .min()
                    )
                result.loc[ind1, 'max'] = (
                    df[col]
                    .apply(lambda x: _num(x, errors=None))
                    .max()
                    )
                result.loc[ind1, 'median'] = (
                    df[col]
                    .apply(lambda x: _num(x, errors=None))
                    .median()
                    )
                result.loc[ind1, 'mean'] = (
                    df[col]
                    .apply(lambda x: _num(x, errors=None))
                    .mean()
                    )

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
    text = f'info: moved from<br>{dir_old}<br>to<br>{dir_new}'
    log(text, f'qp.cd("{path}")', verbosity)
    return



def cp(src, dest, verbosity=3):
    """
    copy file or directory
    """

    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(src))

    if os.path.exists(dest):
        text = f'warning: "{dest}" already exists and will be overwritten'
        log(text, 'qp.cp()', verbosity)

    if os.path.isdir(src):
        shutil.copytree(src, dest)
    else:
        shutil.copy(src, dest)

    log(f'info: copied<br>{src}<br>to<br>{dest}', 'qp.cp()', verbosity)
    return



def mv(src, dest, verbosity=3):
    """
    move file or directory
    """

    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.basename(src))

    if os.path.exists(dest):
        text = f'warning: "{dest}" already exists and will be overwritten'
        log(text, 'qp.mv()', verbosity)

    shutil.move(src, dest)

    text = f'info: moved<br>"{src}"<br>to<br>"{dest}"'
    log(text, 'qp.mv()', verbosity)
    return



def mkdir(name, verbosity=3):
    """
    create directory
    """
    if os.path.isdir(name):
        text = f'info: directory "{name}" already exists'
        log(text, f'qp.mkdir("{name}")', verbosity)
    else:
        os.mkdir(name)
        text = f'info: created directory "{name}"'
        log(text, f'qp.mkdir("{name}")', verbosity)
    return



def isdir(name):
    """
    check if directory exists
    """
    return os.path.isdir(name)



def isfile(name):
    """
    check if file exists
    """
    return os.path.isfile(name)



def ispath(name):
    """
    check if path exists
    """
    return os.path.exists(name)
