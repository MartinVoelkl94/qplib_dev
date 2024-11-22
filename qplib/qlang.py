
import re
import numpy as np
import pandas as pd
import qplib as qp

from IPython.display import display
from ipywidgets import widgets, interactive_output, HBox, VBox, fixed, Layout

from .util import log
from .types import _int, _float, _num, _bool, _datetime, _date, _na, _nk, _yn, _type
from .pd_util import _diff

VERBOSITY = 3
DIFF = None
INPLACE = False


class _Symbol:
    """
    A Symbol used in the query languages syntax.
    """
    def __init__(self, symbol, name, description, unary=None, binary=None, **kwargs):
        self.symbol = symbol
        self.name = name
        self.description = description
        self.unary = unary
        self.binary = binary
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        repr = f'{self.name}\n\tsymbol: "{self.symbol}"\n\tdescription: "{self.description}"'
        if hasattr(self, 'connector'):
            repr += f'\n\tconnector: {self.connector.name}'
        if hasattr(self, 'scope'):
            repr += f'\n\tscope: {self.scope.name}'
        if hasattr(self, 'negation'):
            repr += f'\n\tnegation: {self.negation.name}'
        if hasattr(self, 'operator'):
            repr += f'\n\toperator: {self.operator.name}'
        if hasattr(self, 'value'):
            repr += f'\n\tvalue: "{self.value}"'
        return repr
 
    def __str__(self):
        return self.__repr__()


class _Symbols:
    """
    Multiple Symbols of the same category are collected in a Symbols object.
    """
    def __init__(self, name, *symbols):
        self.name = name
        self.by_name = {symbol.name: symbol for symbol in symbols}
        self.by_symbol = {symbol.symbol: symbol for symbol in symbols}
        for symbol in symbols:
            setattr(self, symbol.name, symbol)

    def __getitem__(self, key):
        if key in self.by_symbol:
            return self.by_symbol[key]
        elif key in self.by_name:
            return self.by_name[key]
        else:
            log(f'error: symbol "{key}" not found in "{self.name}"', 'qp.qlang.Symbols.__getitem__', VERBOSITY)
            return None

    def __iter__(self):
        return iter(self.by_name.values())

    def __repr__(self):
        return f'{self.name}:\n' + '\n\t'.join([str(val) for key,val in self.by_name.items()])
    
    def __str__(self):
        return f'{self.name}:\n' + '\n\t'.join([str(val) for key,val in self.by_name.items()])
    


COMMENT = _Symbol('#', 'COMMENT', 'comments out the rest of the line')
ESCAPE = _Symbol('`', 'ESCAPE', 'escape the next character')

_CONNECTORS = _Symbols('CONNECTORS',
    _Symbol('', 'RESET', 'only the current condition must be fulfilled'),
    _Symbol('&', 'AND', 'this condition and the previous condition/s must be fulfilled'),
    _Symbol('/', 'OR', 'this condition or the previous condition/s must be fulfilled'),
    )

_SCOPES = _Symbols('SCOPES',
    _Symbol('any', 'ANY', 'select whole row if any value in the selected columns fulfills the condition'),
    _Symbol('all', 'ALL', 'select whole row if all values in the selected columns fulfill the condition'),
    _Symbol('idx', 'IDX', 'select whole row if the index of the row fulfills the condition'),
    _Symbol('val', 'VAL', 'select only values (not the whole row) that fulfill the condition'),
    )


_NEGATIONS = _Symbols('NEGATIONS',
    _Symbol('', 'FALSE', 'dont negate the condition'),
    _Symbol('!', 'TRUE', 'negate the condition'),
    )

_OPERATORS = _Symbols('OPERATORS',
    #for changing settings
    _Symbol('verbosity=', 'SET_VERBOSITY', 'change the verbosity/logging level'),
    _Symbol('diff=', 'SET_DIFF', 'change if and how the difference between the old and new dataframe is shown'),


    #for filtering
    _Symbol('>=', 'BIGGER_EQUAL', 'bigger or equal', binary=True),
    _Symbol('<=', 'SMALLER_EQUAL', 'smaller or equal', binary=True),
    _Symbol('>', 'BIGGER', 'bigger', binary=True),
    _Symbol('<', 'SMALLER', 'smaller', binary=True),

    _Symbol('==', 'STRICT_EQUAL', 'equal to (case sensitive)', binary=True),
    _Symbol('=', 'EQUAL', 'equal to', binary=True),

    _Symbol('??', 'STRICT_CONTAINS', 'contains a string (case sensitive)', binary=True),
    _Symbol('?', 'CONTAINS', 'contains a string (not case sensitive)', binary=True),

    _Symbol('r=', 'MATCHES_REGEX', 'matches a regex', binary=True),
    _Symbol('r?', 'CONTAINS_REGEX', 'contains a regex', binary=True),

    _Symbol('~', 'EVAL', 'select rows/values by evaluating a python expression on each value', binary=True),
    _Symbol('col~', 'COL_EVAL', 'select rows/values by evaluating a python expression on a whole column', binary=True),

    _Symbol('load', 'LOAD_SELECTION', 'load a saved selection of rows/values (boolean mask). save using: "´m save <name>', binary=True),

    _Symbol('is any', 'IS_ANY', 'is any value (use to reset selection)', unary=True),
    _Symbol('is str', 'IS_STR', 'is a string', unary=True),
    _Symbol('is int', 'IS_INT', 'is an integer', unary=True),
    _Symbol('is float', 'IS_FLOAT', 'is a float', unary=True),
    _Symbol('is num', 'IS_NUM', 'is a number', unary=True),
    _Symbol('is bool', 'IS_BOOL', 'is a boolean', unary=True),
    _Symbol('is datetime', 'IS_DATETIME', 'is a datetime', unary=True),
    _Symbol('is date', 'IS_DATE', 'is a date', unary=True),
    _Symbol('is na', 'IS_NA', 'is a missing value', unary=True),
    _Symbol('is nk', 'IS_NK', 'is not a known value', unary=True),
    _Symbol('is yn', 'IS_YN', 'is a value representing yes or no', unary=True),
    _Symbol('is yes', 'IS_YES', 'is a value representing yes', unary=True),
    _Symbol('is no', 'IS_NO', 'is a value representing no', unary=True),
    _Symbol('is unique', 'IS_UNIQUE', 'is a unique value', unary=True),
    _Symbol('is first', 'IS_FIRST', 'is the first value (of multiple values)', unary=True),
    _Symbol('is last', 'IS_LAST', 'is the last value (of multiple values)', unary=True),


    #for modifying values and headers
    _Symbol('=', 'SET_VAL', 'replace value with the given string'),
    _Symbol('+=', 'ADD_VAL', 'append a string to the value (coerce to string if needed)'),

    _Symbol('~', 'SET_EVAL', 'replace value by evaluating a python expression for each selected value'),
    _Symbol('col~', 'SET_COL_EVAL', 'replace value by evaluating a python expression for each selected column'),

    _Symbol('r~', 'EXTRACT_REGEX', 'extract a value using a regex with capture group. returns the first capture group'),
    
    _Symbol('sort', 'SORT', 'sort values based on the selected column(s)', unary=True),
    _Symbol('!sort', 'SORT_REVERSE', 'sort values based on the selected column(s) in reverse order', unary=True),


    _Symbol('to str', 'TO_STR', 'convert to string', unary=True, func=str, dtype=str),
    _Symbol('to int', 'TO_INT', 'convert to integer', unary=True, func=_int, dtype='Int64'),
    _Symbol('to float', 'TO_FLOAT', 'convert to float', unary=True, func=_float, dtype='Float64'),
    _Symbol('to num', 'TO_NUM', 'convert to number', unary=True, func=_num, dtype='object'),
    _Symbol('to bool', 'TO_BOOL', 'convert to boolean', unary=True, func=_bool, dtype='bool'),
    _Symbol('to datetime', 'TO_DATETIME', 'convert to datetime', unary=True, func=_datetime, dtype='datetime64[ns]'),
    _Symbol('to date', 'TO_DATE', 'convert to date', unary=True, func=_date, dtype='datetime64[ns]'),
    _Symbol('to na', 'TO_NA', 'convert to missing value', unary=True, func=_na),
    _Symbol('to nk', 'TO_NK', 'convert to not known value', unary=True, func=_nk, dtype='object'),
    _Symbol('to yn', 'TO_YN', 'convert to yes or no value', unary=True, func=_yn, dtype='object'),


    #for adding new columns
    _Symbol('=', 'STR_COL', 'add new column, fill it with the given string and select it', binary=True),
    _Symbol('~', 'EVAL_COL', 'add new column, fill it by evaluating a python expression and select it', binary=True),
    

    #miscellaneous instructions
    _Symbol('=', 'SET_METADATA', 'set contents of the columnn named "meta" to the given string', binary=True),
    _Symbol('+=', 'ADD_METADATA', 'append the given string to the contents of the column named "meta"', binary=True),
    _Symbol('tag', 'TAG_METADATA', 'add a tag of the currently selected column(s) in the form of "\\n@<selected col>: <value>" to the column named "meta"', binary=True),
    _Symbol('~', 'SET_METADATA_EVAL', 'set contents of the column named "meta" by evaluating a python expression for each selected value in the metadata', binary=True),
    _Symbol('col~', 'SET_METADATA_COL_EVAL', 'set contents of the column named "meta" by evaluating a python expression on the whole metadata column', binary=True),
    _Symbol('save', 'SAVE_SELECTION', 'save current selection with given <name>. load using: "´r load <name>', binary=True),
    
    )



def _select_cols(instruction, df_new, masks, cols, diff, verbosity):
    """
    An Instruction to select columns fulfilling a condition.
    """

    connector = instruction.connector
    negation = instruction.negation
    operator = instruction.operator
    value = instruction.value

    cols_all = df_new.columns.to_series()

    cols_new = _filter_series(cols_all, negation, operator, value, verbosity, df_new)

    if cols_new.any() == False:
        log(f'warning: no columns fulfill the condition in "{instruction.str}"',
            'qp.qlang._select_cols', verbosity)

    cols = _update_selected_cols(cols, cols_new, connector, verbosity)

    if cols.any() == False and connector == _CONNECTORS.AND:
        log(f'warning: no columns fulfill the condition in "{instruction.str}" and the previous condition(s)',
            'qp.qlang._select_cols', verbosity)

    return df_new, masks, cols, diff, verbosity



def _select_rows(instruction, df_new, masks, cols, diff, verbosity):
    """
    An Instruction to select rows/values fulfilling a condition.
    """
    
    connector = instruction.connector
    scope = instruction.scope
    negation = instruction.negation
    operator = instruction.operator
    value = instruction.value
    rows_all = df_new.index.to_series()
    mask = pd.DataFrame(np.zeros(df_new.shape, dtype=bool), columns=df_new.columns, index=df_new.index)

    if cols.any() == False:
        log(f'warning: row filter cannot be applied when no columns where selected', 'qp.qlang._select_rows', verbosity)
        masks[0] = mask
        return df_new, masks, cols, diff, verbosity
 

    if value.startswith('@'):
        column = value[1:]
        if column in df_new.columns:
            value = df_new[column]
        else:
            log(f'error: column "{column}" not found in dataframe. cannot use "@{column}" as value for row selection',
                'qp.qlang._select_rows', verbosity)

    

    if scope == _SCOPES.IDX:
        rows = _filter_series(rows_all, negation, operator, value, verbosity, df_new)
        mask.loc[rows, :] = True
    elif operator == _OPERATORS.LOAD_SELECTION:
        if value in masks.keys():
            mask = masks[value]
        else:
            log(f'error: selection "{value}" not found in saved selections', 'qp.qlang._select_rows', verbosity)
            masks[0] = mask
    else: #corresponds to behaviour of _SCOPES.VAL
        for col in df_new.columns[cols]:
            rows = _filter_series(df_new[col], negation, operator, value, verbosity, df_new)
            mask.loc[rows, col] = True

    if scope == _SCOPES.ANY:
        rows = mask.any(axis=1)
        mask.loc[rows, :] = True  
    elif scope == _SCOPES.ALL:
        rows = mask.loc[:, cols].all(axis=1)
        mask.loc[rows, :] = True
        mask.loc[~rows, :] = False
    
    if connector == _CONNECTORS.RESET:
        masks[0] = mask
    elif connector == _CONNECTORS.AND:
        masks[0] = masks[0] & mask
    elif connector == _CONNECTORS.OR:
        masks[0] = masks[0] | mask

    return df_new, masks, cols, diff, verbosity



def _modify_vals(instruction, df_new, masks, cols, diff, verbosity):
    """
    An Instruction to modify the selected values.
    """

    if masks[0].any().any() == False or cols.any() == False:
        log(f'warning: value modification cannot be applied when no values where selected', 'qp.qlang._modify_vals', verbosity)
        return df_new, masks, cols, diff, verbosity


    mask_temp = masks[0].copy()
    mask_temp.loc[:, ~cols] = False

    operator = instruction.operator
    value = instruction.value
    type_conversions = [
        _OPERATORS.TO_STR, _OPERATORS.TO_INT, _OPERATORS.TO_FLOAT, _OPERATORS.TO_NUM, _OPERATORS.TO_BOOL,
        _OPERATORS.TO_DATETIME, _OPERATORS.TO_DATE,
        _OPERATORS.TO_NA, _OPERATORS.TO_NK, _OPERATORS.TO_YN,
        ]
    

    #data modification  
    if operator == _OPERATORS.SET_VAL:
        df_new[mask_temp] = value

    elif operator == _OPERATORS.ADD_VAL:
        df_new[mask_temp] = df_new[mask_temp].astype(str) + value

    elif operator == _OPERATORS.SET_COL_EVAL:
        rows = mask_temp.any(axis=1)
        changed = df_new.loc[rows, cols].apply(lambda x: eval(value, {'col': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}), axis=0)
        df_new = df_new.mask(mask_temp, changed)
    
    elif operator == _OPERATORS.EXTRACT_REGEX:
        rows = mask_temp.any(axis=1)
        for col in df_new.columns[cols]:
            df_new.loc[rows, col] = df_new.loc[rows, col].str.extract(value).loc[rows, 0]
    
    elif operator == _OPERATORS.SORT:
        df_new.sort_values(by=list(df_new.columns[cols]), axis=0, inplace=True)

    elif operator == _OPERATORS.SORT_REVERSE:
        df_new.sort_values(by=list(df_new.columns[cols]), axis=0, ascending=False, inplace=True)

    elif pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
        #data modification
        if operator == _OPERATORS.SET_EVAL:
            rows = mask_temp.any(axis=1)
            if 'x' in value:  #needs to be evaluated for each value
                changed = df_new.loc[rows, cols].map(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:  #only needs to be evaluated once
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                changed = df_new.loc[rows, cols].map(lambda x: eval_result)  #setting would be faster but map is dtype compatible
            df_new = df_new.mask(mask_temp, changed)

        #type conversion
        elif operator in type_conversions:
            rows = mask_temp.any(axis=1)
            changed = df_new.loc[rows, cols].map(lambda x: operator.func(x))
            df_new.loc[rows, cols] = changed
            if hasattr(operator, 'dtype'):
                for col in df_new.columns[cols]:
                    df_new[col] = df_new[col].astype(operator.dtype)

    else:
        #data modification
        if operator == _OPERATORS.SET_EVAL:
            rows = mask_temp.any(axis=1)
            if 'x' in value:  #needs to be evaluated for each value
                changed = df_new.loc[rows, cols].applymap(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:  #only needs to be evaluated once
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                changed = df_new.loc[rows, cols].applymap(lambda x: eval_result)  #setting would be faster but map is dtype compatible
            df_new = df_new.mask(mask_temp, changed)

        #type conversion
        elif operator in type_conversions:
            rows = mask_temp.any(axis=1)
            changed = df_new.loc[rows, cols].applymap(lambda x: operator.func(x))
            df_new.loc[rows, cols] = changed
            if hasattr(operator, 'dtype'):
                for col in df_new.columns[cols]:
                    df_new[col] = df_new[col].astype(operator.dtype)

    return df_new, masks, cols, diff, verbosity


def _modify_headers(instruction, df_new, masks, cols, diff, verbosity):
    """
    An Instruction to modify the headers of the selected column(s).
    """

    if cols.any() == False:
        log(f'warning: header modification cannot be applied when no columns where selected', 'qp.qlang._modify_headers', verbosity)
        return df_new, masks, cols, diff, verbosity

    operator = instruction.operator
    value = instruction.value


    if operator == _OPERATORS.SET_VAL:
        df_new.rename(columns={col: value for col in df_new.columns[cols]}, inplace=True)
        cols.index = df_new.columns
        for mask in masks.values():
            mask.rename(columns={col: value for col in mask.columns[cols]}, inplace=True)


    if operator == _OPERATORS.ADD_VAL:
        df_new.rename(columns={col: col + value for col in df_new.columns[cols]}, inplace=True)
        cols.index = df_new.columns
        for mask in masks.values():
            mask.rename(columns={col: col + value for col in mask.columns[cols]}, inplace=True)

    if operator == _OPERATORS.SET_EVAL:
        df_new.rename(
            columns={
                col: eval(value, {'x': col, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
                for col in df_new.columns[cols]
                },
            inplace=True
            )
        cols.index = df_new.columns
        for mask in masks.values():
            mask.rename(
                columns={
                    col: eval(value, {'x': col, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
                    for col in mask.columns[cols]
                    },
                inplace=True
                )

    return df_new, masks, cols, diff, verbosity


def _new_col(instruction, df_new, masks, cols, diff, verbosity):
    """
    An Instruction to add a new column.
    """

    operator = instruction.operator
    value = instruction.value
    rows = masks[0].any(axis=1)

    if value.startswith('@'):
        column = value[1:]
        if column in df_new.columns:
            value = df_new[column]
        else:
            log(f'error: column "{column}" not found in dataframe. cannot add a new column thats a copy of it',
                'qp.qlang._new_col', verbosity)


    if operator == _OPERATORS.STR_COL:
        for i in range(1, 1001):
            if i == 1000:
                log(f'warning: could not add new column. too many columns named "new<x>"',
                    'qp.qlang._new_col', verbosity)
                break

            header = 'new' + str(i)
            if header not in df_new.columns:
                df_new[header] = ''
                masks[0][header] = rows
                if isinstance(value, pd.Series):
                    df_new.loc[rows, header] = value.astype(str)
                else:
                    df_new.loc[rows, header] = value
                cols = pd.Series([True if col == header else False for col in df_new.columns])
                cols.index = df_new.columns
                break
    
    elif operator == _OPERATORS.EVAL_COL:
        for i in range(1, 1001):
            if i == 1000:
                log(f'warning: could not add new column. too many columns named "new<x>"',
                    'qp.qlang._new_col', verbosity)
                break

            header = 'new' + str(i)
            if header not in df_new.columns:
                masks[0][header] = rows
                value = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
                if isinstance(value, pd.Series):
                    df_new[header] = value
                else:
                    df_new[header] = pd.NA
                    df_new.loc[rows, header] = value
                cols = pd.Series([True if col == header else False for col in df_new.columns])
                cols.index = df_new.columns
                break
    

    return df_new, masks, cols, diff, verbosity


def _miscellaneous(instruction, df_new, masks, cols, diff, verbosity):
    """
    An Instruction for miscellaneous tasks:
    - modifying metadata
    - saving selections
    """
    
    operator = instruction.operator
    value = instruction.value
    rows = masks[0].any(axis=1)
    
    operators_metadata = [
        _OPERATORS.SET_METADATA,
        _OPERATORS.ADD_METADATA,
        _OPERATORS.TAG_METADATA,
        _OPERATORS.SET_METADATA_EVAL,
        _OPERATORS.SET_METADATA_COL_EVAL,
        ]
    if operator in operators_metadata and 'meta' not in df_new.columns:
        log(f'info: no metadata column found in dataframe. creating new column named "meta"',
            'qp.qlang._miscellaneous', verbosity)
        df_new['meta'] = ''
        cols = pd.concat([cols, pd.Series([False])])
        cols.index = df_new.columns
    

    #modify metadata
    if operator == _OPERATORS.SET_METADATA:
        df_new.loc[rows, 'meta'] = value

    elif operator == _OPERATORS.ADD_METADATA:
        df_new.loc[rows, 'meta'] += value

    elif operator == _OPERATORS.TAG_METADATA:
        tag = ''
        for col in df_new.columns[cols]:
            tag += f'@{col}'
        df_new.loc[rows, 'meta'] += f'\n{tag}: {value}'

    elif operator == _OPERATORS.SET_METADATA_EVAL:
        if pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
            if 'x' in value:  #needs to be evaluated for each value
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].map(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:  #only needs to be evaluated once
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].map(lambda x: eval_result)
        else:
            if 'x' in value:
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].applymap(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].applymap(lambda x: eval_result)
    
    elif operator == _OPERATORS.SET_METADATA_COL_EVAL:
        df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].apply(lambda x: eval(value, {'col': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))

    elif operator == _OPERATORS.SAVE_SELECTION:
        if value in masks.keys():
            log(f'warning: a selection was already saved as "{value}". overwriting it',
                'qp.qlang._miscellaneous', verbosity)
        masks[value] = masks[0].copy()

    return df_new, masks, cols, diff, verbosity


def _modify_settings(instruction, df_new, masks, cols, diff, verbosity):
    """
    An instruction to change the query settings.
    """

    operator = instruction.operator
    value = instruction.value
    
    if operator == _OPERATORS.SET_VERBOSITY:
        if value in ['0', '1', '2', '3', '4', '5']:
            verbosity = int(value)
        else:
            log(f'warning: verbosity must be an integer between 0 and 5. "{value}" is not valid',
                'qp.qlang._modify_settings', verbosity)
    
    elif operator == _OPERATORS.SET_DIFF:
        if value.lower() in ['none', '0', 'false']:
            diff = None
        elif value.lower() in ['mix', 'new', 'old', 'new+']:
            diff = value.lower()
        else:
            log(f'warning: diff must be one of [None, mix, old, new, new+]. "{value}" is not valid',
                'qp.qlang._modify_settings', verbosity)

    return df_new, masks, cols, diff, verbosity


_INSTRUCTIONS = _Symbols('INSTRUCTIONS',
                       
    _Symbol('´c', 'SELECT_COLS', 'select columns fulfilling a condition',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        negations=[
            _NEGATIONS.FALSE, #default
            _NEGATIONS.TRUE,
            ],
        operators=[
            _OPERATORS.EQUAL, #default

            #binary
            _OPERATORS.BIGGER_EQUAL, _OPERATORS.SMALLER_EQUAL, _OPERATORS.BIGGER, _OPERATORS.SMALLER,
            _OPERATORS.STRICT_EQUAL, _OPERATORS.EQUAL,
            _OPERATORS.STRICT_CONTAINS, _OPERATORS.CONTAINS,
            _OPERATORS.MATCHES_REGEX, _OPERATORS.CONTAINS_REGEX,
            _OPERATORS.EVAL,
        
            #unary
            _OPERATORS.IS_ANY,
            _OPERATORS.IS_UNIQUE,
            _OPERATORS.IS_NA, _OPERATORS.IS_NK,
            _OPERATORS.IS_STR, _OPERATORS.IS_INT, _OPERATORS.IS_FLOAT, _OPERATORS.IS_NUM, _OPERATORS.IS_BOOL,
            _OPERATORS.IS_DATE, _OPERATORS.IS_DATETIME,
            _OPERATORS.IS_YN, _OPERATORS.IS_YES, _OPERATORS.IS_NO,
            ],
        copy_df=False,
        apply=_select_cols,
        ),


    _Symbol('´r', 'SELECT_ROWS', 'select rows/values fulfilling a condition',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        scopes=[
            _SCOPES.ANY, #default
            _SCOPES.ALL,
            _SCOPES.IDX,
            _SCOPES.VAL,
            ],
        negations=[
            _NEGATIONS.FALSE, #default
            _NEGATIONS.TRUE,
            ],
        operators=[
            _OPERATORS.EQUAL, #default

            #binary
            _OPERATORS.BIGGER_EQUAL, _OPERATORS.SMALLER_EQUAL, _OPERATORS.BIGGER, _OPERATORS.SMALLER,
            _OPERATORS.STRICT_EQUAL, _OPERATORS.EQUAL,
            _OPERATORS.STRICT_CONTAINS, _OPERATORS.CONTAINS,
            _OPERATORS.MATCHES_REGEX, _OPERATORS.CONTAINS_REGEX,
            _OPERATORS.EVAL, _OPERATORS.COL_EVAL,
            _OPERATORS.LOAD_SELECTION,
        
            #unary
            _OPERATORS.IS_ANY,
            _OPERATORS.IS_UNIQUE, _OPERATORS.IS_FIRST, _OPERATORS.IS_LAST,
            _OPERATORS.IS_NA, _OPERATORS.IS_NK,
            _OPERATORS.IS_STR, _OPERATORS.IS_INT, _OPERATORS.IS_FLOAT, _OPERATORS.IS_NUM, _OPERATORS.IS_BOOL,
            _OPERATORS.IS_DATETIME, _OPERATORS.IS_DATE,
            _OPERATORS.IS_YN, _OPERATORS.IS_YES, _OPERATORS.IS_NO,
            ],
        copy_df=False,
        apply=_select_rows,
        ),


    _Symbol('´v', 'MODIFY_VALS', 'modify the selected values',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        operators=[
            _OPERATORS.SET_VAL, #default
            _OPERATORS.ADD_VAL,
            _OPERATORS.SET_EVAL, _OPERATORS.SET_COL_EVAL,
            _OPERATORS.EXTRACT_REGEX,
            _OPERATORS.SORT, _OPERATORS.SORT_REVERSE,
            _OPERATORS.TO_STR, _OPERATORS.TO_INT, _OPERATORS.TO_FLOAT, _OPERATORS.TO_NUM, _OPERATORS.TO_BOOL,
            _OPERATORS.TO_DATETIME, _OPERATORS.TO_DATE, _OPERATORS.TO_NA, _OPERATORS.TO_NK, _OPERATORS.TO_YN,
            ],
        copy_df=True,
        apply=_modify_vals,
        ),

    _Symbol('´h', 'MODIFY_HEADERS', 'modify headers of the selected columns',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        operators=[
            _OPERATORS.SET_VAL, #default
            _OPERATORS.ADD_VAL,
            _OPERATORS.SET_EVAL,
            ],
        copy_df= True,
        apply=_modify_headers,
        ),

    _Symbol('´n', 'NEW_COL', 'add new column',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        operators=[
            _OPERATORS.STR_COL, #default
            _OPERATORS.EVAL_COL,
            ],
        copy_df= True,
        apply=_new_col,
        ),

    _Symbol('´m', 'MISCELLANEOUS', 'miscellaneous instructions',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        operators=[
            _OPERATORS.SET_METADATA, #default
            _OPERATORS.ADD_METADATA,
            _OPERATORS.TAG_METADATA,
            _OPERATORS.SET_METADATA_EVAL,
            _OPERATORS.SET_METADATA_COL_EVAL,
            _OPERATORS.SAVE_SELECTION,
            ],
        copy_df= True,
        apply=_miscellaneous,
        ),

    _Symbol('´s', 'MODIFY_SETTINGS', 'change query settings',
        connectors=[
            _CONNECTORS.RESET,#default
            _CONNECTORS.AND,
            _CONNECTORS.OR,
            ],
        operators=[
            _OPERATORS.SET_VERBOSITY, #default
            _OPERATORS.SET_DIFF,
            ],
        copy_df= False,
        apply=_modify_settings,
        ),
    )



def query(df_old, code=''):
    """
    Used by the dataframe accessors df.q() (DataFrameQuery) and df.qi() (DataFrameQueryInteractive).
    """

    #setup
    verbosity = VERBOSITY
    diff = DIFF
    _check_df(df_old, verbosity=verbosity)


    #parse and apply instructions

    lines, instruction_strs, copy_df = _tokenize_code(code, verbosity)
    if INPLACE:
        df_new = df_old
    elif copy_df:
        df_new = df_old.copy()
    else:
        df_new = df_old
    
    cols = pd.Series([True for col in df_new.columns])
    cols.index = df_new.columns
    mask = pd.DataFrame(np.ones(df_new.shape, dtype=bool), columns=df_new.columns, index=df_new.index)
    masks = {0: mask}  #the save instruction only adds str keys, therefore the default key is an int to avoid conflicts




    for instruction_str in instruction_strs:
        instruction = _parse_instruction(instruction_str, verbosity)
        df_new, masks, cols, diff, verbosity  = instruction.apply(instruction, df_new, masks, cols, diff, verbosity)


    #results

    rows = masks[0].any(axis=1)
    df_filtered = df_new.loc[rows, cols]

    if diff is None:
        return df_filtered 
    else:
        #show difference before and after filtering
        if 'meta' in df_old.columns and 'meta' not in df_filtered.columns:
            df_filtered.insert(0, 'meta', df_old.loc[rows, 'meta'])

        result = _diff(
            df_filtered, df_old,
            mode=diff,
            verbosity=verbosity
            )  
        return result



def _check_df(df, verbosity=3):
    """
    Checks dataframe for issues which could interfere with the query language used by df.q().
    df.q() uses '&', '/' and '´' for expression syntax.
    """
    problems_found = False

    if len(df.index) != len(df.index.unique()):
        log('error: index is not unique', 'qp.qlang._check_df', verbosity)
        problems_found = True

    if len(df.columns) != len(df.columns.unique()):
        log('error: columns are not unique', 'qp.qlang._check_df', verbosity)
        problems_found = True

    problems = {
        '"&"': [],
        '"/"': [],
        '"´"': [],
        'leading whitespace': [],
        'trailing whitespace': [],
        }

    for col in df.columns:
        if isinstance(col, str):
            if '&' in col:
                problems['"&"'].append(col)
            if '/' in col:
                problems['"/"'].append(col)
            if '´' in col:
                problems['"´"'].append(col)
            if col.startswith(' '):
                problems['leading whitespace'].append(col)
            if col.endswith(' '):
                problems['trailing whitespace'].append(col)

    for problem, cols in problems.items():
        if len(cols) > 0:
            log(f'warning: the following column headers contain {problem}, use a tick (`) to escape such characters: {cols}',
                'qp.qlang._check_df', verbosity)
            problems_found = True
    

    symbol_conflicts = []
    symbols = tuple(_SCOPES.by_symbol.keys()) + tuple(_NEGATIONS.by_symbol.keys()) + tuple(_OPERATORS.by_symbol.keys())
    symbols = tuple(symbol for symbol in symbols if symbol != '')

    for col in df.columns:
        if str(col).startswith(tuple(symbols)):
            symbol_conflicts.append(col)

    if len(symbol_conflicts) > 0:
        log(f'warning: the following column headers start with a character sequence that can be read as a query instruction symbol when the default instruction operator is inferred:\n{symbol_conflicts}\nexplicitely use a valid operator to avoid conflicts.',
            'qp.qlang._check_df', verbosity)
        problems_found = True


    if problems_found is False:
        log('info: df was checked. no problems found', 'qp.qlang._check_df', verbosity)



def _tokenize_code(code, verbosity):
    """
    Turns the plain text input string into a list of instruction strings for the instruction parser.
    """

    lines = []
    instructions_all = []
    copy_df = False

    #get lines and instruction blocks
    for line_num, line in enumerate(code.split('\n')):
        line = line.strip()
        lines.append([line_num, line])
        line = line.split(COMMENT.symbol)[0].strip()
        instructions = []
    
        if line == '':
            continue


        escape = False
        chars_in_instruction = 0
        instruction_type = _INSTRUCTIONS.SELECT_COLS.symbol  #default

        for i, char in enumerate(line):
            if escape:
                instructions[-1] += char
                chars_in_instruction += 1
                escape = False
                continue
            elif char == ESCAPE.symbol:
                escape = True
                continue

            if char == '´':
                instruction_type = char + line[i+1]
                instructions.append(char)
                chars_in_instruction = 1
                if instruction_type not in [x.symbol for x in _INSTRUCTIONS]:
                    log(f'error: unknown instruction type "{instruction_type}" in line "{line}"',
                        'qp.qlang._tokenize_code', verbosity)
                if _INSTRUCTIONS[instruction_type].copy_df== True:
                    copy_df = True
            elif char in [_CONNECTORS.AND.symbol, _CONNECTORS.OR.symbol]:
                if chars_in_instruction >= 3:
                    instructions.append(f'{instruction_type} {char}')
                    chars_in_instruction = 3
                elif i == 0:
                    instructions.append(f'{instruction_type} {char}')
                    chars_in_instruction = 3
                else:
                    instructions[-1] += char
                    chars_in_instruction += 1
            elif i == 0:
                instructions.append(f'{instruction_type} {char}')
                chars_in_instruction = 3
            elif char == ' ':
                instructions[-1] += char
            else:
                instructions[-1] += char
                chars_in_instruction += 1

        log(f'debug: parsed line "{line}" into instruction strings: {instructions}',
            'qp.qlang._tokenize_code', verbosity)
        
        instructions_all += instructions

    return lines, instructions_all, copy_df



def _parse_instruction(instruction_str, verbosity):
    """
    Parses an instruction string into an instruction object.
    """

    instruction, text = _extract_symbol(instruction_str, symbols=[x for x in _INSTRUCTIONS], verbosity=verbosity)
    instruction.connector, text = _extract_symbol(text, symbols=instruction.connectors, verbosity=verbosity)

    if hasattr(instruction, 'scopes'):
        instruction.scope, text = _extract_symbol(text, symbols=instruction.scopes, verbosity=verbosity)

    if hasattr(instruction, 'negations'):
        instruction.negation, text = _extract_symbol(text, symbols=instruction.negations, verbosity=verbosity)

    instruction.operator, text = _extract_symbol(text, symbols=instruction.operators, verbosity=verbosity)
    instruction.value = text.strip()

    if instruction.operator.unary and len(instruction.value)>0:
        log(f'warning: unary operator "{instruction.operator}" cannot use a value. value "{instruction.value}" will be ignored',
            'qp.qlang._parse_instruction', verbosity)
        instruction.value = ''

    log(f'debug: parsed "{instruction_str}" as instruction:\n{instruction}',
        'qp.qlang._parse_instruction', verbosity)

    instruction.str = instruction_str
    return instruction


def _extract_symbol(string, symbols, verbosity):
    """
    Looks for expected syntax symbols at the beginning of an instruction string.
    """

    string = string.strip()

    if len(symbols) == 0:
        return None, string
    elif len(symbols) == 1:
        symbol = symbols[0]
        return symbol, string[len(symbol.symbol):].strip()
    else:
        default = symbols[0]
        symbols = symbols[1:]

    for symbol in symbols:
        if string.startswith(symbol.symbol):
            log(f'trace: found symbol in string "{string}":\n{symbol}', 'qp.qlang._extract_symbol', verbosity)
            return symbol, string[len(symbol.symbol):].strip()
    
    if string.startswith(default.symbol):
        return default, string[len(default.symbol):].strip()
    else:
        log(f'trace: no symbol found in string "{string}". using default:\n{default}', 'qp.qlang._extract_symbol', verbosity)
        return default, string


def _filter_series(series, negation, operator, value, verbosity, df_new=None):
    """
    Filters a pandas series by applying a condition.
    Conditions are made up of a comparison operator and for binary operators a value to compare to.
    negation.TRUE inverts the result of the condition.
    """

    if operator in [
        _OPERATORS.BIGGER_EQUAL, _OPERATORS.SMALLER_EQUAL, _OPERATORS.BIGGER, _OPERATORS.SMALLER,
        _OPERATORS.EQUAL, _OPERATORS.STRICT_EQUAL,
        ]:
        value_type = _type(value)
        numeric_dtypes = [
            'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
            'float32', 'float64',
            ]
        datetime_dtypes = [
            'datetime64[ms]', 'datetime64[ms, UTC]',
            'datetime64[us]', 'datetime64[us, UTC]',
            'datetime64[ns]', 'datetime64[ns, UTC]',
            'datetime64[s]', 'datetime64[s, UTC]',
            ]
        string_dtypes = ['object', 'string']

        if value_type in ('int', 'float', 'num'):
            value = pd.to_numeric(value, errors='coerce')
            if series.dtype not in numeric_dtypes:
                series = pd.to_numeric(series, errors='coerce')

        elif value_type in ('date', 'datetime'):
            value = _datetime(value, errors='ignore')
            if series.dtype not in datetime_dtypes:
                log(f'warning: series "{series.name}" is not a datetime series, consider converting it with "´v to datetime"',
                    'qp.qlang._filter_series', verbosity)
                series = pd.to_datetime(series, errors='coerce')

        elif value_type == 'str':
            if series.dtype not in string_dtypes:
                series = series.astype(str)
            if operator == _OPERATORS.EQUAL:
                value = value.lower()
                series = series.str.lower()

        elif isinstance(value, pd.Series):
            if operator == _OPERATORS.EQUAL:
                if series.dtype in string_dtypes or value.dtype in string_dtypes:
                    series = series.str.lower()
                    value = value.str.lower()
                


    #numeric/date/datetime/order comparison
    if operator == _OPERATORS.BIGGER_EQUAL:
        filtered = series >= value
    elif operator == _OPERATORS.SMALLER_EQUAL:
        filtered = series <= value
    elif operator == _OPERATORS.BIGGER:
        filtered = series > value
    elif operator == _OPERATORS.SMALLER:
        filtered = series < value


    #string equality comparison
    elif operator == _OPERATORS.STRICT_EQUAL:
        filtered = series == value
    elif operator == _OPERATORS.EQUAL:
        filtered = series == value


    #substring comparison
    elif operator == _OPERATORS.STRICT_CONTAINS:
        filtered = series.astype(str).str.contains(value, case=True, regex=False)
    elif operator == _OPERATORS.CONTAINS:
        filtered = series.astype(str).str.contains(value, case=False, regex=False)


    #regex comparison
    elif operator == _OPERATORS.MATCHES_REGEX:
        filtered = series.astype(str).str.fullmatch(value) 
    elif operator == _OPERATORS.CONTAINS_REGEX:
        filtered = series.astype(str).str.contains(value)


    #lambda function
    elif operator == _OPERATORS.EVAL:
        filtered = series.apply(lambda x: eval(value, {'x': x, 'col': series, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp}))
    elif operator == _OPERATORS.COL_EVAL:
        filtered = eval(value, {'col': series, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})


    #type checks
    elif operator == _OPERATORS.IS_STR:
        filtered = series.apply(lambda x: isinstance(x, str))
    elif operator == _OPERATORS.IS_INT:
        filtered = series.apply(lambda x: isinstance(x, int))
    elif operator == _OPERATORS.IS_FLOAT:
        filtered = series.apply(lambda x: isinstance(x, float))
    elif operator == _OPERATORS.IS_NUM:
        filtered = series.apply(lambda x: _num(x, errors='ERROR')) != 'ERROR'
    elif operator == _OPERATORS.IS_BOOL:
        filtered = series.apply(lambda x: isinstance(x, bool))

    elif operator == _OPERATORS.IS_DATETIME:
        filtered = series.apply(lambda x: _datetime(x, errors='ERROR')) != 'ERROR'
    elif operator == _OPERATORS.IS_DATE:
        filtered = series.apply(lambda x: _date(x, errors='ERROR')) != 'ERROR'

    elif operator == _OPERATORS.IS_ANY:
        filtered = series.apply(lambda x: True)
    elif operator == _OPERATORS.IS_NA:
        filtered = series.apply(lambda x: _na(x, errors='ERROR')) != 'ERROR'
    elif operator == _OPERATORS.IS_NK:
        filtered = series.apply(lambda x: _nk(x, errors='ERROR')) != 'ERROR'
    elif operator == _OPERATORS.IS_YN:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR')) != 'ERROR'
    elif operator == _OPERATORS.IS_YES:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR', yes=1)) == 1
    elif operator == _OPERATORS.IS_NO:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR', no=0)) == 0
        
    elif operator == _OPERATORS.IS_UNIQUE:
        filtered = series.duplicated(keep=False) == False
    elif operator == _OPERATORS.IS_FIRST:
        filtered = series.duplicated(keep='first') == False
    elif operator == _OPERATORS.IS_LAST:
        filtered = series.duplicated(keep='last') == False

    else:
        log(f'error: operator "{operator}" is not implemented', 'qp.qlang._filter_series', verbosity)
        filtered = None


    if negation == _NEGATIONS.TRUE:
        filtered = ~filtered

    return filtered


def _update_selected_cols(values, values_new, connector, verbosity):
    """
    Updates the previously selected columns based on the new selection.
    """
    if values is None:
        values = values_new
    elif connector == _CONNECTORS.RESET:
        values = values_new
    elif connector in [_CONNECTORS.AND, _SCOPES.ALL]:
        values &= values_new
    elif connector in [_CONNECTORS.OR, _SCOPES.ANY]:
        values |= values_new
    else:
        log(f'error: connector "{connector}" is not implemented', 'qp.qlang._update_selected_cols', verbosity)
    return values


@pd.api.extensions.register_dataframe_accessor('check')
class DataFrameCheck:
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __call__(self, verbosity=3):
        _check_df(self.df, verbosity=verbosity)
        return self.df


@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    A query language for pandas data exploration/analysis/modification.
    df.qi() without any args can be used to interactively build a query in Jupyter notebooks.

    
    examples:

    #select col
    df.q('id')

    #equivalent to
    df.q('´c id')

    #select multiple cols
    df.q('id / name')

    #select rows in a col which fullfill a condition
    df.q('id  ´r > 20000')

    #select rows fullfilling multiple conditions in the same col
    df.q('id  ´r > 20000 & < 30000')

    #select rows fullfilling both conditions in different cols
    df.q('id  ´r > 20000   ´c name  ´r & ?john')

    #select rows fullfilling either condition in different cols
    df.q('id  ´r > 20000   ´c name  ´r / ?john')

    """

    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():\n' + self.__doc__
    
    def __call__(self, code=''):
        return query(self.df, code)



@pd.api.extensions.register_dataframe_accessor('qi')
class DataFrameQueryInteractiveMode:
    """
    Interactive version of df.q() for building queries in Jupyter notebooks.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self):
        kwargs = {'df': fixed(self.df), 'code': ''}

        #code input
        ui_code = widgets.Textarea(
            value='´s verbosity=3\n´s diff=None\n\n#Enter query code here\n\n',
            layout=Layout(width='99%', height='97%')
            )


        #query builder

        instruction = _INSTRUCTIONS.SELECT_COLS

        i_type = widgets.Dropdown(
            options=[(f'{s.symbol}: {s.description}', s.symbol) for s in _INSTRUCTIONS],
            value=instruction.symbol,
            )
        
        i_scope = widgets.Dropdown(
            disabled=True,
            options=[''],
            value='',
            )

        i_negate = widgets.ToggleButtons(
            options=[('dont negate condition', ''), ('negate condition', '!')],
            value='',
            )

        i_operator = widgets.Dropdown(
            options=[(f'{s.symbol}: {s.description}', s.symbol) for s in instruction.operators],
            value=instruction.operators[0].symbol,
            )
        
        i_value = widgets.Text(
            value='',
            )
        

        i_text = widgets.Text(
            value=f'\n{i_type.value} {i_scope.value} {i_negate.value}{i_operator.value} {i_value.value}',
            disabled=True,
            )
        

        def update_options(*args):
            instruction = _INSTRUCTIONS[i_type.value]

            if hasattr(instruction, 'scopes'):
                i_scope.disabled = False
                i_scope.options = [(f'{s.symbol}: {s.description}', s.symbol) for s in instruction.scopes]
            else:
                i_scope.disabled = True
                i_scope.options = ['']

            if hasattr(instruction, 'negations'):
                i_negate.disabled = False
                i_negate.options = [('dont negate condition', ''), ('negate condition', '!')]
            else:
                i_negate.disabled = True
                i_negate.options = ['', '']

            i_operator.options = [(f'{s.symbol}: {s.description}', s.symbol) for s in instruction.operators]
            i_operator.value = instruction.operators[0].symbol

        def update_text(*args):
            i_text.value = f'{i_type.value} {i_scope.value} {i_negate.value}{i_operator.value} {i_value.value}\n'

        i_type.observe(update_options, 'value')
        i_type.observe(update_text, 'value')
        i_scope.observe(update_text, 'value')
        i_negate.observe(update_text, 'value')
        i_operator.observe(update_text, 'value')
        i_value.observe(update_text, 'value')

        
        ui_add_instruction = widgets.Button(
            button_style='success',
            tooltip='adds the selected instruction to the query code',
            icon='check'
            )

        def add_instruction(ui_code, i_text):
            if i_text.value.startswith('´c'):
                ui_code.value += f'\n{i_text.value}'
            else:
                ui_code.value += f'   {i_text.value}'

        ui_add_instruction.on_click(lambda b: add_instruction(ui_code, i_text))

        ui_input = VBox([
            widgets.HTML(value='<b>query builder:</b>'),
            i_text,
            i_type,
            i_scope,
            i_negate,
            i_operator,
            i_value,
            ui_add_instruction,
            ])

        
        #some general info and statistics about the df
        mem_usage = self.df.memory_usage().sum() / 1024
        ui_details = widgets.HTML(
            value=f"""
            <b>rows:</b> {len(self.df.index)}<br>
            <b>columns:</b> {len(self.df.columns)}<br>
            <b>memory usage:</b> {mem_usage:,.3f}kb<br>
            <b>unique values:</b> {self.df.nunique().sum()}<br>
            <b>missing values:</b> {self.df.isna().sum().sum()}<br>
            <b>columns:</b><br> {'<br>'.join([f'{col} ({dtype})' for col, dtype in list(zip(self.df.columns, self.df.dtypes))])}<br>
            """
            ) 

        ui_tabs = widgets.Tab(
            children=[
                ui_code,
                ui_details,
                widgets.HTML(value=DataFrameQuery.__doc__.replace('\n', '<br>').replace('    ', '&emsp;')),
                ],
            titles=['code', 'details', 'readme'],
            layout=Layout(width='50%', height='95%')
            )
        

        
        ui = HBox([ui_tabs, ui_input], layout=Layout(width='100%', height='300px'))

        kwargs['code'] = ui_code

        display(ui)
        out = HBox([interactive_output(_interactive_mode, kwargs)], layout=Layout(overflow_y='auto'))
        display(out)


def _interactive_mode(**kwargs):
    df = kwargs.pop('df')
    code = kwargs.pop('code')
    result = query(df, code)
    display(result)
    return result 


