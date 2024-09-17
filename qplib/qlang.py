
import re
import numpy as np
import pandas as pd
import qplib as qp

from IPython.display import display
from ipywidgets import widgets, interactive_output, HBox, VBox, fixed, Layout

from .util import log
from .types import _int, _float, _num, _bool, _datetime, _date, _na, _nk, _yn, _type
from .pd_util import _check_df, _diff, _format_df




class Symbol:
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
        repr = f'{self.name}\n\tsymbol: "{self.symbol}"\n\tdescription: "{self.description})"'
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


class Symbols:
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
            log(f'error: symbol "{key}" not found in "{self.name}"', 'qp.qlang.Symbols.__getitem__', 3)
            return None

    def __iter__(self):
        return iter(self.by_name.values())

    def __repr__(self):
        return f'{self.name}:\n' + '\n\t'.join([str(val) for key,val in self.by_name.items()])
    
    def __str__(self):
        return f'{self.name}:\n' + '\n\t'.join([str(val) for key,val in self.by_name.items()])



INPLACE = False
COMMENT = Symbol('#', 'COMMENT', 'comments out the rest of the line')
ESCAPE = Symbol('`', 'ESCAPE', 'escape the next character')

CONNECTORS = Symbols('CONNECTORS',
    Symbol('', 'RESET', 'only the current condition must be fulfilled'),
    Symbol('&', 'AND', 'this condition and the previous condition/s must be fulfilled'),
    Symbol('/', 'OR', 'this condition or the previous condition/s must be fulfilled'),
    )

SCOPES = Symbols('SCOPES',
    Symbol('any', 'ANY', 'any of the currently selected columns must fulfill the condition'),
    Symbol('all', 'ALL', 'all of the currently selected columns must fulfill the condition'),
    Symbol('idx', 'IDX', 'the index of the dataframe must fulfill the condition'),
    )

NEGATIONS = Symbols('NEGATIONS',
    Symbol('', 'FALSE', 'dont negate the condition'),
    Symbol('!', 'TRUE', 'negate the condition'),
    )

OPERATORS = Symbols('OPERATORS',
    #for changing settings
    Symbol('verbosity=', 'SET_VERBOSITY', 'change the verbosity level'),
    Symbol('diff=', 'SET_DIFF', 'change the diff setting'),


    #for filtering
    Symbol('>=', 'BIGGER_EQUAL', 'bigger or equal', binary=True),
    Symbol('<=', 'SMALLER_EQUAL', 'smaller or equal', binary=True),
    Symbol('>', 'BIGGER', 'bigger', binary=True),
    Symbol('<', 'SMALLER', 'smaller', binary=True),

    Symbol('==', 'STRICT_EQUAL', 'equal to (case sensitive)', binary=True),
    Symbol('=', 'EQUAL', 'equal to', binary=True),

    Symbol('??', 'STRICT_CONTAINS', 'contains a string (case sensitive)', binary=True),
    Symbol('?', 'CONTAINS', 'contains a string (not case sensitive)', binary=True),

    Symbol('r=', 'MATCHES_REGEX', 'matches a regex', binary=True),
    Symbol('r?', 'CONTAINS_REGEX', 'contains a regex', binary=True),

    Symbol('~', 'EVAL', 'select values by evaluating a python expression on each value', binary=True),
    Symbol('col~', 'COL_EVAL', 'select rows by evaluating a python expression on a whole column', binary=True),

    Symbol('@', 'LOAD_SELECTION', 'load a saved selection from a boolean column', binary=True),

    Symbol('is any', 'IS_ANY', 'is any value', unary=True),
    Symbol('is str', 'IS_STR', 'is string', unary=True),
    Symbol('is int', 'IS_INT', 'is integer', unary=True),
    Symbol('is float', 'IS_FLOAT', 'is float', unary=True),
    Symbol('is num', 'IS_NUM', 'is number', unary=True),
    Symbol('is bool', 'IS_BOOL', 'is boolean', unary=True),
    Symbol('is datetime', 'IS_DATETIME', 'is datetime', unary=True),
    Symbol('is date', 'IS_DATE', 'is date', unary=True),
    Symbol('is na', 'IS_NA', 'is missing value', unary=True),
    Symbol('is nk', 'IS_NK', 'is not known value', unary=True),
    Symbol('is yn', 'IS_YN', 'is yes or no value', unary=True),
    Symbol('is yes', 'IS_YES', 'is yes value', unary=True),
    Symbol('is no', 'IS_NO', 'is no value', unary=True),
    Symbol('is unique', 'IS_UNIQUE', 'is a unique value', unary=True),
    Symbol('is first', 'IS_FIRST', 'is the first value (of multiple values)', unary=True),
    Symbol('is last', 'IS_LAST', 'is the last value (of multiple values)', unary=True),


    #for modifying values and headers
    Symbol('=', 'SET_VAL', 'replace value with the given string'),
    Symbol('+=', 'ADD_VAL', 'append a string to the value (coerce to string if needed)'),

    Symbol('~', 'SET_EVAL', 'replace value by evaluating a python expression for each selected value/header'),
    Symbol('col~', 'SET_COL_EVAL', 'replace value by evaluating a python expression for each selected column'),
    
    Symbol('sort', 'SORT', 'sort values based on the selected column(s)', unary=True),

    Symbol('to str', 'TO_STR', 'convert to string', unary=True),
    Symbol('to int', 'TO_INT', 'convert to integer', unary=True),
    Symbol('to float', 'TO_FLOAT', 'convert to float', unary=True),
    Symbol('to num', 'TO_NUM', 'convert to number', unary=True),
    Symbol('to bool', 'TO_BOOL', 'convert to boolean', unary=True),
    Symbol('to datetime', 'TO_DATETIME', 'convert to datetime', unary=True),
    Symbol('to date', 'TO_DATE', 'convert to date', unary=True),
    Symbol('to na', 'TO_NA', 'convert to missing value', unary=True),
    Symbol('to nk', 'TO_NK', 'convert to not known value', unary=True),
    Symbol('to yn', 'TO_YN', 'convert to yes or no value', unary=True),


    #for adding new columns
    Symbol('=', 'STR_COL', 'add new column, fill it with the given string and select it', binary=True),
    Symbol('~', 'EVAL_COL', 'add new column, fill it by evaluating a python expression and select it', binary=True),
    Symbol('@', 'SAVE_SELECTION', 'add a new boolean column with the given name and select it. all currently selected rows are set to True, the rest to False', binary=True),
    
    
    #for modifying metadata (part of miscellaneous instructions)
    Symbol('=', 'SET_METADATA', 'set contents of the columnn named "meta" to the given string', binary=True),
    Symbol('+=', 'ADD_METADATA', 'append the given string to the contents of the column named "meta"', binary=True),
    Symbol('@', 'TAG_METADATA', 'add a tag of the currently selected column(s) in the form of "<value>@<selected col>;" to the column named "meta"', binary=True),
    Symbol('~', 'SET_METADATA_EVAL', 'set contents of the column named "meta" by evaluating a python expression for each selected value in the metadata', binary=True),
    Symbol('col~', 'SET_METADATA_COL_EVAL', 'set contents of the column named "meta" by evaluating a python expression on the whole metadata column', binary=True),
    )



def _select_cols(instruction, df_new, rows, cols, diff, verbosity):
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

    cols = _update_selection(cols, cols_new, connector)

    if cols.any() == False and connector == CONNECTORS.AND:
        log(f'warning: no columns fulfill the condition in "{instruction.str}" and the previous conditions',
            'qp.qlang._select_cols', verbosity)

    return df_new, rows, cols, diff, verbosity



def _select_rows(instruction, df_new, rows, cols, diff, verbosity):
    """
    An Instruction to select rows fulfilling a condition.
    """

    connector = instruction.connector
    scope = instruction.scope
    negation = instruction.negation
    operator = instruction.operator
    value = instruction.value
    rows_all = df_new.index.to_series()
        
    if value.startswith('@'):
        column = value[1:]
        if column in df_new.columns:
            value = df_new[column]
        else:
            log(f'error: column "{column}" not found in dataframe. cannot use "@{column}" as value for row selection',
                'qp.qlang._select_rows', verbosity)


    if cols.any() == False:
        log(f'warning: row filter cannot be applied when no columns where selected', 'qp.qlang._select_rows', verbosity)
        rows = pd.Series(False, index=rows_all.index)
        return df_new, rows, cols, diff, verbosity
            
    if scope == SCOPES.IDX:
        rows_new = _filter_series(rows_all, negation, operator, value, verbosity, df_new)
        rows = _update_selection(rows, rows_new, connector)

    else:
        rows_temp = None
        for col in df_new.columns[cols]:
            rows_new = _filter_series(df_new[col], negation, operator, value, verbosity, df_new)
            rows_temp = _update_selection(rows_temp, rows_new, scope)
        rows = _update_selection(rows, rows_temp, connector)

        if rows_temp.any() == False:
            log(f'warning: no rows fulfill the condition in "{instruction}"', 'qp.qlang._select_rows', verbosity)

    return df_new, rows, cols, diff, verbosity


def _modify_vals(instruction, df_new, rows, cols, diff, verbosity):
    """
    An Instruction to modify the selected values.
    """

    operator = instruction.operator
    value = instruction.value

    #data modification  
    if operator == OPERATORS.SET_VAL:
        df_new.loc[rows, cols] = value
    elif operator == OPERATORS.ADD_VAL:
        df_new.loc[rows, cols] = df_new.loc[rows, cols].astype(str) + value
    elif operator == OPERATORS.SET_COL_EVAL:
        df_new.loc[:, cols] = df_new.loc[:, cols].apply(lambda x: eval(value, {'col': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}), axis=0)
    elif operator == OPERATORS.SORT:
        df_new.sort_values(by=list(df_new.columns[cols]), axis=0, inplace=True)
        rows.index = df_new.index

    elif pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
        #data modification
        if operator == OPERATORS.SET_EVAL:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))


        #type conversion
        elif operator == OPERATORS.TO_STR:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(str)
        elif operator == OPERATORS.TO_INT:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_int)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('Int64')
        elif operator == OPERATORS.TO_FLOAT:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_float)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('Float64')
        elif operator == OPERATORS.TO_NUM:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_num)
        elif operator == OPERATORS.TO_BOOL:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_bool)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('boolean')
        
        elif operator == OPERATORS.TO_DATETIME:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_datetime)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('datetime64[ns]')
        elif operator == OPERATORS.TO_DATE:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_date)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('datetime64[ns]').dt.floor('d')

        elif operator == OPERATORS.TO_NA:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_na)
        elif operator == OPERATORS.TO_NK:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_nk)
        elif operator == OPERATORS.TO_YN:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].map(_yn)

    else:
        #data modification
        if operator == OPERATORS.SET_EVAL:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))

        #type conversion
        elif operator == OPERATORS.TO_STR:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(str)
        elif operator == OPERATORS.TO_INT:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_int)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('Int64')
        elif operator == OPERATORS.TO_FLOAT:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_float)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('Float64')
        elif operator == OPERATORS.TO_NUM:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_num)
        elif operator == OPERATORS.TO_BOOL:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_bool)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('boolean')
        
        elif operator == OPERATORS.TO_DATETIME:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_datetime)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('datetime64[ns]')
        elif operator == OPERATORS.TO_DATE:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_date)
            for col in df_new.columns[cols]:
                df_new[col] = df_new[col].astype('datetime64[ns]').dt.floor('d')

        elif operator == OPERATORS.TO_NA:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_na)
        elif operator == OPERATORS.TO_NK:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_nk)
        elif operator == OPERATORS.TO_YN:
            df_new.loc[rows, cols] = df_new.loc[rows, cols].applymap(_yn)

    return df_new, rows, cols, diff, verbosity


def _modify_headers(instruction, df_new, rows, cols, diff, verbosity):
    """
    An Instruction to modify the headers of the selected column(s).
    """

    operator = instruction.operator
    value = instruction.value

    if operator == OPERATORS.SET_VAL:
        df_new.rename(columns={col: value for col in df_new.columns[cols]}, inplace=True)
        cols.index = df_new.columns

    if operator == OPERATORS.ADD_VAL:
        df_new.rename(columns={col: col + value for col in df_new.columns[cols]}, inplace=True)
        cols.index = df_new.columns

    if operator == OPERATORS.SET_EVAL:
        df_new.rename(
            columns={
                col: eval(value, {'x': col, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
                for col in df_new.columns[cols]
                },
            inplace=True
            )
        cols.index = df_new.columns

    return df_new, rows, cols, diff, verbosity


def _new_col(instruction, df_new, rows, cols, diff, verbosity):
    """
    An Instruction to add a new column.
    """

    operator = instruction.operator
    value = instruction.value

    if operator == OPERATORS.STR_COL:
        for i in range(1, 1001):
            if i == 1000:
                log(f'warning: could not add new column. too many columns named "new<x>"',
                    'qp.qlang._new_col', verbosity)
                break

            header = 'new' + str(i)
            if header not in df_new.columns:
                df_new[header] = ''
                df_new.loc[rows, header] = value
                cols = pd.Series([True if col == header else False for col in df_new.columns])
                cols.index = df_new.columns
                break
    
    elif operator == OPERATORS.EVAL_COL:
        for i in range(1, 1001):
            if i == 1000:
                log(f'warning: could not add new column. too many columns named "new<x>"',
                    'qp.qlang._new_col', verbosity)
                break

            header = 'new' + str(i)
            if header not in df_new.columns: 
                value = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
                if isinstance(value, pd.Series):
                    df_new[header] = value
                else:
                    df_new[header] = pd.NA
                    df_new.loc[rows, header] = value
                cols = pd.Series([True if col == header else False for col in df_new.columns])
                cols.index = df_new.columns
                break
    

    elif operator == OPERATORS.SAVE_SELECTION:
        if value in df_new.columns:
            log(f'warning: column "{value}" already exists in dataframe. selecting existing col and resetting values',
                'qp.qlang._new_col', verbosity)
        df_new[value] = rows
        cols = pd.Series([True if col == value else False for col in df_new.columns])
        cols.index = df_new.columns

    return df_new, rows, cols, diff, verbosity


def _miscellaneous(instruction, df_new, rows, cols, diff, verbosity):
    """
    An Instruction for miscellaneous tasks, for example, modifying metadata.
    """
    
    operator = instruction.operator
    value = instruction.value
    
    operators_metadata = [
        OPERATORS.SET_METADATA,
        OPERATORS.ADD_METADATA,
        OPERATORS.TAG_METADATA,
        OPERATORS.SET_METADATA_EVAL,
        OPERATORS.SET_METADATA_COL_EVAL,
        ]
    if operator in operators_metadata and 'meta' not in df_new.columns:
        log(f'info: no metadata column found in dataframe. creating new column named "meta',
            'qp.qlang.Miscellaneous.apply', verbosity)
        df_new['meta'] = ''
        cols = pd.concat([cols, pd.Series([False])])
        cols.index = df_new.columns
    

    #modify metadata
    if operator == OPERATORS.SET_METADATA:
        df_new.loc[rows, 'meta'] = value

    elif operator == OPERATORS.ADD_METADATA:
        df_new.loc[rows, 'meta'] += value

    elif operator == OPERATORS.TAG_METADATA:
        tag = ''
        for col in df_new.columns[cols]:
            tag += f'{value}@{col};'
        df_new.loc[rows, 'meta'] += tag

    elif operator == OPERATORS.SET_METADATA_EVAL:
        if pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
            df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].map(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
        else:
            df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].applymap(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
        
    elif operator == OPERATORS.SET_METADATA_COL_EVAL:
        df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].apply(lambda x: eval(value, {'col': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))

    return df_new, rows, cols, diff, verbosity

def _modify_settings(instruction, df_new, rows, cols, diff, verbosity):
    """
    An instruction to change the settings for the query.
    """

    operator = instruction.operator
    value = instruction.value
    
    if operator == OPERATORS.SET_VERBOSITY:
        if value in ['0', '1', '2', '3', '4', '5']:
            verbosity = int(value)
        else:
            log(f'warning: verbosity must be an integer between 0 and 5. "{value}" is not valid',
                'qp.qlang._modify_settings', verbosity)
    
    elif operator == OPERATORS.SET_DIFF:
        if value.lower() in ['none', '0', 'false']:
            diff = None
        elif value.lower() in ['mix', 'new', 'old', 'new+']:
            diff = value.lower()
        else:
            log(f'warning: diff must be one of [None, "mix", "old", "new", "new+"]. "{value}" is not valid',
                'qp.qlang._modify_settings', verbosity)

    return df_new, rows, cols, diff, verbosity


INSTRUCTIONS = Symbols('INSTRUCTIONS',
                       
    Symbol('´c', 'SELECT_COLS', 'select columns fulfilling a condition',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        negations=[
            NEGATIONS.FALSE, #default
            NEGATIONS.TRUE
            ],
        operators=[
            OPERATORS.EQUAL, #default

            #binary
            OPERATORS.BIGGER_EQUAL, OPERATORS.SMALLER_EQUAL, OPERATORS.BIGGER, OPERATORS.SMALLER,
            OPERATORS.STRICT_EQUAL, OPERATORS.EQUAL,
            OPERATORS.STRICT_CONTAINS, OPERATORS.CONTAINS,
            OPERATORS.MATCHES_REGEX, OPERATORS.CONTAINS_REGEX,
            OPERATORS.EVAL,
            OPERATORS.LOAD_SELECTION,
        
            #unary
            OPERATORS.IS_ANY,
            OPERATORS.IS_UNIQUE,
            OPERATORS.IS_NA, OPERATORS.IS_NK,
            OPERATORS.IS_STR, OPERATORS.IS_INT, OPERATORS.IS_FLOAT, OPERATORS.IS_NUM, OPERATORS.IS_BOOL,
            OPERATORS.IS_DATE, OPERATORS.IS_DATETIME,
            OPERATORS.IS_YN, OPERATORS.IS_YES, OPERATORS.IS_NO,
            ],
        copy_df= False,
        apply=_select_cols,
        ),


    Symbol('´r', 'SELECT_ROWS', 'select rows fulfilling a condition',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        scopes=[
            SCOPES.ANY, #default
            SCOPES.ALL,
            SCOPES.IDX
            ],
        negations=[
            NEGATIONS.FALSE, #default
            NEGATIONS.TRUE
            ],
        operators=[
            OPERATORS.EQUAL, #default

            #binary
            OPERATORS.BIGGER_EQUAL, OPERATORS.SMALLER_EQUAL, OPERATORS.BIGGER, OPERATORS.SMALLER,
            OPERATORS.STRICT_EQUAL, OPERATORS.EQUAL,
            OPERATORS.STRICT_CONTAINS, OPERATORS.CONTAINS,
            OPERATORS.MATCHES_REGEX, OPERATORS.CONTAINS_REGEX,
            OPERATORS.EVAL, OPERATORS.COL_EVAL,
            OPERATORS.LOAD_SELECTION,
        
            #unary
            OPERATORS.IS_ANY,
            OPERATORS.IS_UNIQUE, OPERATORS.IS_FIRST, OPERATORS.IS_LAST,
            OPERATORS.IS_NA, OPERATORS.IS_NK,
            OPERATORS.IS_STR, OPERATORS.IS_INT, OPERATORS.IS_FLOAT, OPERATORS.IS_NUM, OPERATORS.IS_BOOL,
            OPERATORS.IS_DATETIME, OPERATORS.IS_DATE,
            OPERATORS.IS_YN, OPERATORS.IS_YES, OPERATORS.IS_NO,
            ],
        copy_df= False,
        apply=_select_rows,
        ),


    Symbol('´v', 'MODIFY_VALS', 'modify the selected values',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        operators=[
            OPERATORS.SET_VAL, #default
            OPERATORS.ADD_VAL,
            OPERATORS.SET_EVAL, OPERATORS.SET_COL_EVAL,
            OPERATORS.SORT,
            OPERATORS.TO_STR, OPERATORS.TO_INT, OPERATORS.TO_FLOAT, OPERATORS.TO_NUM, OPERATORS.TO_BOOL,
            OPERATORS.TO_DATETIME, OPERATORS.TO_DATE, OPERATORS.TO_NA, OPERATORS.TO_NK, OPERATORS.TO_YN,
            ],
        copy_df= True,
        apply=_modify_vals,
        ),

    Symbol('´h', 'MODIFY_HEADERS', 'modify headers of the selected columns',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        operators=[
            OPERATORS.SET_VAL, #default
            OPERATORS.ADD_VAL,
            OPERATORS.SET_EVAL,
            ],
        copy_df= True,
        apply=_modify_headers,
        ),

    Symbol('´n', 'NEW_COL', 'add new column',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        operators=[
            OPERATORS.STR_COL, #default
            OPERATORS.EVAL_COL,
            OPERATORS.SAVE_SELECTION,
            ],
        copy_df= True,
        apply=_new_col,
        ),

    Symbol('´m', 'MISCELLANEOUS', 'miscellaneous instructions',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        operators=[
            OPERATORS.SET_METADATA, #default
            OPERATORS.ADD_METADATA,
            OPERATORS.TAG_METADATA,
            OPERATORS.SET_METADATA_EVAL,
            OPERATORS.SET_METADATA_COL_EVAL,
            ],
        copy_df= True,
        apply=_miscellaneous,
        ),

    Symbol('´s', 'MODIFY_SETTINGS', 'change query settings',
        connectors=[
            CONNECTORS.RESET,#default
            CONNECTORS.AND,
            CONNECTORS.OR
            ],
        operators=[
            OPERATORS.SET_VERBOSITY, #default
            OPERATORS.SET_DIFF
            ],
        copy_df= False,
        apply=_modify_settings,
        ),
    )



def query(df_old, code=''):
    """
    A query language for pandas data exploration/analysis/modification.

    examples:
    df.q('id')  #selects the column 'id'
    df.q('id  ´r > 100)  #selects col "id" and rows where the value is greater than 100
    df.q('´c = id  ´r > 100) #same as above but more explicit
    df.q('id  ´r > 100  ´c / name  ´r ? john')  #selects col "id" and
        #rows where the value is greater than 100 or col "name" and rows where the value contains "john"
    """

    #setup
    _check_df(df_old)
    diff = None
    verbosity = 3


    #parse and apply instructions

    lines, instruction_strs, copy_df = _tokenize_code(code, verbosity)
    if INPLACE:
        df_new = df_old
    elif copy_df:
        df_new = df_old.copy()
    else:
        df_new = df_old
    
    cols = pd.Series([True for col in df_new.columns])
    rows = pd.Series([True for row in df_new.index])
    cols.index = df_new.columns
    rows.index = df_new.index

    for instruction_str in instruction_strs:
        instruction = _parse_instruction(instruction_str, verbosity)
        df_new, rows, cols, diff, verbosity  = instruction.apply(instruction, df_new, rows, cols, diff, verbosity)


    #results

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


def _tokenize_code(code, verbosity):
    """
    Turns the plain text input string into a list of instructions for the query parser.
    """

    lines = []
    instructions_all = []

    #get lines and instruction blocks
    for line_num, line in enumerate(code.split('\n')):
        line = line.strip()
        lines.append([line_num, line])
        line = line.split(COMMENT.symbol)[0].strip()
        instructions = []
        copy_df = False
    
        if line == '':
            continue


        escape = False
        chars_in_instruction = 0
        instruction_type = INSTRUCTIONS.SELECT_COLS.symbol  #default

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
                if instruction_type not in [x.symbol for x in INSTRUCTIONS]:
                    log(f'error: unknown instruction type "{instruction_type}" in line "{line}"',
                        'qp.qlang._tokenize', verbosity)
                if INSTRUCTIONS[instruction_type].copy_df== True:
                    copy_df = True
            elif char in [CONNECTORS.AND.symbol, CONNECTORS.OR.symbol]:
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
            'qp.qlang._tokenize', verbosity)
        
        instructions_all += instructions

    return lines, instructions_all, copy_df



def _parse_instruction(instruction_str, verbosity):
    """
    Parses an instruction string into an instruction object.
    """

    instruction, text = _extract_symbol(instruction_str, symbols=[x for x in INSTRUCTIONS], verbosity=verbosity)
    instruction.str = instruction_str
    instruction.repr = f'{instruction.name}:\n'

    instruction.connector, text = _extract_symbol(text, symbols=instruction.connectors, verbosity=verbosity)
    instruction.repr += f'\tconnector: {instruction.connector}\n'

    if hasattr(instruction, 'scopes'):
        instruction.scope, text = _extract_symbol(text, symbols=instruction.scopes, verbosity=verbosity)
        instruction.repr += f'\tscope: {instruction.scope}\n'

    if hasattr(instruction, 'negations'):
        instruction.negation, text = _extract_symbol(text, symbols=instruction.negations, verbosity=verbosity)
        instruction.repr += f'\tnegation: {instruction.negation}\n'

    instruction.operator, text = _extract_symbol(text, symbols=instruction.operators, verbosity=verbosity)
    instruction.repr += f'\toperator: {instruction.operator}\n'

    instruction.value = text.strip()
    instruction.repr += f'\tvalue: {instruction.value}\n'

    if instruction.operator.unary and len(instruction.value)>0:
        log(f'warning: unary operator "{instruction.operator}" cannot use a value. value "{instruction.value}" will be ignored',
            'qp.qlang._parse_instruction', verbosity)
        instruction.value = ''

    log(f'debug: parsed "{instruction.str}" as instruction:\n{instruction.repr}',
        'qp.qlang._parse_instruction', verbosity)

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
            log(f'trace: found symbol "{symbol}" in string "{string}"', 'qp.qlang._extract_symbol', verbosity)
            return symbol, string[len(symbol.symbol):].strip()
    
    if string.startswith(default.symbol):
        return default, string[len(default.symbol):].strip()
    else:
        log(f'trace: no symbol found in string "{string}". using default "{default}"', 'qp.qlang._extract_symbol', verbosity)
        return default, string


def _filter_series(series, negation, operator, value, verbosity, df_new=None):
    """
    Filters a pandas series according to the given instruction.
    """

    if operator in [
        OPERATORS.BIGGER_EQUAL, OPERATORS.SMALLER_EQUAL, OPERATORS.BIGGER, OPERATORS.SMALLER,
        OPERATORS.EQUAL, OPERATORS.STRICT_EQUAL,
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
            if operator == OPERATORS.EQUAL:
                value = value.lower()
                series = series.str.lower()

        elif isinstance(value, pd.Series):
            if operator == OPERATORS.EQUAL:
                if series.dtype in string_dtypes or value.dtype in string_dtypes:
                    series = series.str.lower()
                    value = value.str.lower()
                


    #numeric/date/datetime/order comparison
    if operator == OPERATORS.BIGGER_EQUAL:
        filtered = series >= value
    elif operator == OPERATORS.SMALLER_EQUAL:
        filtered = series <= value
    elif operator == OPERATORS.BIGGER:
        filtered = series > value
    elif operator == OPERATORS.SMALLER:
        filtered = series < value


    #string equality comparison
    elif operator == OPERATORS.STRICT_EQUAL:
        filtered = series == value
    elif operator == OPERATORS.EQUAL:
        filtered = series == value


    #substring comparison
    elif operator == OPERATORS.STRICT_CONTAINS:
        filtered = series.astype(str).str.contains(value, case=True, regex=False)
    elif operator == OPERATORS.CONTAINS:
        filtered = series.astype(str).str.contains(value, case=False, regex=False)


    #regex comparison
    elif operator == OPERATORS.MATCHES_REGEX:
        filtered = series.astype(str).str.fullmatch(value) 
    elif operator == OPERATORS.CONTAINS_REGEX:
        filtered = series.astype(str).str.contains(value)


    #lambda function
    elif operator == OPERATORS.EVAL:
        filtered = series.apply(lambda x: eval(value, {'x': x, 'col': series, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp}))
    elif operator == OPERATORS.COL_EVAL:
        filtered = eval(value, {'col': series, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})

    #load saved selection
    elif operator == OPERATORS.LOAD_SELECTION:
        if value in df_new.columns:
            filtered = df_new[value]
        else:
            log(f'error: column "{value}" does not exist in dataframe. cannot load selection',
                'qp.qlang._filter_series', verbosity)


    #type checks
    elif operator == OPERATORS.IS_STR:
        filtered = series.apply(lambda x: isinstance(x, str))
    elif operator == OPERATORS.IS_INT:
        filtered = series.apply(lambda x: isinstance(x, int))
    elif operator == OPERATORS.IS_FLOAT:
        filtered = series.apply(lambda x: isinstance(x, float))
    elif operator == OPERATORS.IS_NUM:
        filtered = series.apply(lambda x: _num(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_BOOL:
        filtered = series.apply(lambda x: isinstance(x, bool))

    elif operator == OPERATORS.IS_DATETIME:
        filtered = series.apply(lambda x: _datetime(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_DATE:
        filtered = series.apply(lambda x: _date(x, errors='ERROR')) != 'ERROR'

    elif operator == OPERATORS.IS_ANY:
        filtered = series.apply(lambda x: True)
    elif operator == OPERATORS.IS_NA:
        filtered = series.apply(lambda x: _na(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_NK:
        filtered = series.apply(lambda x: _nk(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_YN:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_YES:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR', yes=1)) == 1
    elif operator == OPERATORS.IS_NO:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR', no=0)) == 0
        
    elif operator == OPERATORS.IS_UNIQUE:
        filtered = series.duplicated(keep=False) == False
    elif operator == OPERATORS.IS_FIRST:
        filtered = series.duplicated(keep='first') == False
    elif operator == OPERATORS.IS_LAST:
        filtered = series.duplicated(keep='last') == False

    else:
        log(f'error: operator "{operator}" is not implemented', 'qp.qlang._filter_series', verbosity)
        filtered = None


    if negation == NEGATIONS.TRUE:
        filtered = ~filtered

    return filtered


def _update_selection(values, values_new, connector):
    """
    Updates the previously selected rows or columns based on the new selection.
    """
    if values is None:
        values = values_new
    elif connector == CONNECTORS.RESET:
        values = values_new
    elif connector in [CONNECTORS.AND, SCOPES.ALL]:
        values &= values_new
    elif connector in [CONNECTORS.OR, SCOPES.ANY]:
        values |= values_new
    return values




@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    A wrapper for the qp.query function implemented as a dataframe accessor.
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
    Wrapper for qp.qlang.query for interactive use in Jupyter notebooks.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self):
        kwargs = {'df': fixed(self.df), 'code': ''}

        #code input
        ui_code = widgets.Textarea(
            value='´s verbosity=3\n´s diff=None\n\n',
            placeholder='Enter query code here',
            layout=Layout(width='99%', height='97%')
            )


        #query builder

        instruction = INSTRUCTIONS.SELECT_COLS

        i_type = widgets.Dropdown(
            options=[(s.description, s.symbol) for s in INSTRUCTIONS],
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
            instruction = INSTRUCTIONS[i_type.value]

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
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='adds the selected instruction to the query code',
            icon='check' # (FontAwesome names without the `fa-` prefix)
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
            <b>shape:</b> {self.df.shape}<br>
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
                widgets.HTML(value=query.__doc__.replace('\n', '<br>').replace('    ', '&emsp;')),
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


