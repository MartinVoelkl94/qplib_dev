
import numpy as np
import pandas as pd
import copy
import re
import qplib as qp

from IPython.display import display
from ipywidgets import widgets, interactive_output, HBox, VBox, fixed, Layout

from .util import log
from .types import qp_int, qp_float, qp_num, qp_bool, qp_datetime, qp_date, qp_na, qp_nk, qp_yn, qpDict
from .pd_util import _check_df, _show_differences, _format_df, indexQpExtension, seriesQpExtension, dfQpExtension









##################     syntax symbols     ##################

class Symbol:
    def __init__(self, symbol, description, unary=False, binary=False, modes=None):
        #default symbol = None
        self.symbol = symbol
        self.description = description
        self.unary = unary
        self.binary = binary
        self.modes = modes
    def __repr__(self):
        return f"Symbol('{self.symbol}': '{self.description})'"


COMMENT = Symbol('#', 'comment', 'comments out the rest of the line')

#there are 4 different types of instructions
FILTER_COLS = Symbol('', 'default. filter columns by a condition')
FILTER_ROWS = Symbol('´', 'filter rows by a condition')
MODIFY_VALS = Symbol('´´', 'modify the currently selected values')
MODIFY_DF = Symbol('´´´', 'ignore the current selection and modify the whole dataframe. eg for adding new cols')


#filter conditions can be combined with the previous conditions
CONNECTOR_RESET = Symbol('', 'default. only the current condition must be fulfilled')
CONNECTOR_AND =   Symbol('&&', 'this condition and the previous condition/s must be fulfilled')
CONNECTOR_OR =    Symbol('//', 'this condition or the previous condition/s must be fulfilled')

#when filtering rows, the condition can be applied to all or any of the previously selected columns
COLS_ANY = Symbol('any', 'default. any of the currently selected columns must fulfill the condition')
COLS_ALL = Symbol('all', 'all of the currently selected columns must fulfill the condition')

#negation of filter conditions
NEGATE_FALSE = Symbol('', 'default. dont negate the condition')
NEGATE_TRUE = Symbol('!', 'negate the condition')



#binary operators for filtering

OP_BIGGER_EQUAL = Symbol('>=', 'bigger or equal', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_SMALLER_EQUAL = Symbol('<=', 'smaller or equal', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_BIGGER = Symbol('>', 'bigger', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_SMALLER = Symbol('<', 'smaller', binary=True, modes=[FILTER_COLS, FILTER_ROWS])

OP_STRICT_EQUAL = Symbol('==', 'strict equal', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_EQUAL = Symbol('=', 'default. equal', binary=True, modes=[FILTER_COLS, FILTER_ROWS])

OP_REGEX_MATCH = Symbol('~~', 'regex match', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_REGEX_SEARCH = Symbol('~', 'regex search', binary=True, modes=[FILTER_COLS, FILTER_ROWS])

OP_STRICT_CONTAINS = Symbol('(())', 'strict contains', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_CONTAINS = Symbol('()', 'contains', binary=True, modes=[FILTER_COLS, FILTER_ROWS])

OP_X_EVAL = Symbol('x?', 'filter values by evaluating a python expression on each value', binary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_COL_EVAL = Symbol('col?', 'filter rows by evaluating a python expression on a whole column', binary=True, modes=[FILTER_ROWS])


#unary operators for filtering

OP_IS_STR = Symbol('is str', 'is string', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_INT = Symbol('is int', 'is integer', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_FLOAT = Symbol('is float', 'is float', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_NUM = Symbol('is num', 'is number', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_BOOL = Symbol('is bool', 'is boolean', unary=True, modes=[FILTER_COLS, FILTER_ROWS])

OP_IS_DATETIME = Symbol('is datetime', 'is datetime', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_DATE = Symbol('is date', 'is date', unary=True, modes=[FILTER_COLS, FILTER_ROWS])

OP_IS_ANY = Symbol('is any', 'is any value', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_NA = Symbol('is na', 'is missing value', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_NK = Symbol('is nk', 'is not known value', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_YN = Symbol('is yn', 'is yes or no value', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_YES = Symbol('is yes', 'is yes value', unary=True, modes=[FILTER_COLS, FILTER_ROWS])
OP_IS_NO = Symbol('is no', 'is no value', unary=True, modes=[FILTER_COLS, FILTER_ROWS])



#binary operators for modifying values
OP_SET_VAL = Symbol('=', 'default. change currently selected values to the provided string', binary=True, modes=[MODIFY_VALS])
OP_ADD_VAL = Symbol('+=', 'add str to currently selected values (they are coerced to string)', binary=True, modes=[MODIFY_VALS])

OP_SET_X_EVAL = Symbol('x=', 'change values by evaluating a python expression for each currently selected value', binary=True, modes=[MODIFY_VALS])
OP_SET_COL_EVAL = Symbol('col=', 'change values by evaluating a python expression for each currently selected column', binary=True, modes=[MODIFY_VALS])
OP_SET_HEADER_EVAL = Symbol('header=', 'change values by evaluating a python expression for each currently selected headers', binary=True, modes=[MODIFY_VALS])

#unary operators for modifying values
OP_TO_STR = Symbol('to str', 'convert currently selected values to string', unary=True, modes=[MODIFY_VALS])
OP_TO_INT = Symbol('to int', 'convert currently selected values to integer', unary=True, modes=[MODIFY_VALS])
OP_TO_FLOAT = Symbol('to float', 'convert currently selected values to float', unary=True, modes=[MODIFY_VALS])
OP_TO_NUM = Symbol('to num', 'convert currently selected values to number', unary=True, modes=[MODIFY_VALS])
OP_TO_BOOL = Symbol('to bool', 'convert currently selected values to boolean', unary=True, modes=[MODIFY_VALS])

OP_TO_DATETIME = Symbol('to datetime', 'convert currently selected values to datetime', unary=True, modes=[MODIFY_VALS])
OP_TO_DATE = Symbol('to date', 'convert currently selected values to date', unary=True, modes=[MODIFY_VALS])

OP_TO_NA = Symbol('to na', 'convert currently selected values to missing value', unary=True, modes=[MODIFY_VALS])
OP_TO_NK = Symbol('to nk', 'convert currently selected values to not known value', unary=True, modes=[MODIFY_VALS])
OP_TO_YN = Symbol('to yn', 'convert currently selected values to yes or no value', unary=True, modes=[MODIFY_VALS])


#binary operators for modifying the whole dataframe
OP_ADD_COL = Symbol('=', 'add a new string column to the dataframe and select it instead of current selection', binary=True, modes=[MODIFY_DF])



######################     parsing     ######################

def _parse_line(line, verbosity):

    instructions = []
    line = line.strip()
    if line == '' or line.startswith(COMMENT.symbol):
        return None

    line = line.split(COMMENT.symbol)[0]
    #wip: add option to escape # with \#


    strings = re.split(f'({MODIFY_DF.symbol}|{MODIFY_VALS.symbol}|{FILTER_ROWS.symbol})', line)
    for i, string in enumerate(strings):
        if string in ['', FILTER_ROWS.symbol, MODIFY_VALS.symbol, MODIFY_DF.symbol]:
            continue

        if i == 0:
            mode = ''
        elif strings[i-1] in [FILTER_ROWS.symbol, MODIFY_VALS.symbol, MODIFY_DF.symbol]:
            mode = strings[i-1]


        substrings = re.split(f'({CONNECTOR_AND.symbol}|{CONNECTOR_OR.symbol})', string)

        for j, substring in enumerate(substrings):
            substring = substring.strip()
            if substring in [CONNECTOR_AND.symbol, CONNECTOR_OR.symbol, '']:
                continue
            elif j == 0:
                connector = ''
            else:
                connector = substrings[j-1]

            token = mode + connector + substring

            if mode in ['', FILTER_ROWS.symbol]:
                instruction = InstructionFilter(token, verbosity=verbosity)
                instructions.append(instruction)
            elif mode == MODIFY_VALS.symbol:
                instruction = InstructionModifyVals(token, verbosity=verbosity)
                instructions.append(instruction)
            elif mode == MODIFY_DF.symbol:
                instruction = InstructionModifyDf(token, verbosity=verbosity)
                instructions.append(instruction)
            else:
                log(f'mode is not implemented. \n<br>line:{line} \n<br>token: {token} \n<br>mode: {mode} ',
                    level='error', source='_parse_line', input=line, verbosity=verbosity)
                
            log(f'parsed token "{token}" in line "{line}"',
                level='trace', source='_parse_line', verbosity=verbosity)


    if len(instructions) == 0:
        return None
    else:
        return instructions


class InstructionFilter:
    def __init__(self, token, verbosity=3):
        self.parse(token, verbosity=verbosity)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        string = f"""
            \n<br>{self.mode}
            \n<br>{self.connector.symbol}
            \n<br>{self.which_cols}
            \n<br>{self.negate.symbol}
            \n<br>{self.operator.symbol}
            \n<br>{self.value}
            """
        return string

    def parse(self, token, verbosity=3):
        self.token = token
        self.mode, token = _read_symbol(token, [FILTER_ROWS], FILTER_COLS, verbosity)
        self.connector, token = _read_symbol(token, [CONNECTOR_AND, CONNECTOR_OR], CONNECTOR_RESET, verbosity)

        #row conditions also specify in which cols the condition is applied
        if self.mode == FILTER_ROWS:
            self.which_cols, token = _read_symbol(token, [COLS_ALL], COLS_ANY, verbosity)
        else:
            self.which_cols = None
        
        self.negate, token = _read_symbol(token, [NEGATE_TRUE], NEGATE_FALSE, verbosity)
        self.operator, token = _read_symbol(
            token,
            symbols = [
                OP_BIGGER_EQUAL, OP_SMALLER_EQUAL, OP_BIGGER, OP_SMALLER,
                OP_STRICT_EQUAL, OP_EQUAL,
                OP_REGEX_MATCH, OP_REGEX_SEARCH,
                OP_STRICT_CONTAINS, OP_CONTAINS,
                OP_X_EVAL, OP_COL_EVAL,
                OP_IS_STR, OP_IS_INT, OP_IS_FLOAT, OP_IS_NUM, OP_IS_BOOL,
                OP_IS_DATE, OP_IS_DATETIME,
                OP_IS_ANY, OP_IS_NA, OP_IS_NK, OP_IS_YN, OP_IS_YES, OP_IS_NO,
                ],
            default = OP_EQUAL,
            verbosity = verbosity,
            )
        self.value = token.strip()


        if self.operator.unary and len(self.value)>0:
            log(f'unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                level='warning', source='_parse_token', verbosity=verbosity)
            self.value = ''
    
        log(f'parsed token "{self.token}" as instruction: {self.__str__()}',
            level='debug', source='_parse_token', verbosity=verbosity)

    def filter(self, df, cols_filtered, rows_filtered, verbosity=3):
        if self.mode == FILTER_COLS:
            cols_filtered_new = self._filter_series(df.columns.to_series(), verbosity)
            cols_filtered = _update_index(cols_filtered, cols_filtered_new, self.connector, verbosity)

            if cols_filtered_new.any() == False:
                log(f'no columns fulfill the condition in "{self.token}"',
                    level='warning', source='df.q', input=self.token, verbosity=verbosity)
            
            return cols_filtered


        elif self.mode == FILTER_ROWS:
            if cols_filtered.any() == False:
                log(f'row filter cannot be applied when no columns where selected',
                    level='warning', source='df.q', input=self.token, verbosity=verbosity)
                return rows_filtered
                

            rows_filtered_temp = None
            for i, col in enumerate(df.columns[cols_filtered]):
                rows_filtered_new = self._filter_series(df[col], verbosity, df)
                rows_filtered_temp = _update_index(rows_filtered_temp, rows_filtered_new, self.which_cols, verbosity)
            rows_filtered = _update_index(rows_filtered, rows_filtered_temp, self.connector, verbosity)

            if rows_filtered_temp.any() == False:
                log(f'no rows fulfill the condition in "{self.token}"',
                    level='warning', source='df.q', input=self.token, verbosity=verbosity)
                
            return rows_filtered

    def _filter_series(self, series, verbosity=3, df=None,):
        operator = self.operator
        value = self.value


        if operator == OP_BIGGER_EQUAL:
            filtered = pd.to_numeric(series, errors='coerce') >= pd.to_numeric(value)
        elif operator == OP_SMALLER_EQUAL:
            filtered = pd.to_numeric(series, errors='coerce') <= pd.to_numeric(value)
        elif operator == OP_BIGGER:
            filtered = pd.to_numeric(series, errors='coerce') > pd.to_numeric(value)
        elif operator == OP_SMALLER:
            filtered = pd.to_numeric(series, errors='coerce') < pd.to_numeric(value)
            
            
        #regex comparison
        elif operator == OP_REGEX_MATCH:
            filtered = series.astype(str).str.fullmatch(value) 
        elif operator == OP_REGEX_SEARCH:
            filtered = series.astype(str).str.contains(value)


        #string equality comparison
        elif operator == OP_STRICT_EQUAL:
            filtered = series.astype(str) == value
        elif operator == OP_EQUAL:
            value_lenient = [value]
            try:
                value_lenient.append(str(float(value)))
                value_lenient.append(str(int(float(value))))
            except:
                value_lenient.append(value.lower())
            filtered = series.astype(str).str.lower().isin(value_lenient)
            
        #substring comparison
        elif operator == OP_STRICT_CONTAINS:
            filtered = series.astype(str).str.contains(value, case=True, regex=False)
        elif operator == OP_CONTAINS:
            filtered = series.astype(str).str.contains(value, case=False, regex=False)


        #lambda function
        elif operator == OP_X_EVAL:
            filtered = series.apply(lambda x: eval(value, {'x': x, 'col': series, 'df': df, 'pd': pd, 'np': np, 'qp': qp}))
        elif operator == OP_COL_EVAL:
            filtered = eval(value, {'col': series, 'df': df, 'pd': pd, 'np': np, 'qp': qp})


        #type checks
        elif operator == OP_IS_STR:
            filtered = series.apply(lambda x: isinstance(x, str))
        elif operator == OP_IS_INT:
            filtered = series.apply(lambda x: isinstance(x, int))
        elif operator == OP_IS_FLOAT:
            filtered = series.apply(lambda x: isinstance(x, float))
        elif operator == OP_IS_NUM:
            filtered = series.apply(lambda x: qp_num(x, errors='ERROR')) != 'ERROR'
        elif operator == OP_IS_BOOL:
            filtered = series.apply(lambda x: isinstance(x, bool))

        elif operator == OP_IS_DATETIME:
            filtered = series.apply(lambda x: qp_datetime(x, errors='ERROR')) != 'ERROR'
        elif operator == OP_IS_DATE:
            filtered = series.apply(lambda x: qp_date(x, errors='ERROR')) != 'ERROR'

        elif operator == OP_IS_ANY:
            filtered = series.apply(lambda x: True)
        elif operator == OP_IS_NA:
            filtered = series.apply(lambda x: qp_na(x, errors='ERROR')) != 'ERROR'
        elif operator == OP_IS_NK:
            filtered = series.apply(lambda x: qp_nk(x, errors='ERROR')) != 'ERROR'
        elif operator == OP_IS_YN:
            filtered = series.apply(lambda x: qp_yn(x, errors='ERROR')) != 'ERROR'
        elif operator == OP_IS_YES:
            filtered = series.apply(lambda x: qp_yn(x, errors='ERROR', yes=1)) == 1
        elif operator == OP_IS_NO:
            filtered = series.apply(lambda x: qp_yn(x, errors='ERROR', no=0)) == 0

        else:
            log(f'operator "{operator}" is not implemented',
                level='error', source='_filter()', input=series.qp._input, verbosity=verbosity)
            filtered = None


        if self.negate == NEGATE_TRUE:
            filtered = ~filtered

        return filtered


class InstructionModifyVals:
    def __init__(self, token, verbosity=3):
        self.parse(token, verbosity=verbosity)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        string = f"""
            \n<br>{self.mode}
            \n<br>{self.connector.symbol}
            \n<br>{self.operator.symbol}
            \n<br>{self.value}
            """
        return string


    def parse(self, token, verbosity=3):
        self.token = token
        self.mode, token = _read_symbol(token, [MODIFY_VALS], MODIFY_VALS, verbosity)
        self.connector, token = _read_symbol(token, [CONNECTOR_AND], CONNECTOR_AND, verbosity)
        self.operator, token = _read_symbol(
            token,
            symbols = [
                OP_SET_VAL, OP_ADD_VAL,
                OP_SET_X_EVAL, OP_SET_COL_EVAL, OP_SET_HEADER_EVAL,
                OP_TO_STR, OP_TO_INT, OP_TO_FLOAT, OP_TO_NUM, OP_TO_BOOL,
                OP_TO_DATETIME, OP_TO_DATE,
                OP_TO_NA, OP_TO_NK, OP_TO_YN,
                ],
            default = OP_SET_VAL,
            verbosity = verbosity,
            )
        self.value = token.strip()
    
        log(f'parsed token "{self.token}" as instruction: {self.__str__()}',
            level='debug', source='_parse_token', verbosity=verbosity)

        if self.operator.unary and len(self.value)>0:
            log(f'unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                level='warning', source='_parse_token', verbosity=verbosity)
            self.value = ''


    def modify_vals(self, df, cols_filtered, rows_filtered, verbosity=3):
        operator = self.operator
        value = self.value
        cols = cols_filtered
        rows = rows_filtered

        
        if pd.__version__ >= '2.1.0':
            #data modification
            if operator == OP_SET_VAL:
                df.loc[rows, cols] = value
            elif operator == OP_ADD_VAL:
                df.loc[rows, cols] = df.loc[rows, cols].astype(str) + value
            
            elif operator == OP_SET_X_EVAL:
                df.loc[rows, cols] = df.loc[rows, cols].map(lambda x: eval(value, {'x': x, 'df': df, 'pd': pd, 'np': np, 'qp': qp}))
            elif operator == OP_SET_COL_EVAL:
                df.loc[:, cols] = df.loc[:, cols].apply(lambda x: eval(value, {'col': x, 'df': df, 'pd': pd, 'np': np, 'qp': qp}), axis=0)
            elif operator == OP_SET_HEADER_EVAL:
                df.columns = df.columns.map(lambda x: eval(value, {'header': x, 'df': df, 'pd': pd, 'np': np, 'qp': qp}))


            #type conversion
            elif operator == OP_TO_STR:
                df.loc[rows, cols] = df.loc[rows, cols].map(str)
            elif operator == OP_TO_INT:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_int)
            elif operator == OP_TO_FLOAT:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_float)
            elif operator == OP_TO_NUM:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_num)
            elif operator == OP_TO_BOOL:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_bool)
            
            elif operator == OP_TO_DATETIME:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_datetime)
            elif operator == OP_TO_DATE:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_date)

            elif operator == OP_TO_NA:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_na)
            elif operator == OP_TO_NK:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_nk)
            elif operator == OP_TO_YN:
                df.loc[rows, cols] = df.loc[rows, cols].map(qp_yn)

        else:
            #data modification
            if operator == OP_SET_VAL:
                df.loc[rows, cols] = value
            elif operator == OP_ADD_VAL:
                df.loc[rows, cols] = df.loc[rows, cols].astype(str) + value
            
            elif operator == OP_SET_X_EVAL:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(lambda x: eval(value, {'x': x, 'df': df, 'pd': pd, 'np': np, 'qp': qp}))
            elif operator == OP_SET_COL_EVAL:
                df.loc[:, cols] = df.loc[:, cols].apply(lambda x: eval(value, {'col': x, 'df': df, 'pd': pd, 'np': np, 'qp': qp}), axis=0)
            elif operator == OP_SET_HEADER_EVAL:
                df.columns = df.columns.applymap(lambda x: eval(value, {'header': x, 'df': df, 'pd': pd, 'np': np, 'qp': qp}))


            #type conversion
            elif operator == OP_TO_STR:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(str)
            elif operator == OP_TO_INT:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_int)
            elif operator == OP_TO_FLOAT:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_float)
            elif operator == OP_TO_NUM:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_num)
            elif operator == OP_TO_BOOL:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_bool)
            
            elif operator == OP_TO_DATETIME:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_datetime)
            elif operator == OP_TO_DATE:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_date)

            elif operator == OP_TO_NA:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_na)
            elif operator == OP_TO_NK:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_nk)
            elif operator == OP_TO_YN:
                df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_yn)

        return df


class InstructionModifyDf:
    def __init__(self, token, verbosity=3):
        self.parse(token, verbosity=verbosity)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        string = f"""
            \n<br>{self.mode}
            \n<br>{self.connector.symbol}
            \n<br>{self.operator.symbol}
            \n<br>{self.value}
            """
        return string


    def parse(self, token, verbosity=3):
        self.token = token
        self.mode, token = _read_symbol(token, [MODIFY_DF], MODIFY_DF, verbosity)
        self.connector, token = _read_symbol(token, [CONNECTOR_AND], CONNECTOR_AND, verbosity)
        self.operator, token = _read_symbol(
            token,
            symbols = [
                OP_ADD_COL,
                ],
            default = OP_ADD_COL,
            verbosity = verbosity,
            )
        self.value = token.strip()
    
        log(f'parsed token "{self.token}" as instruction: {self.__str__()}',
            level='debug', source='_parse_token', verbosity=verbosity)

        if self.operator.unary and len(self.value)>0:
            log(f'unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                level='warning', source='_parse_token', verbosity=verbosity)
            self.value = ''


def _read_symbol(token, symbols, default, verbosity=3):
    """
    Reads the first characters of a token and returns the corresponding symbol name and the remaining token.
    Keys in symbol_dictionary which are None are used as default values.
    """

    for symbol in symbols:
        if token.startswith(symbol.symbol):
            log(f'found symbol "{symbol}" in token "{token}"',
                level='trace', source='_read_symbol', verbosity=verbosity)
            return symbol, token[len(symbol.symbol):].strip()
    
    log(f'no symbol found in token "{token}". using default "{default}"',
        level='trace', source='_read_symbol', verbosity=verbosity)
    if token.startswith(default.symbol):
        token = token[len(default.symbol):].strip()
    return default, token

def _update_index(values, values_new, connector, verbosity=3):
    if values is None:
        values = values_new
    elif connector == CONNECTOR_RESET:
        values = values_new
    elif connector in [CONNECTOR_AND, COLS_ALL]:
        values &= values_new
    elif connector in [CONNECTOR_OR, COLS_ANY]:
        values |= values_new
    return values



#####################     query api     #####################

@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    """


    def __init__(self, df: pd.DataFrame):
        _check_df(df)
        self.df = df


    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():' + self.__doc__
    

    def __call__(self,
            code='',  #code in string form for filtering and modifying data
            inplace=True,  #make modifications inplace or just return a new dataframe.
            verbosity=3,  #verbosity level for logging. 0: no logging, 1: errors, 2: warnings, 3: info, 4: debug
            diff=None,  #[None, 'mix', 'old', 'new', 'new+']
            diff_max_cols=200,  #maximum number of columns to display when using diff. None: show all
            diff_max_rows=20,  #maximum number of rows to display when using diff. None: show all
            **kwargs
            ):

        #######################     setup     #######################

        #input string for logging
        code_str = code.replace('\n', '\n\t')
        input_str = f".q(\n\tr\"\"\"\n\t{code_str}\n\t\"\"\","
   
        if diff is not None:
            input_str += f"\n\tdiff='{diff}',"
        # if max_cols is not None:
        #     input_str += f"\n\tmax_cols={max_cols},"
        # if max_rows is not None:
        #     input_str += f"\n\tmax_rows={max_rows},"
        
        input_str += f"\n\tinplace={inplace},"
        input_str += f"\n\tverbosity={verbosity},"

        for kwarg in kwargs:
            input_str += f"\n\t{kwarg}='{kwargs[kwarg]}'"

        self.df.qp._input = input_str + "\n\t)"

                
        if inplace is False:
            df = self.df.copy()
        else:
            df = self.df  
        df.qp = self.df.qp 



        #######################     main     #######################

        cols_filtered = pd.Index([True for col in df.columns])
        rows_filtered = pd.Index([True for row in df.index])

        instructions = []
        for line in code.split('\n'):
            instructions_temp = _parse_line(line, verbosity=verbosity)
            if instructions_temp is None:
                continue
            instructions += instructions_temp
        
        if len(instructions) == 0:
            log(f'no instructions found in code "{code}"',
                level='warning', source='df.q', verbosity=verbosity)
            return df



        for instruction in instructions:
            
            if instruction.mode == FILTER_COLS:
                cols_filtered = instruction.filter(df, cols_filtered, rows_filtered, verbosity=verbosity)

            elif instruction.mode == FILTER_ROWS:
                rows_filtered = instruction.filter(df, cols_filtered, rows_filtered, verbosity=verbosity)

            elif instruction.mode == MODIFY_VALS:
                df.loc[:, :] = instruction.modify_vals(df, cols_filtered, rows_filtered, verbosity=verbosity).loc[:, :]

            elif instruction.mode == MODIFY_DF:
                if instruction.operator == OP_ADD_COL:
                    if instruction.value in df.columns:
                        log(f'column "{instruction.value}" already exists in dataframe. selecting col instead of creating a new one',
                            level='warning', source='df.q', verbosity=verbosity)
                        cols_filtered = pd.Index([True if col == instruction.value else False for col in df.columns])
                    else:
                        df[instruction.value] = ''
                        cols_filtered = pd.Index([True if col == instruction.value else False for col in df.columns])
                        


        #################     display settings     #################

        df_filtered = df.loc[rows_filtered, cols_filtered]
        df_filtered.qp = self.df.qp
    
        if diff is None:
            return df_filtered 
        else:
            #show difference before and after filtering

            if 'meta' in df.columns and 'meta' not in df_filtered.columns:
                df_filtered.insert(0, 'meta', df.loc[rows_filtered, 'meta'])

            result = _show_differences(
                df_filtered, self.df, show=diff,
                max_cols=diff_max_cols, max_rows=diff_max_rows,
                verbosity=verbosity)  
            return  result
    



#################     interactive mode     #################

@pd.api.extensions.register_dataframe_accessor('qi')
class DataFrameQueryInteractiveMode:
    """
    Wrapper for df.q() for interactive use in Jupyter notebooks.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, num_filters=5):
        kwargs = {'df': fixed(self.df)}

        #code input
        ui_code = widgets.Textarea(
            value='',
            placeholder='Enter query code here',
            layout=Layout(width='95%', height='100%')
        )
        kwargs['code'] = ui_code



        ui_diff = widgets.ToggleButtons(
            options=[None, 'mix', 'old', 'new', 'new+'],
            description='show differences mode:',
            tooltips=[
                'dont show differences, just show the new (filtered) dataframe.',
                'show new (filtered) dataframe plus all the removed (filtered) values from the old dataframe. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'show old (unfiltered) dataframe. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'show new (filtered) dataframe. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'show new (filtered) dataframe but also adds metadata columns with the prefix "#". If a value changed, the metadata column contains the old value. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                ],
            )
        kwargs['diff'] = ui_diff

        ui_verbosity = widgets.ToggleButtons(
            options=[0, 1, 2, 3, 4, 5],
            value=3,
            description='verbosity level:',
            tooltips=[
                'no logging',
                'only errors',
                'errors and warnings',
                'errors, warnings and info',
                'errors, warnings, info and debug',
                'errors, warnings, info, debug and trace',
                ],
            )
        kwargs['verbosity'] = ui_verbosity

        ui_inplace = widgets.ToggleButtons(
            options=[True, False],
            value=False,
            description='make modifications inplace:',
            tooltips=[
                'make modifications inplace, e.g. change the original dataframe.',
                'return a new dataframe with the modifications. lower performance.',
                ],
            )
        kwargs['inplace'] = ui_inplace


        ui_settings = VBox([
            ui_diff,
            ui_verbosity,
            ui_inplace,
            ])
        

        ui_help = widgets.Tab(
            children=[
                ui_settings,
                widgets.HTML(value="syntax"),
                widgets.HTML(value="operators"),
                widgets.HTML(value="modifiers"),
                ],
            titles=['settings', 'syntax', 'operators', 'modifiers'],
            layout=Layout(width='50%', height='95%')
            )
        

        ui_input = VBox([ui_code], layout=Layout(width='50%', height='100%'))
        ui = HBox([ui_input, ui_help], layout=Layout(width='100%', height='300px'))

        display(ui)
        out = HBox([interactive_output(_interactive_mode, kwargs)], layout=Layout(overflow_y='auto'))
        display(out)


def _interactive_mode(**kwargs):

    df = kwargs.pop('df')

    result = df.q(
        code=kwargs['code'],
        inplace=kwargs['inplace'],
        diff=kwargs['diff'],
        verbosity=kwargs['verbosity'],
        # max_cols=kwargs['max_cols'],
        # max_rows=kwargs['max_rows'],
        )


    
    display(result)
    print('input code: ', df.qp._input)
    return result 







