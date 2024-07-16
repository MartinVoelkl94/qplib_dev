
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







class Symbol:
    def __init__(self, symbol, description, unary=False, binary=False, function=None, options=None):
        #default symbol = None
        self.symbol = symbol
        self.description = description
        self.unary = unary
        self.binary = binary
        self.function = function
        self.options = options
    def __repr__(self):
        return f"Symbol('{self.symbol}': '{self.description})'"

class Symbols:   
    COMMENT = Symbol('#', 'comments out the rest of the line')
    ESCAPE = Symbol('`', 'escape the next character')



    #there are 4 different types of instructions
    SETTINGS = Symbol('´s', 'settings for the query', function='settings')
    SELECT_COLS = Symbol('´c', 'default. select columns by a condition', function='select_cols')
    SELECT_ROWS = Symbol('´r', 'select rows by a condition', function='select_rows')
    MODIFY_VALS = Symbol('´m', 'modify the currently selected values', function='modify_vals')
    NEW_COL = Symbol('´n', 'add a new column', function='new_col')
    TYPES = [SETTINGS, SELECT_COLS, SELECT_ROWS, MODIFY_VALS, NEW_COL]


    #select conditions can be combined with the previous conditions
    CONNECTOR_RESET = Symbol('', 'default. only the current condition must be fulfilled')
    CONNECTOR_AND =   Symbol('&', 'this condition and the previous condition/s must be fulfilled')
    CONNECTOR_OR =    Symbol('/', 'this condition or the previous condition/s must be fulfilled')

    #when filtering rows, what should the condition apply to
    COLS_ANY = Symbol('any', 'default. any of the currently selected columns must fulfill the condition')
    COLS_ALL = Symbol('all', 'all of the currently selected columns must fulfill the condition')
    INDEX = Symbol('idx', 'the index of the dataframe must fulfill the condition')

    #negation of selection conditions
    NEGATE_FALSE = Symbol('', 'default. dont negate the condition')
    NEGATE_TRUE = Symbol('!', 'negate the condition')



    #binary operators for filtering

    OP_BIGGER_EQUAL = Symbol('>=', 'bigger or equal', binary=True)
    OP_SMALLER_EQUAL = Symbol('<=', 'smaller or equal', binary=True)
    OP_BIGGER = Symbol('>', 'bigger', binary=True)
    OP_SMALLER = Symbol('<', 'smaller', binary=True)

    OP_STRICT_EQUAL = Symbol('==', 'strict equal', binary=True)
    OP_EQUAL = Symbol('=', 'default. equal', binary=True)

    OP_REGEX_MATCH = Symbol('~~', 'regex match', binary=True)
    OP_REGEX_SEARCH = Symbol('~', 'regex search', binary=True)

    OP_STRICT_CONTAINS = Symbol('(())', 'strict contains', binary=True)
    OP_CONTAINS = Symbol('()', 'contains', binary=True)

    OP_X_EVAL = Symbol('x?', 'select values by evaluating a python expression on each value', binary=True)
    OP_COL_EVAL = Symbol('col?', 'select rows by evaluating a python expression on a whole column', binary=True)
    
    OP_LOAD = Symbol('§', 'load a saved selection', binary=True)


    #unary operators for filtering

    OP_IS_STR = Symbol('is str', 'is string', unary=True)
    OP_IS_INT = Symbol('is int', 'is integer', unary=True)
    OP_IS_FLOAT = Symbol('is float', 'is float', unary=True)
    OP_IS_NUM = Symbol('is num', 'is number', unary=True)
    OP_IS_BOOL = Symbol('is bool', 'is boolean', unary=True)

    OP_IS_DATETIME = Symbol('is datetime', 'is datetime', unary=True)
    OP_IS_DATE = Symbol('is date', 'is date', unary=True)

    OP_IS_ANY = Symbol('is any', 'is any value', unary=True)
    OP_IS_NA = Symbol('is na', 'is missing value', unary=True)
    OP_IS_NK = Symbol('is nk', 'is not known value', unary=True)
    OP_IS_YN = Symbol('is yn', 'is yes or no value', unary=True)
    OP_IS_YES = Symbol('is yes', 'is yes value', unary=True)
    OP_IS_NO = Symbol('is no', 'is no value', unary=True)



    #binary operators for modifying values
    OP_SET_VAL = Symbol('=', 'default. change currently selected values to the provided string', binary=True)
    OP_ADD_VAL = Symbol('+=', 'add str to currently selected values (they are coerced to string)', binary=True)

    OP_SET_X_EVAL = Symbol('x=', 'change values by evaluating a python expression for each currently selected value', binary=True)
    OP_SET_COL_EVAL = Symbol('col=', 'change values by evaluating a python expression for each currently selected column', binary=True)
    OP_SET_HEADER_EVAL = Symbol('header=', 'change values by evaluating a python expression for each currently selected headers', binary=True)

    #unary operators for modifying values
    OP_TO_STR = Symbol('to str', 'convert currently selected values to string', unary=True)
    OP_TO_INT = Symbol('to int', 'convert currently selected values to integer', unary=True)
    OP_TO_FLOAT = Symbol('to float', 'convert currently selected values to float', unary=True)
    OP_TO_NUM = Symbol('to num', 'convert currently selected values to number', unary=True)
    OP_TO_BOOL = Symbol('to bool', 'convert currently selected values to boolean', unary=True)

    OP_TO_DATETIME = Symbol('to datetime', 'convert currently selected values to datetime', unary=True)
    OP_TO_DATE = Symbol('to date', 'convert currently selected values to date', unary=True)

    OP_TO_NA = Symbol('to na', 'convert currently selected values to missing value', unary=True)
    OP_TO_NK = Symbol('to nk', 'convert currently selected values to not known value', unary=True)
    OP_TO_YN = Symbol('to yn', 'convert currently selected values to yes or no value', unary=True)


    #binary operators for modifying the whole dataframe
    OP_NEW_COL_STR = Symbol('=', 'add a new string column to the dataframe and select it instead of current selection', binary=True)
    OP_NEW_COL_TRUE = Symbol('§', 'add a new boolean column and select it. all currently selected rows are set to True', binary=True)

class Expression:
    def __init__(self, text, function, line_num):
        self.text = text
        self.function = function
        self.line_num = line_num
        self.frepr = f'Expression({self.text})'
    def __repr__(self):
        return f'Instruction({self.__dict__})'


@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    """

    def __init__(self, df: pd.DataFrame):
        _check_df(df)
        self.df_og = df
        self.symbols = Symbols()

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
        
        #setup

        self.code = code
        self.inplace = inplace
        self.verbosity = verbosity
        self.diff = diff
        self.diff_max_cols = diff_max_cols
        self.diff_max_rows = diff_max_rows

        if inplace is False:
            self.df = self.df_og.copy()
        else:
            self.df = self.df_og  
        self.df.qp = self.df_og.qp 

        self.cols_filtered = pd.Index([True for col in self.df.columns])
        self.rows_filtered = pd.Index([True for row in self.df.index])



        #main

        self.parse(self.code)

        for expression in self.expressions:
            expression.symbol = expression.text[:2]
            expression.type, text_temp = self.match_symbol(expression.text, None, self.symbols.TYPES)
            expression.function = expression.type.function
            if expression.type is None:
                log(f'warning: no defined behaviour for "{expression.symbol}"', 'df.q()', self.verbosity)
                continue
    
            self.expression = expression

            #parse and evaluate the expression
            self.__getattribute__(expression.function)()



        #results

        self.df_filtered = self.df.loc[self.rows_filtered, self.cols_filtered]
        self.df_filtered.qp = self.df.qp
        self.df_filtered.qp.code = self.code
    
        if self.diff is None:
            return self.df_filtered 
        else:
            #show difference before and after filtering

            if 'meta' in self.df.columns and 'meta' not in self.df_filtered.columns:
                self.df_filtered.insert(0, 'meta', self.df.loc[self.rows_filtered, 'meta'])

            result = _show_differences(
                self.df_filtered, self.df, show=self.diff,
                max_cols=self.diff_max_cols, max_rows=self.diff_max_rows,
                verbosity=self.verbosity)  
            return  result  
   

    def parse(self, code):
        self.lines = []
        self.expressions = []
        self.instructions = []

        #get lines and expression blocks
        for line_num, line in enumerate(self.code.split('\n')):
            line = line.strip()
            self.lines.append([line_num, line])
            line = line.split(self.symbols.COMMENT.symbol)[0].strip()
        
            if line == '':
                continue


            AND = self.symbols.CONNECTOR_AND.symbol
            OR = self.symbols.CONNECTOR_OR.symbol

            escape = False
            expression_type = self.symbols.SELECT_COLS.symbol  #default
            for i, char in enumerate(line):
                if escape:
                    self.expressions[-1].text += char
                    escape = False
                    continue
                elif char == self.symbols.ESCAPE.symbol:
                    escape = True
                    continue

                if char == '´':
                    expression_type = char + line[i+1]
                    self.expressions.append(Expression(char, None, line_num))
                elif char in [AND, OR]:
                    self.expressions.append(Expression(f'{expression_type} {char}', None, line_num))
                elif i == 0:
                    self.expressions.append(Expression(f'{expression_type} {char}', None, line_num))
                else:
                    self.expressions[-1].text += char

    def match_symbol(self, string, default, symbols):
        string = string.strip()

        for symbol in symbols:
            if string.startswith(symbol.symbol):
                log(f'trace: found symbol "{symbol}" in string "{string}"', '_read_symbol', self.verbosity)
                return symbol, string[len(symbol.symbol):].strip()
        
        log(f'trace: no symbol found in string "{string}". using default "{default}"', '_read_symbol', self.verbosity)
        
        if default is None:
            return None, string
        if string.startswith(default.symbol):
            string = string[len(default.symbol):].strip()
        return default, string


    def settings(self):

        text = self.expression.text[2:].strip()

        setting = None
        for setting_temp in ['verbosity', 'diff', 'diff_max_cols', 'diff_max_rows']:
            if text.startswith(setting_temp):
                setting = setting_temp
                text = text[len(setting_temp):].strip()
                break
        
        if setting is None:
            log(f'warning: no setting found in "{self.expression.text}"', 'df.q.settings()', self.verbosity)
            return
        
        if text[0] != '=':
            log(f'warning: settings can only be set with "=" not with "{text[0]}"', 'df.q.settings()', self.verbosity)
            return
        else:
            value = text[1:].strip()


        if setting == 'verbosity':
            if value in ['0', '1', '2', '3', '4', '5']:
                self.verbosity = int(value)
            else:
                log(f'warning: verbosity must be an integer between 0 and 5. "{value}" is not valid',
                    'df.q.settings()', self.verbosity)
        
        elif setting == 'diff':
            if value in ['none', 'None', 'NONE', '0']:
                self.diff = None
            elif value.lower() in ['mix', 'new', 'old', 'new+']:
                self.diff = value.lower()
            else:
                log(f'warning: diff must be one of [None, "mix", "old", "new", "new+"]. "{value}" is not valid',
                    'df.q.settings()', self.verbosity)
                
        elif setting == 'diff_max_cols':
            if value in ['none', 'None', 'NONE', '0']:
                self.diff_max_cols = None
            try:
                self.diff_max_cols = int(value)
            except:
                log(f'warning: diff_max_cols must be an integer or None. "{value}" is not valid',
                    'df.q.settings()', self.verbosity)
                
        elif setting == 'diff_max_rows':
            if value in ['none', 'None', 'NONE', '0']:
                self.diff_max_rows = None
            try:
                self.diff_max_rows = int(value)
            except:
                log(f'warning: diff_max_rows must be an integer or None. "{value}" is not valid',
                    'df.q.settings()', self.verbosity)


    def select_cols(self):

        #parse the expression

        connector, text = self.match_symbol(
            self.expression.text[2:],
            default=self.symbols.CONNECTOR_RESET,
            symbols=[
                self.symbols.CONNECTOR_AND,
                self.symbols.CONNECTOR_OR
                ]
            )

     
        negate, text = self.match_symbol(
            text,
            default=self.symbols.NEGATE_FALSE,
            symbols=[self.symbols.NEGATE_TRUE]
            )
        
        operator, text = self.match_symbol(
            text,
            default=self.symbols.OP_EQUAL,
            symbols=[
                self.symbols.OP_BIGGER_EQUAL,
                self.symbols.OP_SMALLER_EQUAL,
                self.symbols.OP_BIGGER,
                self.symbols.OP_SMALLER,
                self.symbols.OP_STRICT_EQUAL,
                self.symbols.OP_EQUAL,
                self.symbols.OP_REGEX_MATCH,
                self.symbols.OP_REGEX_SEARCH,
                self.symbols.OP_STRICT_CONTAINS,
                self.symbols.OP_CONTAINS,
                self.symbols.OP_X_EVAL,
                self.symbols.OP_COL_EVAL,
                self.symbols.OP_IS_STR,
                self.symbols.OP_IS_INT,
                self.symbols.OP_IS_FLOAT,
                self.symbols.OP_IS_NUM,
                self.symbols.OP_IS_BOOL,
                self.symbols.OP_IS_DATE,
                self.symbols.OP_IS_DATETIME,
                self.symbols.OP_IS_ANY,
                self.symbols.OP_IS_NA,
                self.symbols.OP_IS_NK,
                self.symbols.OP_IS_YN,
                self.symbols.OP_IS_YES,
                self.symbols.OP_IS_NO,
                ],
            )

        value = text.strip()

        if operator.unary and len(value)>0:
            log(f'warning: unary operator "{operator}" cannot use a value. value "{value}" will be ignored',
                '_parse_expression', self.verbosity)
            value = ''
    
        self.expression.connector = connector
        self.expression.negate = negate
        self.expression.operator = operator
        self.expression.value = value
        self.expression.frepr = f"""
            <br>type: {self.expression.type}
            <br>text: {self.expression.text}
            <br>function: {self.expression.function}
            <br>connector: {self.expression.connector.symbol}
            <br>negate: {self.expression.negate.symbol}
            <br>operator: {self.expression.operator.symbol}
            <br>value: {self.expression.value}
            """

        log(f'debug: parsed "{self.expression.text}" as expression: {self.expression.frepr}',
            'df.q.select_cols()', self.verbosity)


        #select cols using parsed expression
    
        cols_filtered_new = self._filter_series(self.df.columns.to_series())
        self.cols_filtered = self._update_index(self.cols_filtered, cols_filtered_new, connector)

        if cols_filtered_new.any() == False:
            log(f'warning: no columns fulfill the condition in "{self.expression.text}"',
                'df.q.select_cols()', self.verbosity)


    def select_rows(self):

        #parse the expression

        connector, text = self.match_symbol(
            self.expression.text[2:],
            default=self.symbols.CONNECTOR_RESET,
            symbols=[
                self.symbols.CONNECTOR_AND,
                self.symbols.CONNECTOR_OR
                ]
            )
        
        select_which, text = self.match_symbol(
            text,
            default=self.symbols.COLS_ANY,
            symbols=[
                self.symbols.COLS_ANY,
                self.symbols.COLS_ALL,
                self.symbols.INDEX
                ]
            )

     
        negate, text = self.match_symbol(
            text,
            default=self.symbols.NEGATE_FALSE,
            symbols=[self.symbols.NEGATE_TRUE]
            )
        
        operator, text = self.match_symbol(
            text,
            default=self.symbols.OP_EQUAL,
            symbols=[
                self.symbols.OP_BIGGER_EQUAL,
                self.symbols.OP_SMALLER_EQUAL,
                self.symbols.OP_BIGGER,
                self.symbols.OP_SMALLER,
                self.symbols.OP_STRICT_EQUAL,
                self.symbols.OP_EQUAL,
                self.symbols.OP_REGEX_MATCH,
                self.symbols.OP_REGEX_SEARCH,
                self.symbols.OP_STRICT_CONTAINS,
                self.symbols.OP_CONTAINS,
                self.symbols.OP_LOAD,
                self.symbols.OP_X_EVAL,
                self.symbols.OP_COL_EVAL,
                self.symbols.OP_IS_STR,
                self.symbols.OP_IS_INT,
                self.symbols.OP_IS_FLOAT,
                self.symbols.OP_IS_NUM,
                self.symbols.OP_IS_BOOL,
                self.symbols.OP_IS_DATE,
                self.symbols.OP_IS_DATETIME,
                self.symbols.OP_IS_ANY,
                self.symbols.OP_IS_NA,
                self.symbols.OP_IS_NK,
                self.symbols.OP_IS_YN,
                self.symbols.OP_IS_YES,
                self.symbols.OP_IS_NO,
                ],
            )

        value = text.strip()

        if operator.unary and len(value)>0:
            log(f'warning: unary operator "{operator}" cannot use a value. value "{value}" will be ignored',
                '_parse_expression', self.verbosity)
            value = ''
    
        self.expression.connector = connector
        self.expression.select_which = select_which
        self.expression.negate = negate
        self.expression.operator = operator
        self.expression.value = value
        self.expression.frepr = f"""
            <br>type: {self.expression.type}
            <br>text: {self.expression.text}
            <br>function: {self.expression.function}
            <br>connector: {self.expression.connector.symbol}
            <br>negate: {self.expression.negate.symbol}
            <br>operator: {self.expression.operator.symbol}
            <br>value: {self.expression.value}
            """

        log(f'debug: parsed "{self.expression.text}" as expression: {self.expression.frepr}',
            'df.q.select_cols()', self.verbosity)


        #select rows using parsed expression

        if self.cols_filtered.any() == False:
            log(f'warning: row filter cannot be applied when no columns where selected', 'df.q', self.verbosity)
            return
                
        if select_which == self.symbols.INDEX:
            rows_filtered_new = self._filter_series(self.df.index.to_series())
            self.rows_filtered = self._update_index(self.rows_filtered, rows_filtered_new, connector)

        else:
            rows_filtered_temp = None
            for col in self.df.columns[self.cols_filtered]:
                rows_filtered_new = self._filter_series(self.df[col])
                rows_filtered_temp = self._update_index(rows_filtered_temp, rows_filtered_new, select_which)
            self.rows_filtered = self._update_index(self.rows_filtered, rows_filtered_temp, connector)

            if rows_filtered_temp.any() == False:
                log(f'warning: no rows fulfill the condition in "{self.expression.text}"', 'df.q', self.verbosity)
            

    def _filter_series(self, series):
        operator = self.expression.operator
        value = self.expression.value
        ops = self.symbols


        if operator == ops.OP_BIGGER_EQUAL:
            filtered = pd.to_numeric(series, errors='coerce') >= pd.to_numeric(value)
        elif operator == ops.OP_SMALLER_EQUAL:
            filtered = pd.to_numeric(series, errors='coerce') <= pd.to_numeric(value)
        elif operator == ops.OP_BIGGER:
            filtered = pd.to_numeric(series, errors='coerce') > pd.to_numeric(value)
        elif operator == ops.OP_SMALLER:
            filtered = pd.to_numeric(series, errors='coerce') < pd.to_numeric(value)
            
            
        #regex comparison
        elif operator == ops.OP_REGEX_MATCH:
            filtered = series.astype(str).str.fullmatch(value) 
        elif operator == ops.OP_REGEX_SEARCH:
            filtered = series.astype(str).str.contains(value)


        #string equality comparison
        elif operator == ops.OP_STRICT_EQUAL:
            filtered = series.astype(str) == value
        elif operator == ops.OP_EQUAL:
            value_lenient = [value]
            try:
                value_lenient.append(str(float(value)))
                value_lenient.append(str(int(float(value))))
            except:
                value_lenient.append(value.lower())
            filtered = series.astype(str).str.lower().isin(value_lenient)
            
        #substring comparison
        elif operator == ops.OP_STRICT_CONTAINS:
            filtered = series.astype(str).str.contains(value, case=True, regex=False)
        elif operator == ops.OP_CONTAINS:
            filtered = series.astype(str).str.contains(value, case=False, regex=False)


        #lambda function
        elif operator == ops.OP_X_EVAL:
            filtered = series.apply(lambda x: eval(value, {'x': x, 'col': series, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}))
        elif operator == ops.OP_COL_EVAL:
            filtered = eval(value, {'col': series, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp})

        #load saved selection
        elif operator == ops.OP_LOAD:
            if value in self.df.columns:
                filtered = self.df[value]
            else:
                log(f'error: column "{value}" does not exist in dataframe. cannot load selection',
                    '_filter()', self.verbosity)


        #type checks
        elif operator == ops.OP_IS_STR:
            filtered = series.apply(lambda x: isinstance(x, str))
        elif operator == ops.OP_IS_INT:
            filtered = series.apply(lambda x: isinstance(x, int))
        elif operator == ops.OP_IS_FLOAT:
            filtered = series.apply(lambda x: isinstance(x, float))
        elif operator == ops.OP_IS_NUM:
            filtered = series.apply(lambda x: qp_num(x, errors='ERROR')) != 'ERROR'
        elif operator == ops.OP_IS_BOOL:
            filtered = series.apply(lambda x: isinstance(x, bool))

        elif operator == ops.OP_IS_DATETIME:
            filtered = series.apply(lambda x: qp_datetime(x, errors='ERROR')) != 'ERROR'
        elif operator == ops.OP_IS_DATE:
            filtered = series.apply(lambda x: qp_date(x, errors='ERROR')) != 'ERROR'

        elif operator == ops.OP_IS_ANY:
            filtered = series.apply(lambda x: True)
        elif operator == ops.OP_IS_NA:
            filtered = series.apply(lambda x: qp_na(x, errors='ERROR')) != 'ERROR'
        elif operator == ops.OP_IS_NK:
            filtered = series.apply(lambda x: qp_nk(x, errors='ERROR')) != 'ERROR'
        elif operator == ops.OP_IS_YN:
            filtered = series.apply(lambda x: qp_yn(x, errors='ERROR')) != 'ERROR'
        elif operator == ops.OP_IS_YES:
            filtered = series.apply(lambda x: qp_yn(x, errors='ERROR', yes=1)) == 1
        elif operator == ops.OP_IS_NO:
            filtered = series.apply(lambda x: qp_yn(x, errors='ERROR', no=0)) == 0

        else:
            log(f'error: operator "{operator}" is not implemented', '_filter()', verbosity)
            filtered = None


        if self.expression.negate == ops.NEGATE_TRUE:
            filtered = ~filtered

        return filtered

    def _update_index(self, values, values_new, connector):
        if values is None:
            values = values_new
        elif connector == self.symbols.CONNECTOR_RESET:
            values = values_new
        elif connector in [self.symbols.CONNECTOR_AND, self.symbols.COLS_ALL]:
            values &= values_new
        elif connector in [self.symbols.CONNECTOR_OR, self.symbols.COLS_ANY]:
            values |= values_new
        return values


    def modify_vals(self):

        #parse the expression

        connector, text = self.match_symbol(
            self.expression.text[2:],
            default=self.symbols.CONNECTOR_RESET,
            symbols=[]
            )

        operator, text = self.match_symbol(
            text,
            default=self.symbols.OP_SET_VAL,
            symbols=[
                self.symbols.OP_SET_VAL,
                self.symbols.OP_ADD_VAL,
                self.symbols.OP_SET_X_EVAL,
                self.symbols.OP_SET_COL_EVAL,
                self.symbols.OP_SET_HEADER_EVAL,
                self.symbols.OP_TO_STR,
                self.symbols.OP_TO_INT,
                self.symbols.OP_TO_FLOAT,
                self.symbols.OP_TO_NUM,
                self.symbols.OP_TO_BOOL,
                self.symbols.OP_TO_DATE,
                self.symbols.OP_TO_DATETIME,
                self.symbols.OP_TO_NA,
                self.symbols.OP_TO_NK,
                self.symbols.OP_TO_YN,
                ],
            )

        value = text.strip()

        if operator.unary and len(value)>0:
            log(f'warning: unary operator "{operator}" cannot use a value. value "{value}" will be ignored',
                'df.q.modify_vals()', self.verbosity)
            value = ''
    
        self.expression.connector = connector
        self.expression.operator = operator
        self.expression.value = value
        self.expression.frepr = f"""
            <br>type: {self.expression.type}
            <br>text: {self.expression.text}
            <br>function: {self.expression.function}
            <br>connector: {self.expression.connector.symbol}
            <br>operator: {self.expression.operator.symbol}
            <br>value: {self.expression.value}
            """

        log(f'debug: parsed "{self.expression.text}" as expression: {self.expression.frepr}',
            'df.q.modify_vals()', self.verbosity)


        #modify values using parsed expression
        ops = self.symbols
        rows = self.rows_filtered
        cols = self.cols_filtered
   
        if pd.__version__ >= '2.1.0':
            #data modification
            if operator == ops.OP_SET_VAL:
                self.df.loc[rows, cols] = value
            elif operator == ops.OP_ADD_VAL:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].astype(str) + value
            
            elif operator == ops.OP_SET_X_EVAL:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(lambda x: eval(value, {'x': x, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}))
            elif operator == ops.OP_SET_COL_EVAL:
                self.df.loc[:, cols] = self.df.loc[:, cols].apply(lambda x: eval(value, {'col': x, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}), axis=0)
            elif operator == ops.OP_SET_HEADER_EVAL:
                self.df.columns = self.df.columns.map(lambda x: eval(value, {'header': x, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}))


            #type conversion
            elif operator == ops.OP_TO_STR:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(str)
            elif operator == ops.OP_TO_INT:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_int)
            elif operator == ops.OP_TO_FLOAT:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_float)
            elif operator == ops.OP_TO_NUM:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_num)
            elif operator == ops.OP_TO_BOOL:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_bool)
            
            elif operator == ops.OP_TO_DATETIME:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_datetime)
            elif operator == ops.OP_TO_DATE:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_date)

            elif operator == ops.OP_TO_NA:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_na)
            elif operator == ops.OP_TO_NK:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_nk)
            elif operator == ops.OP_TO_YN:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].map(qp_yn)

        else:
            #data modification
            if operator == ops.OP_SET_VAL:
                self.df.loc[rows, cols] = value
            elif operator == ops.OP_ADD_VAL:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].astype(str) + value
            
            elif operator == ops.OP_SET_X_EVAL:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(lambda x: eval(value, {'x': x, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}))
            elif operator == ops.OP_SET_COL_EVAL:
                self.df.loc[:, cols] = self.df.loc[:, cols].apply(lambda x: eval(value, {'col': x, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}), axis=0)
            elif operator == ops.OP_SET_HEADER_EVAL:
                self.df.columns = self.df.columns.applymap(lambda x: eval(value, {'header': x, 'df': self.df, 'pd': pd, 'np': np, 'qp': qp}))


            #type conversion
            elif operator == ops.OP_TO_STR:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(str)
            elif operator == ops.OP_TO_INT:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_int)
            elif operator == ops.OP_TO_FLOAT:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_float)
            elif operator == ops.OP_TO_NUM:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_num)
            elif operator == ops.OP_TO_BOOL:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_bool)
            
            elif operator == ops.OP_TO_DATETIME:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_datetime)
            elif operator == ops.OP_TO_DATE:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_date)

            elif operator == ops.OP_TO_NA:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_na)
            elif operator == ops.OP_TO_NK:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_nk)
            elif operator == ops.OP_TO_YN:
                self.df.loc[rows, cols] = self.df.loc[rows, cols].applymap(qp_yn)
      

    def new_col(self):

        #parse the expression

        connector, text = self.match_symbol(
            self.expression.text[2:],
            default=self.symbols.CONNECTOR_RESET,
            symbols=[]
            )

        operator, text = self.match_symbol(
            text,
            default=self.symbols.OP_NEW_COL_STR,
            symbols=[
                self.symbols.OP_NEW_COL_STR,
                self.symbols.OP_NEW_COL_TRUE,
                ]
            )

        value = text.strip()

        if operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{operator}" cannot use a value. value "{value}" will be ignored',
                'df.q.new_col()', self.verbosity)
            value = ''
    
        self.expression.connector = connector
        self.expression.operator = operator
        self.expression.value = value
        self.expression.frepr = f"""
            <br>type: {self.expression.type}
            <br>text: {self.expression.text}
            <br>function: {self.expression.function}
            <br>connector: {self.expression.connector.symbol}
            <br>operator: {self.expression.operator.symbol}
            <br>value: {self.expression.value}
            """

        log(f'debug: parsed "{self.expression.text}" as expression: {self.expression.frepr}',
            'df.q.new_col()', self.verbosity)


        #add new column using parsed expression
        ops = self.symbols
        rows = self.rows_filtered
        cols = self.cols_filtered
        
        if operator == ops.OP_NEW_COL_STR:
            if value in self.df.columns:
                log(f'warning: column "{value}" already exists in dataframe. selecting existing col',
                    'df.q.new_col', self.verbosity)
                self.cols_filtered = pd.Index([True if col == value else False for col in self.df.columns])
            else:
                self.df[value] = ''
                self.cols_filtered = pd.Index([True if col == value else False for col in self.df.columns])
        if operator == ops.OP_NEW_COL_TRUE:
            if value in self.df.columns:
                log(f'warning: column "{value}" already exists in dataframe. selecting existing col and resetting values',
                    'df.q.new_col', self.verbosity)
                self.df[value] = self.rows_filtered
                self.cols_filtered = pd.Index([True if col == value else False for col in self.df.columns])
            else:
                self.df[value] = self.rows_filtered
                self.cols_filtered = pd.Index([True if col == value else False for col in self.df.columns])
                       



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






