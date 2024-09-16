
import re
import numpy as np
import pandas as pd
import qplib as qp

from IPython.display import display
from ipywidgets import widgets, interactive_output, HBox, VBox, fixed, Layout

from .util import log
from .types import _int, _float, _num, _bool, _datetime, _date, _na, _nk, _yn, qpDict
from .pd_util import _check_df, _diff, _format_df, indexQpExtension, seriesQpExtension, dfQpExtension




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
        return f'{self.name} (symbol: "{self.symbol}" description: "{self.description})"'
    
    def __str__(self):
        return f'{self.name} (symbol: "{self.symbol}" description: "{self.description})"'

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


class ChangeSettings:
    """
    An instruction to change the settings for the query.
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.CHANGE_SETTINGS
        self.connector = CONNECTORS.RESET
        self.operator = OPERATORS.SET_VERBOSITY
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.operators = [OPERATORS.SET_VERBOSITY, OPERATORS.SET_DIFF, OPERATORS.SET_INPLACE]
        self.verbosity = verbosity

    def __repr__(self):
        return f'CHANGE_SETTINGS:\n\tconnector: {self.connector}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self, verbosity=None):
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.ChangeSettings.parse', self.verbosity)
        
    def apply(self, query_obj):
        operator = self.operator
        value = self.value

        if operator == OPERATORS.SET_VERBOSITY:
            if value in ['0', '1', '2', '3', '4', '5']:
                query_obj.verbosity = int(value)
                for instruction in query_obj.instructions:
                    instruction.verbosity = query_obj.verbosity
            else:
                log(f'warning: verbosity must be an integer between 0 and 5. "{value}" is not valid',
                    'qp.qlang.ChangeSettings.apply', query_obj.verbosity)
        
        elif operator == OPERATORS.SET_DIFF:
            if value.lower() in ['none', '0']:
                query_obj.diff = None
            elif value.lower() in ['mix', 'new', 'old', 'new+']:
                query_obj.diff = value.lower()
            else:
                log(f'warning: diff must be one of [None, "mix", "old", "new", "new+"]. "{value}" is not valid',
                    'qp.qlang.ChangeSettings.apply', query_obj.verbosity)
                
        elif operator == OPERATORS.SET_INPLACE:
            if value.lower() in ['true', '1', 'yes']:
                query_obj.inplace = True
                query_obj.df = query_obj.df_og 
                query_obj.df.qp = query_obj.df_og.qp

            elif value.lower() in ['false', '0', 'no']:
                query_obj.inplace = False
                query_obj.df = None
            else:
                log(f'warning: inplace must be a boolean. "{value}" is not valid',
                    'qp.qlang.ChangeSettings.apply', query_obj.verbosity)
    

class SelectCols:
    """
    An Instruction to select columns fulfilling a condition.
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.SELECT_COLS
        self.connector = CONNECTORS.RESET
        self.negation = NEGATION.FALSE
        self.operator = OPERATORS.EQUAL
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.negations = [NEGATION.TRUE]
        self.operators = [
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
            ]
        self.verbosity = verbosity
             
    def __repr__(self):
        return f'SELECT_COLS:\n\tconnector: {self.connector}\n\tnegation: {self.negation}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self):
        #parse the expression
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.negation, text = _extract_symbol(text, self.negation, self.negations, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        if self.operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                'qp.qlang.SelectCols.parse', self.verbosity)
            self.value = ''

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.SelectCols.parse', self.verbosity)

    def apply(self, query_obj):
        if query_obj.df is None:
            df = query_obj.df_og
        else:
            df = query_obj.df

        cols = df.columns.to_series()

        cols_filtered_new = _filter_series(query_obj, cols, instruction=self)

        if cols_filtered_new.any() == False:
            log(f'warning: no columns fulfill the condition in "{self.text}"',
                'qp.qlang.SelectCols.apply', self.verbosity)

        query_obj.cols_filtered = _update_selection(query_obj.cols_filtered, cols_filtered_new, self.connector)


class SelectRows:
    """
    An Instruction to select rows fulfilling a condition.
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.SELECT_ROWS
        self.connector = CONNECTORS.RESET
        self.scope = SCOPE.ANY  #only for rows
        self.negation = NEGATION.FALSE
        self.operator = OPERATORS.EQUAL
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.scopes = [SCOPE.ANY, SCOPE.ALL, SCOPE.INDEX]  #only for rows
        self.negations = [NEGATION.TRUE]
        self.operators = [
            #binary
            OPERATORS.BIGGER_EQUAL, OPERATORS.SMALLER_EQUAL, OPERATORS.BIGGER, OPERATORS.SMALLER,
            OPERATORS.STRICT_EQUAL, OPERATORS.EQUAL,
            OPERATORS.STRICT_CONTAINS, OPERATORS.CONTAINS,
            OPERATORS.MATCHES_REGEX, OPERATORS.CONTAINS_REGEX,
            OPERATORS.EVAL, OPERATORS.COL_EVAL,  #only for rows
            OPERATORS.LOAD_SELECTION,
        
            #unary
            OPERATORS.IS_ANY,
            OPERATORS.IS_UNIQUE, OPERATORS.IS_FIRST, OPERATORS.IS_LAST,
            OPERATORS.IS_NA, OPERATORS.IS_NK,
            OPERATORS.IS_STR, OPERATORS.IS_INT, OPERATORS.IS_FLOAT, OPERATORS.IS_NUM, OPERATORS.IS_BOOL,
            OPERATORS.IS_DATE, OPERATORS.IS_DATETIME,
            OPERATORS.IS_YN, OPERATORS.IS_YES, OPERATORS.IS_NO,
            ]
        self.verbosity = verbosity
             
    def __repr__(self):
        return f'SELECT_ROWS:\n\tconnector: {self.connector}\n\tscope: {self.scope}\n\tnegation: {self.negation}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self):
        #parse the expression
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.scope, text = _extract_symbol(text, self.scope, self.scopes, self.verbosity)
        self.negation, text = _extract_symbol(text, self.negation, self.negations, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        if self.operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                'qp.qlang.SelectRows.parse', self.verbosity)
            self.value = ''

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.SelectRows.parse', self.verbosity)

    def apply(self, query_obj):
        if query_obj.df is None:
            df = query_obj.df_og
        else:
            df = query_obj.df

        #select rows using parsed expression
        scope = self.scope
        connector = self.connector
        verbosity = query_obj.verbosity

        rows = df.index.to_series()
        cols_filtered = query_obj.cols_filtered

        if cols_filtered.any() == False:
            log(f'warning: row filter cannot be applied when no columns where selected', 'qp.qlang.SelectRows.apply', verbosity)
            return
                
        if scope == SCOPE.INDEX:
            rows_filtered_new = _filter_series(query_obj, rows, instruction=self)
            query_obj.rows_filtered = _update_selection(query_obj.rows_filtered, rows_filtered_new, connector)

        else:
            rows_filtered_temp = None
            for col in df.columns[cols_filtered]:
                rows_filtered_new = _filter_series(query_obj, df[col], instruction=self)
                rows_filtered_temp = _update_selection(rows_filtered_temp, rows_filtered_new, scope)
            query_obj.rows_filtered = _update_selection(query_obj.rows_filtered, rows_filtered_temp, connector)

            if rows_filtered_temp.any() == False:
                log(f'warning: no rows fulfill the condition in "{self.text}"', 'qp.qlang.SelectRows.apply', verbosity)


class ModifyVals:
    """
    An Instruction to modify the selected values.
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.MODIFY_VALS
        self.connector = CONNECTORS.RESET
        self.operator = OPERATORS.SET_VAL
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.operators = [
            OPERATORS.SET_VAL, OPERATORS.ADD_VAL,
            OPERATORS.SET_EVAL, OPERATORS.SET_COL_EVAL,
            OPERATORS.TO_STR, OPERATORS.TO_INT, OPERATORS.TO_FLOAT, OPERATORS.TO_NUM, OPERATORS.TO_BOOL,
            OPERATORS.TO_DATE, OPERATORS.TO_DATETIME, OPERATORS.TO_NA, OPERATORS.TO_NK, OPERATORS.TO_YN,
            ]
        self.verbosity = verbosity

    def __repr__(self):
        return f'MODIFY_VALS:\n\tconnector: {self.connector}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self):
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        if self.operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                'qp.qlang.ModifyVals.parse', self.verbosity)
            self.value = ''

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.ModifyVals.parse', self.verbosity)
        
    def apply(self, query_obj):
        if query_obj.df is None:
            query_obj.df = query_obj.df_og.copy()  #default is inplace=False
            query_obj.df.qp = query_obj.df.qp
        
        rows = query_obj.rows_filtered
        cols = query_obj.cols_filtered

        operator = self.operator
        value = self.value

        #data modification  
        if operator == OPERATORS.SET_VAL:
            query_obj.df.loc[rows, cols] = value
        elif operator == OPERATORS.ADD_VAL:
            query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].astype(str) + value
        elif operator == OPERATORS.SET_COL_EVAL:
            query_obj.df.loc[:, cols] = query_obj.df.loc[:, cols].apply(lambda x: eval(value, {'col': x, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp, 're': re}), axis=0)


        elif pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
            #data modification
            if operator == OPERATORS.SET_EVAL:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(lambda x: eval(value, {'x': x, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp, 're': re}))


            #type conversion
            elif operator == OPERATORS.TO_STR:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(str)
            elif operator == OPERATORS.TO_INT:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_int)
            elif operator == OPERATORS.TO_FLOAT:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_float)
            elif operator == OPERATORS.TO_NUM:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_num)
            elif operator == OPERATORS.TO_BOOL:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_bool)
            
            elif operator == OPERATORS.TO_DATETIME:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_datetime)
            elif operator == OPERATORS.TO_DATE:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_date)

            elif operator == OPERATORS.TO_NA:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_na)
            elif operator == OPERATORS.TO_NK:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_nk)
            elif operator == OPERATORS.TO_YN:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].map(_yn)

        else:
            #data modification
            if operator == OPERATORS.SET_EVAL:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(lambda x: eval(value, {'x': x, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp, 're': re}))

            #type conversion
            elif operator == OPERATORS.TO_STR:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(str)
            elif operator == OPERATORS.TO_INT:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_int)
            elif operator == OPERATORS.TO_FLOAT:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_float)
            elif operator == OPERATORS.TO_NUM:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_num)
            elif operator == OPERATORS.TO_BOOL:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_bool)
            
            elif operator == OPERATORS.TO_DATETIME:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_datetime)
            elif operator == OPERATORS.TO_DATE:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_date)

            elif operator == OPERATORS.TO_NA:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_na)
            elif operator == OPERATORS.TO_NK:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_nk)
            elif operator == OPERATORS.TO_YN:
                query_obj.df.loc[rows, cols] = query_obj.df.loc[rows, cols].applymap(_yn)


class ModifyHeaders:
    """
    An Instruction to modify the headers of the selected column(s).
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.MODIFY_HEADERS
        self.connector = CONNECTORS.RESET
        self.operator = OPERATORS.SET_VAL
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.operators = [
            OPERATORS.SET_VAL,
            OPERATORS.ADD_VAL,
            OPERATORS.SET_EVAL,
            ]
        self.verbosity = verbosity

    def __repr__(self):
        return f'MODIFY_HEADERS:\n\tconnector: {self.connector}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self):
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        if self.operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                'qp.qlang.ModifyHeaders.parse', self.verbosity)
            self.value = ''

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.ModifyHeaders.parse', self.verbosity)
        
    def apply(self, query_obj):
        if query_obj.df is None:
            query_obj.df = query_obj.df_og.copy()  #default is inplace=False
            query_obj.df.qp = query_obj.df.qp
        
        cols = query_obj.cols_filtered

        operator = self.operator
        value = self.value


        if operator == OPERATORS.SET_VAL:
            query_obj.df.rename(columns={col: value for col in query_obj.df.columns[cols]}, inplace=True)
            query_obj.cols_filtered.index = query_obj.df.columns

        if operator == OPERATORS.ADD_VAL:
            query_obj.df.rename(columns={col: col + value for col in query_obj.df.columns[cols]}, inplace=True)
            query_obj.cols_filtered.index = query_obj.df.columns

        if operator == OPERATORS.SET_EVAL:
            query_obj.df.rename(columns={
                col: eval(value, {'x': col, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp})
                for col in query_obj.df.columns[cols]}, inplace=True)
            query_obj.cols_filtered.index = query_obj.df.columns
        


class NewCol:
    """
    An Instruction to add a new column.
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.NEW_COL
        self.connector = CONNECTORS.RESET
        self.operator = OPERATORS.STR_COL
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.operators = [
            OPERATORS.STR_COL,
            OPERATORS.EVAL_COL,
            OPERATORS.SAVE_SELECTION,
            ]
        self.verbosity = verbosity
    
    def __repr__(self):
        return f'NEW_COL:\n\tconnector: {self.connector}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self):
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        if self.operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                'qp.qlang.NewCol.parse', self.verbosity)
            self.value = ''

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.NewCol.parse', self.verbosity)
        
    def apply(self, query_obj):
        if query_obj.df is None:
            query_obj.df = query_obj.df_og.copy()  #default is inplace=False
            query_obj.df.qp = query_obj.df.qp
        
        if self.operator == OPERATORS.STR_COL:
            for i in range(1, 1001):
                if i == 1000:
                    log(f'warning: could not add new column. too many columns named "new<x>"',
                        'qp.qlang.NewCol.apply', query_obj.verbosity)
                    break

                header = 'new' + str(i)
                if header not in query_obj.df.columns:
                    query_obj.df[header] = ''
                    query_obj.df.loc[query_obj.rows_filtered, header] = self.value
                    query_obj.cols_filtered = pd.Series([True if col == header else False for col in query_obj.df.columns])
                    query_obj.cols_filtered.index = query_obj.df.columns
                    break
        
        elif self.operator == OPERATORS.EVAL_COL:
            for i in range(1, 1001):
                if i == 1000:
                    log(f'warning: could not add new column. too many columns named "new<x>"',
                        'qp.qlang.NewCol.apply', query_obj.verbosity)
                    break

                header = 'new' + str(i)
                if header not in query_obj.df.columns: 
                    value = eval(self.value, {'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp})
                    if isinstance(value, pd.Series):
                        query_obj.df[header] = value
                    else:
                        query_obj.df[header] = pd.NA
                        query_obj.df.loc[query_obj.rows_filtered, header] = value
                    query_obj.cols_filtered = pd.Series([True if col == header else False for col in query_obj.df.columns])
                    query_obj.cols_filtered.index = query_obj.df.columns
                    break
        

        elif self.operator == OPERATORS.SAVE_SELECTION:
            if self.value in query_obj.df.columns:
                log(f'warning: column "{self.value}" already exists in dataframe. selecting existing col and resetting values',
                    'qp.qlang.NewCol.apply', query_obj.verbosity)
            query_obj.df[self.value] = query_obj.rows_filtered
            query_obj.cols_filtered = pd.Series([True if col == self.value else False for col in query_obj.df.columns])
            query_obj.cols_filtered.index = query_obj.df.columns


class Miscellaneous:
    """
    An Instruction for miscellaneous tasks, for example, modifying metadata.
    """

    def __init__(self, text=None, linenum=None, verbosity=3):
        self.text = text
        self.linenum = linenum

        #default values
        self.type = TYPES.MISCELLANEOUS
        self.connector = CONNECTORS.RESET
        self.operator = OPERATORS.SET_METADATA
        self.value = ''

        #possible values (omitting those without a symbol)
        self.connectors = [CONNECTORS.AND, CONNECTORS.OR]
        self.operators = [
            OPERATORS.SET_METADATA,
            OPERATORS.ADD_METADATA,
            OPERATORS.TAG_METADATA,
            OPERATORS.SET_METADATA_EVAL,
            OPERATORS.SET_METADATA_COL_EVAL,
            ]
        self.verbosity = verbosity
    
    def __repr__(self):
        return f'NEW_COL:\n\tconnector: {self.connector}\n\toperator: {self.operator}\n\tvalue: {self.value}'
    
    def parse(self):
        self.connector, text = _extract_symbol(self.text[2:], self.connector, self.connectors, self.verbosity)
        self.operator, text = _extract_symbol(text, self.operator, self.operators, self.verbosity)
        self.value = text.strip()

        if self.operator.unary and len(self.value)>0:
            log(f'warning: unary operator "{self.operator}" cannot use a value. value "{self.value}" will be ignored',
                'qp.qlang.Miscellaneous.parse', self.verbosity)
            self.value = ''

        log(f'debug: parsed "{self.text}" as instruction: {self}',
            'qp.qlang.Miscellaneous.parse', self.verbosity)
        
    def apply(self, query_obj):
        if query_obj.df is None:
            query_obj.df = query_obj.df_og.copy()  #default is inplace=False
            query_obj.df.qp = query_obj.df.qp
        
        rows = query_obj.rows_filtered
        cols = query_obj.cols_filtered

        operators_metadata = [
            OPERATORS.SET_METADATA,
            OPERATORS.ADD_METADATA,
            OPERATORS.TAG_METADATA,
            OPERATORS.SET_METADATA_EVAL,
            OPERATORS.SET_METADATA_COL_EVAL,
            ]
        if self.operator in operators_metadata and 'meta' not in query_obj.df.columns:
            log(f'info: no metadata column found in dataframe. creating new column named "meta',
                'qp.qlang.Miscellaneous.apply', query_obj.verbosity)
            query_obj.df['meta'] = ''
            query_obj.cols_filtered = pd.concat([query_obj.cols_filtered, pd.Series([False])])
            query_obj.cols_filtered.index = query_obj.df.columns
            cols = query_obj.cols_filtered


        #modify metadata
        if self.operator == OPERATORS.SET_METADATA:
            query_obj.df.loc[rows, 'meta'] = self.value

        elif self.operator == OPERATORS.ADD_METADATA:
            query_obj.df.loc[rows, 'meta'] += self.value

        elif self.operator == OPERATORS.TAG_METADATA:
            tag = ''
            for col in query_obj.df.columns[cols]:
                tag += f'{self.value}@{col};'
            query_obj.df.loc[rows, 'meta'] += tag

        elif self.operator == OPERATORS.SET_METADATA_EVAL:
            if pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
                query_obj.df.loc[rows, 'meta'] = query_obj.df.loc[rows, 'meta'].map(lambda x: eval(self.value, {'x': x, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:
                query_obj.df.loc[rows, 'meta'] = query_obj.df.loc[rows, 'meta'].applymap(lambda x: eval(self.value, {'x': x, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            
        elif self.operator == OPERATORS.SET_METADATA_COL_EVAL:
            query_obj.df.loc[rows, 'meta'] = query_obj.df.loc[rows, 'meta'].apply(lambda x: eval(self.value, {'col': x, 'df': query_obj.df, 'pd': pd, 'np': np, 'qp': qp, 're': re}))

        






COMMENT = Symbol('#', 'COMMENT', 'comments out the rest of the line')
ESCAPE = Symbol('`', 'ESCAPE', 'escape the next character')


TYPES = Symbols('TYPES',
    Symbol('´s', 'CHANGE_SETTINGS', 'change query settings', instruction=ChangeSettings),
    Symbol('´c', 'SELECT_COLS', 'select columns fulfilling a condition', instruction=SelectCols),
    Symbol('´r', 'SELECT_ROWS', 'select rows fulfilling a condition', instruction=SelectRows),
    Symbol('´v', 'MODIFY_VALS', 'modify the selected values', instruction=ModifyVals),
    Symbol('´h', 'MODIFY_HEADERS', 'modify headers of the selected columns', instruction=ModifyHeaders),
    Symbol('´n', 'NEW_COL', 'add new column', instruction=NewCol),
    Symbol('´m', 'MISCELLANEOUS', 'miscellaneous instructions', instruction=Miscellaneous),
    )

CONNECTORS = Symbols('CONNECTORS',
    Symbol('', 'RESET', 'only the current condition must be fulfilled'),
    Symbol('&', 'AND', 'this condition and the previous condition/s must be fulfilled'),
    Symbol('/', 'OR', 'this condition or the previous condition/s must be fulfilled'),
    )

SCOPE = Symbols('SCOPE',
    Symbol('any', 'ANY', 'any of the currently selected columns must fulfill the condition'),
    Symbol('all', 'ALL', 'all of the currently selected columns must fulfill the condition'),
    Symbol('idx', 'INDEX', 'the index of the dataframe must fulfill the condition'),
    )

NEGATION = Symbols('NEGATION',
    Symbol('', 'FALSE', 'dont negate the condition'),
    Symbol('!', 'TRUE', 'negate the condition'),
    )

OPERATORS = Symbols('OPERATORS',
    #for changing settings
    Symbol('verbosity=', 'SET_VERBOSITY', 'change the verbosity level'),
    Symbol('diff=', 'SET_DIFF', 'change the diff setting'),
    Symbol('inplace=', 'SET_INPLACE', 'change the inplace setting'),


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



def _tokenize(code, verbosity=3):
    """
    Turns the plain text input string into a list of instructions for the query parser.
    """

    lines = []
    instructions = []

    #get lines and instruction blocks
    for line_num, line in enumerate(code.split('\n')):
        line = line.strip()
        lines.append([line_num, line])
        line = line.split(COMMENT.symbol)[0].strip()
    
        if line == '':
            continue


        escape = False
        chars_in_instruction = 0
        instruction_type = TYPES.SELECT_COLS.symbol  #default

        for i, char in enumerate(line):
            if escape:
                instructions[-1].text += char
                chars_in_instruction += 1
                escape = False
                continue
            elif char == ESCAPE.symbol:
                escape = True
                continue

            if char == '´':
                instruction_type = char + line[i+1]
                instructions.append(TYPES[instruction_type].instruction(char, line_num, verbosity))
                chars_in_instruction = 1
            elif char in [CONNECTORS.AND.symbol, CONNECTORS.OR.symbol]:
                if chars_in_instruction >= 3:
                    instructions.append(TYPES[instruction_type].instruction(f'{instruction_type} {char}', line_num, verbosity))
                    chars_in_instruction = 3
                elif i == 0:
                    instructions.append(TYPES[instruction_type].instruction(f'{instruction_type} {char}', line_num, verbosity))
                    chars_in_instruction = 3
                else:
                    instructions[-1].text += char
                    chars_in_instruction += 1
            elif i == 0:
                instructions.append(TYPES[instruction_type].instruction(f'{instruction_type} {char}', line_num, verbosity))
                chars_in_instruction = 3
            elif char == ' ':
                instructions[-1].text += char
            else:
                instructions[-1].text += char
                chars_in_instruction += 1

    return lines, instructions


def _extract_symbol(string, default, symbols, verbosity):
    """
    Looks for expected syntax symbols at the beginning of an instruction string.
    """

    string = string.strip()

    for symbol in symbols:
        if string.startswith(symbol.symbol):
            log(f'trace: found symbol "{symbol}" in string "{string}"', 'qp.qlang._extract_symbol', verbosity)
            return symbol, string[len(symbol.symbol):].strip()
    
    log(f'trace: no symbol found in string "{string}". using default "{default}"', 'qp.qlang._extract_symbol', verbosity)
    
    if default is None:
        return None, string
    if string.startswith(default.symbol):
        string = string[len(default.symbol):].strip()
    return default, string


def _filter_series(query_obj, series, instruction):
    """
    Filters a pandas series according to the given instruction.
    """

    negation = instruction.negation
    operator = instruction.operator
    value = instruction.value
    verbosity = instruction.verbosity
    df = query_obj.df


    #numeric comparison
    if operator == OPERATORS.BIGGER_EQUAL:
        filtered = pd.to_numeric(series, errors='coerce') >= pd.to_numeric(value)
    elif operator == OPERATORS.SMALLER_EQUAL:
        filtered = pd.to_numeric(series, errors='coerce') <= pd.to_numeric(value)
    elif operator == OPERATORS.BIGGER:
        filtered = pd.to_numeric(series, errors='coerce') > pd.to_numeric(value)
    elif operator == OPERATORS.SMALLER:
        filtered = pd.to_numeric(series, errors='coerce') < pd.to_numeric(value)


    #string equality comparison
    elif operator == OPERATORS.STRICT_EQUAL:
        filtered = series.astype(str) == value
    elif operator == OPERATORS.EQUAL:
        value_lenient = [value]
        try:
            value_lenient.append(str(float(value)))
            value_lenient.append(str(int(float(value))))
        except:
            value_lenient.append(value.lower())
        filtered = series.astype(str).str.lower().isin(value_lenient)


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
        filtered = series.apply(lambda x: eval(value, {'x': x, 'col': series, 'df': df, 'pd': pd, 'np': np, 'qp': qp}))
    elif operator == OPERATORS.COL_EVAL:
        filtered = eval(value, {'col': series, 'df': df, 'pd': pd, 'np': np, 'qp': qp})

    #load saved selection
    elif operator == OPERATORS.LOAD_SELECTION:
        if value in df.columns:
            filtered = df[value]
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


    if negation == NEGATION.TRUE:
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
    elif connector in [CONNECTORS.AND, SCOPE.ALL]:
        values &= values_new
    elif connector in [CONNECTORS.OR, SCOPE.ANY]:
        values |= values_new
    return values


@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    A query language for pandas data exploration/analysis/modification.

    examples:
    df.q('id')  #selects the column 'id'
    df.q('id  ´r > 100)  #selects col "id" and rows where the value is greater than 100
    df.q('´c = id  ´r > 100) #same as above but more explicit
    df.q('id  ´r > 100  ´c / name  ´r ? john')  #selects col "id" and
        #rows where the value is greater than 100 or col "name" and rows where the value contains "john"
    """

    def __init__(self, df: pd.DataFrame):
        _check_df(df)
        self.df_og = df

    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():\n' + self.__doc__
    
    def __call__(self,
            code='',  #code in string form for filtering and modifying data
            inplace=False,  #make modifications inplace or just return a new dataframe.
            verbosity=3,  #verbosity level for logging. 0: no logging, 1: errors, 2: warnings, 3: info, 4: debug
            diff=None,  #[None, 'mix', 'old', 'new', 'new+']
            diff_max_cols=200,  #maximum number of columns to display when using diff. None: show all
            diff_max_rows=20,  #maximum number of rows to display when using diff. None: show all
            ):
        
        #setup

        self.code = code
        self.inplace = inplace
        self.verbosity = verbosity
        self.diff = diff
        self.diff_max_cols = diff_max_cols
        self.diff_max_rows = diff_max_rows

        if inplace is False:
            self.df = None
        else:
            self.df = self.df_og 
            self.df.qp = self.df_og.qp 

        self.cols_filtered = pd.Series([True for col in self.df_og.columns])
        self.rows_filtered = pd.Series([True for row in self.df_og.index])
        self.cols_filtered.index = self.df_og.columns
        self.rows_filtered.index = self.df_og.index



        #instructions

        self.lines, self.instructions = _tokenize(self.code, self.verbosity)

        for instruction in self.instructions:
            instruction.parse()
            instruction.apply(self)

   
        #results
        if self.df is None:
            df = self.df_og
        else:
            df = self.df

        self.df_filtered = df.loc[self.rows_filtered, self.cols_filtered]
        self.df_filtered.qp = df.qp
        self.df_filtered.qp.code = self.code
    
        if self.diff is None:
            return self.df_filtered 
        else:
            #show difference before and after filtering

            if 'meta' in df.columns and 'meta' not in self.df_filtered.columns:
                self.df_filtered.insert(0, 'meta', df.loc[self.rows_filtered, 'meta'])

            result = _diff(
                self.df_filtered, df, mode=self.diff,
                max_cols=self.diff_max_cols, max_rows=self.diff_max_rows,
                verbosity=self.verbosity)  
            return  result  
   


@pd.api.extensions.register_dataframe_accessor('qi')
class DataFrameQueryInteractiveMode:
    """
    Wrapper for df.q() for interactive use in Jupyter notebooks.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self):
        kwargs = {'df': fixed(self.df)}

        #code input
        ui_code = widgets.Textarea(
            value='',
            placeholder='Enter query code here',
            layout=Layout(height='95%')
            )


        #query builder

        instruction = TYPES.SELECT_COLS.instruction()

        i_type = widgets.Dropdown(
            options=[(s.description, s.symbol) for s in TYPES],
            value=instruction.type.symbol,
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
            value=instruction.operator.symbol,
            )
        
        i_value = widgets.Text(
            value='',
            )
        

        i_text = widgets.Text(
            value=f'\n{i_type.value} {i_scope.value} {i_negate.value}{i_operator.value} {i_value.value}',
            disabled=True,
            )
        

        def update_options(*args):
            instruction = TYPES[i_type.value].instruction()

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
            i_operator.value = instruction.operator.symbol

        def update_text(*args):
            i_text.value = f'{i_type.value} {i_scope.value} {i_negate.value}{i_operator.value} {i_value.value}'

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
        
        
        #some general info and statistics about the df
        ui_details = widgets.HTML(
            value=f"""
            <b>shape:</b> {self.df.shape}<br>
            <b>memory usage:</b> {self.df.memory_usage().sum()} bytes<br>
            <b>unique values:</b> {self.df.nunique().sum()}<br>
            <b>missing values:</b> {self.df.isna().sum().sum()}<br>
            <b>columns:</b><br> {'<br>'.join([f'{col} ({dtype})' for col, dtype in list(zip(self.df.columns, self.df.dtypes))])}<br>
            """
            )
        

        ui_info = widgets.Tab(
            children=[
                ui_settings,
                ui_details,
                widgets.HTML(value=DataFrameQuery.__doc__),
                ],
            titles=['settings', 'details', 'readme'],
            layout=Layout(width='30%', height='95%')
            )
        

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
        
        # ui_input = HBox([ui_code, ui_instruction_builder], layout=Layout(width='50%', height='100%'))
        ui = HBox([ui_code, ui_input, ui_info], layout=Layout(width='100%', height='300px'))

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
    return result 







