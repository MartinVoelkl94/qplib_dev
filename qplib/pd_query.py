
import numpy as np
import pandas as pd
import copy
import re
import qplib as qp

from IPython.display import display
from ipywidgets import widgets, interactive_output, HBox, VBox, fixed, Layout

from .util import log, qpDict
from .pd_util import _check_df

from .types import qp_int
from .types import qp_float
from .types import qp_num
from .types import qp_bool
from .types import qp_date
from .types import qp_datetime
from .types import qp_na
from .types import qp_nk
from .types import qp_yn

from .pd_util import indexQpExtension, seriesQpExtension, dfQpExtension





_operators1 = qpDict({
    #need a value for comparison or modification
    'bigger_equal': '>=',
    'smaller_equal': '<=',
    'bigger': '>',
    'smaller': '<',
    'strict_equal': '==',
    'equal': '=',
    'regex_match': '~~',
    'regex_search': '~',
    'strict_contains': '(())',
    'contains': '()',

    'lambda_condition': '?',
    'lambda_condition_col': 'col?',
    })
_operators2 = qpDict({
    #these dont need a values for comparison or modification
    'is_str': 'is str',
    'is_int': 'is int',
    'is_float': 'is float',
    'is_num': 'is num',
    'is_bool': 'is bool',

    'is_datetime': 'is datetime',
    'is_date': 'is date',

    'is_any': 'is any',
    'is_na': 'is na',
    'is_nk': 'is nk',
    'is_yn': 'is yn',
    'is_yes': 'is yes',
    'is_no': 'is no',
    })
_operators = qpDict({**_operators1, **_operators2})
_operators_only_df = qpDict({
    'lambda_condition_col': _operators['lambda_condition_col'],
    })

_modifiers1 = qpDict({
    #need a value for comparison or modification
    'set_x': ['x=', 'x ='],
    'set_col': ['col=', 'col ='],
    'set_row': ['row=', 'row ='],
    'set_col_tags': ['col#=', 'col# =', 'col #=', 'col # ='],
    'set_row_tags': ['row#=', 'row# =', 'row #=', 'row # ='],
    })
_modifiers2 = qpDict({
    #these dont need a values for comparison or modification
    'to_str': 'to str',
    'to_int': 'to int',
    'to_float': 'to float',
    'to_num': 'to num',
    'to_bool': 'to bool',

    'to_datetime': 'to datetime',
    'to_date': 'to date',

    'to_na': 'to na',
    'to_nk': 'to nk',
    'to_yn': 'to yn',
    })
_modifiers = qpDict({**_modifiers1, **_modifiers2})
_modifiers_only_df = qpDict({
    'set_col': _modifiers['set_col'],
    'set_row': _modifiers['set_row'],
    'set_col_tags': _modifiers['set_col_tags'],
    'set_row_tags': _modifiers['set_row_tags'],
    })



@pd.api.extensions.register_index_accessor('q')
class IndexQuery:
    """
    A custom query language for filtering and modifying data  in pandas Indices.

    each query consists of 2 string expressions, each of which can be left empty:
        1. filter: '=abc', '!=1', 'is int && >0'
        2. data modification: 'to num', 'x= str(x)'

    eg.: se.q('is str', 'x= "prefix" + x')

    multiple conditions in an expression as well as multiple whole queries can be connected.

    Connectors:
        &&: the previous conditions AND this new conditions have to apply
        //: the previous conditions OR this new conditions have to apply
        >>: ignore previous conditions and use these new conditions INSTEAD

    Operators:
        >: bigger
        <: smaller
        >=: bigger or equal
        <=: smaller or equal
        
        ~: contains a regex pattern (re.search)
        ~~: matches a regex pattern (re.match)

        =: equal
        ==: strictly equal (case sensitive)

        (): contains a string
        (()): contains a string (case sensitive)

        ?: filter using a python expression (must evaluate to True or False)

        is str: is a string
        is int: is an integer
        is float: is a float
        is num: is a number (int or float)
        is bool: is a boolean

        is date: is a date (quite format lenient but defaults to european formats)
        is datetime: is a datetime

        is any: matches any value, use to select all
        is na: not available data (quite lenient)
        is nk: not known data (quite lenient)
        is yk: is a yes or no value
        is yes: is a yes value
        is no: is a no value

    Modifiers:
        x=: set x to a value

        to str: convert to string
        to int: convert to integer
        to float: convert to float
        to num: convert to number
        to bool: convert to boolean

        to date: convert to date
        to datetime: convert to datetime

        to na: convert to not available
        to nk: convert to not known
        to yn: convert to yes or no
    """


    def __init__(self, idx: pd.Index):
        idx.qp._operators = _operators
        idx.qp._operators_only_df = _operators_only_df
        idx.qp._modifiers = _modifiers
        idx.qp._modifiers_only_df = _modifiers_only_df
        self.idx = idx 


    #wip
    # def check(self):
    #     _check_index(self.idx)


    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():' + self.__doc__
    

    def __call__(self, *expressions, verbosity=3):

        #setup
        idx = self.idx
        idx.qp = self.idx.qp
        idx.qp._input = f".q{expressions}"
        
        index_expression = index_condition = pd.Index([True for i in idx])
        for i_expression,expression in enumerate(expressions):
            expression_type = ['filter index', 'modify index'][i_expression%2]
            if expression == '':
                continue


            if expression_type == 'filter index':
                ast = _parse_expression(idx, expression, 'filter index', verbosity)

                for condition in ast['conditions']:
                    index_new = _apply_condition(idx, condition, _operators, verbosity, index=idx)

                    index_condition = _update_index(index_condition, index_new, condition['condition_connector'], i=None, verbosity=verbosity)
                index_expression = _update_index(index_expression, index_condition, ast['connector'], i=None, verbosity=verbosity)

                if index_expression.any() == False and verbosity >= 2:
                        log(f'no values fulfill the condition(s) in "{expression}"', level='warning', source='idx.q', input=idx.qp._input)


            elif expression_type == 'modify index':
                ast = _parse_expression_modify(idx, expression, expression_type, verbosity)

                for modification in ast['modifications']:
                    se = idx.to_series()
                    se.qp = idx.qp
                    idx = pd.Index(_apply_modification(se, index_expression, modification, verbosity, index=idx))
                    idx.qp = self.idx.qp
        


        idx_filtered = idx[index_expression]
        idx_filtered.qp = idx.qp

        return idx_filtered


@pd.api.extensions.register_series_accessor('q')
class SeriesQuery:
    """
    A custom query language for filtering and modifying data  in pandas Series.

    each query consists of 2 string expressions, each of which can be left empty:
        1. filter: '=abc', '!=1', 'is int && >0'
        2. data modification: 'to num', 'x= str(x)'

    eg.: se.q('is str', 'x= "prefix" + x')

    multiple conditions in an expression as well as multiple whole queries can be connected.

    Connectors:
        &&: the previous conditions AND this new conditions have to apply
        //: the previous conditions OR this new conditions have to apply
        >>: ignore previous conditions and use these new conditions INSTEAD

    Operators:
        >: bigger
        <: smaller
        >=: bigger or equal
        <=: smaller or equal
        
        ~: contains a regex pattern (re.search)
        ~~: matches a regex pattern (re.match)

        =: equal
        ==: strictly equal (case sensitive)

        (): contains a string
        (()): contains a string (case sensitive)

        ?: filter using a python expression (must evaluate to True or False)

        is str: is a string
        is int: is an integer
        is float: is a float
        is num: is a number (int or float)
        is bool: is a boolean

        is date: is a date (quite format lenient but defaults to european formats)
        is datetime: is a datetime

        is any: matches any value, use to select all
        is na: not available data (quite lenient)
        is nk: not known data (quite lenient)
        is yk: is a yes or no value
        is yes: is a yes value
        is no: is a no value

    Modifiers:
        x=: set x to a value

        to str: convert to string
        to int: convert to integer
        to float: convert to float
        to num: convert to number
        to bool: convert to boolean

        to date: convert to date
        to datetime: convert to datetime

        to na: convert to not available
        to nk: convert to not known
        to yn: convert to yes or no
    """


    def __init__(self, se: pd.Series):
        se.qp._operators = _operators
        se.qp._operators_only_df = _operators_only_df
        se.qp._modifiers = _modifiers
        se.qp._modifiers_only_df = _modifiers_only_df
        self.se = se 


    #wip
    # def check(self):
    #     _check_series(self.se)


    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():' + self.__doc__
    

    def __call__(self, *expressions, verbosity=3):

        #setup
        se = copy.deepcopy(self.se)
        se.qp = self.se.qp
        se.qp._input = f".q{expressions}"
        
        index_expression = index_condition = pd.Index([True for i in se])
        for i_expression,expression in enumerate(expressions):
            expression_type = ['filter series', 'modify series'][i_expression%2]
            if expression == '':
                continue


            if expression_type == 'filter series':
                ast = _parse_expression(se, expression, expression_type, verbosity)

                for condition in ast['conditions']:
                    index_new = _apply_condition(se, condition, _operators, verbosity, series=se)

                    index_condition = _update_index(index_condition, index_new, condition['condition_connector'], i=None, verbosity=verbosity)
                index_expression = _update_index(index_expression, index_condition, ast['connector'], i=None, verbosity=verbosity)

                if index_expression.any() == False and verbosity >= 2:
                    log(f'no values fulfill the condition(s) in "{expression}"', level='warning', source='se.q', input=se.qp._input)


            elif expression_type == 'modify series':
                ast = _parse_expression_modify(se, expression, expression_type, verbosity)

                for modification in ast['modifications']:
                    se = _apply_modification(se, index_expression, modification, verbosity, series=se)
                    se.qp = self.se.qp


        se_filtered = se[index_expression]
        se_filtered.qp = se.qp
        return se_filtered


@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    A custom query language for filtering and modifying data and metadata in pandas DataFrames.

    each query consists of 3 string expressions, any of which can be left empty:
        1. column filter: '=col1', '!=col2', '()1 // ()2'
        2. row filter: 'is date', 'is int && >0', '? len(str(x)) > 5'
        3. (meta-)data modification: 'x= str(x)', 'col#= integer column', 'row#= positive'

    eg.: df.q('=col1', 'is int && >0', 'row#= positive')

    multiple conditions in an expression as well as multiple whole queries can be connected.

    Connectors:
        &&: the previous conditions AND this new conditions have to apply
        //: the previous conditions OR this new conditions have to apply
        >>: ignore previous conditions and use these new conditions INSTEAD

    Operators:
        >: bigger
        <: smaller
        >=: bigger or equal
        <=: smaller or equal
        
        ~: contains a regex pattern (re.search)
        ~~: matches a regex pattern (re.match)

        =: equal
        ==: strictly equal (case sensitive)

        (): contains a string
        (()): contains a string (case sensitive)

        ?: filter using a python expression (must evaluate to True or False)

        is str: is a string
        is int: is an integer
        is float: is a float
        is num: is a number (int or float)
        is bool: is a boolean

        is date: is a date (quite format lenient but defaults to european formats)
        is datetime: is a datetime

        is any: matches any value, use to select all
        is na: not available data (quite lenient)
        is nk: not known data (quite lenient)
        is yk: is a yes or no value
        is yes: is a yes value
        is no: is a no value

    Modifiers:
        x=: set x to a value
        col=: set whole column to a value
        row=: set whole row to a value
        col#=: tag columns
        row#=: tag rows

        to str: convert to string
        to int: convert to integer
        to float: convert to float
        to num: convert to number
        to bool: convert to boolean

        to date: convert to date
        to datetime: convert to datetime

        to na: convert to not available
        to nk: convert to not known
        to yn: convert to yes or no
    """


    def __init__(self, df: pd.DataFrame):
        _check_df(df)
        self.df = df
        self.df.qp._operators = _operators
        self.df.qp._modifiers = _modifiers


    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():' + self.__doc__
    

    def __call__(self,
            *expressions,  #string expressions for filtering and modifying data
            diff=None,  #[None, 'mix', 'old', 'new', 'new+']
            max_cols=200,  #maximum number of columns to display. None: show all
            max_rows=20,  #maximum number of rows to display. None: show all
            inplace=False,  #make modifications inplace or just return a new dataframe
            verbosity=3,  #verbosity level for logging. 0: no logging, 1: errors, 2: warnings, 3: info, 4: debug
            **kwargs
            ):

        ###########################################
        #                  setup                  #
        ###########################################

        #input string for logging
        input_str = ".q("
        for i,expression in enumerate(expressions):
            if (i+1)%3 == 1:
                input_str += f"\n\t{expression !r},"
            else:
                input_str += f" {expression !r},"
   
        if diff is not None:
            input_str += f"\n\tdiff='{diff}',"
        if max_cols is not None:
            input_str += f"\n\tmax_cols={max_cols},"
        if max_rows is not None:
            input_str += f"\n\tmax_rows={max_rows},"
        
        input_str += f"\n\tinplace={inplace},"
        input_str += f"\n\tverbosity={verbosity},"

        for kwarg in kwargs:
            input_str += f"\n\t{kwarg}='{kwargs[kwarg]}'"

        self.df.qp._input = input_str + "\n\t)"


        if inplace is False:
            df = copy.deepcopy(self.df)
        else:
            df = self.df

        #inserting metadata row at the top, sadly a bit hacky
        #because there does not seem to be an inplace function for that
        if '#' not in df.index:
            if verbosity >= 3:
                log(f'inserting metadata row "#" at the top', level='info', source='df.q()')
            df_old = df.copy()
            index_old = df.index
            index_new = pd.Index(['#', *index_old])
            
            df.loc['#'] = ''
            df.set_index(index_new, inplace=True)
            df.loc[index_old, :] = df_old
            df.loc['#'] = ''

        #inserting metadata column at the start
        if '#' not in df.columns:
            if verbosity >= 3:
                log(f'inserting metadata column "#" at the start', level='info', source='df.q()')
            df.insert(0, '#', '')


        df_tagged = df
        df = df.iloc[1:, 1:]
        df_tagged.qp = self.df.qp
        df.qp = self.df.qp


        
        ###########################################
        #               run queries               #
        ###########################################

        cols_filtered = cols_filtered_condition = pd.Index([True for col in df.columns])
        rows_filtered = rows_filtered_condition = rows_filtered_col = pd.Index([True for row in df.index])
        

        for i_expression,expression in enumerate(expressions):
            expression_type = ['col', 'row', 'modify'][i_expression%3]
            if expression == '':
                continue


            #filter columns
            if expression_type == 'col':
                ast = _parse_expression(df, expression, expression_type, verbosity)

                for i_condition,condition in enumerate(ast['conditions']):
                    if condition['by_tag'] == 'row or col metadata':
                        cols_filtered_new = _apply_condition(df_tagged.iloc[0, 1:], condition, _operators, verbosity, df=df)
                    else:
                        cols_filtered_new = _apply_condition(df.columns, condition, _operators, verbosity, df=df)

                    cols_filtered_condition = _update_index(cols_filtered_condition, cols_filtered_new, condition['condition_connector'], i_condition, verbosity=verbosity)
                cols_filtered = _update_index(cols_filtered, cols_filtered_condition, ast['connector'], i_expression, verbosity=verbosity)

                if cols_filtered.any() == False and verbosity >= 2:
                    log(f'no columns fulfill the condition(s) in "{expression}"', level='warning', source='df.q', input=df.qp._input)

                


            #filter rows
            elif expression_type == 'row':
                ast = _parse_expression(df, expression, expression_type, verbosity)

                for i_condition,condition in enumerate(ast['conditions']):
                    for i_col,col in enumerate(df.columns[cols_filtered_condition]):
                        if condition['by_tag'] == 'row or col metadata':
                            rows_filtered_new = _apply_condition(df_tagged.iloc[1:, 0], condition, _operators, verbosity, df=df)
                        else:
                            rows_filtered_new = _apply_condition(df[col], condition, _operators, verbosity, df=df)

                        rows_filtered_col = _update_index(rows_filtered_col, rows_filtered_new, condition['which_cols'], i_col, verbosity=verbosity)
                    rows_filtered_condition = _update_index(rows_filtered_condition, rows_filtered_col, condition['condition_connector'], i_condition, verbosity=verbosity)
                rows_filtered = _update_index(rows_filtered, rows_filtered_condition, ast['connector'], i_expression-1, verbosity=verbosity)

                if rows_filtered.any() == False and verbosity >= 2:
                    log(f'no rows fulfill the condition(s) in "{expression}"', level='warning', source='df.q', input=df.qp._input)



            #modify data
            elif expression_type == 'modify':
                cols_filtered_no_metadata = [col for col in df.columns[cols_filtered] if not col.startswith('#')]
                rows_filtered_no_metadata = [row for row in df.index[rows_filtered] if row != '#']
                ast = _parse_expression_modify(df, expression, expression_type, verbosity)

                for modification in ast['modifications']:
                            
                    #tag columns
                    if modification['modifier'] in _modifiers['set_col_tags']:
                        df_tagged.loc['#', cols_filtered_no_metadata] = df_tagged.loc['#', cols_filtered_no_metadata].apply(
                            lambda x: eval(modification['value'])
                            )
                    
                    #tag rows
                    elif modification['modifier'] in _modifiers['set_row_tags']:
                        df_tagged.loc[rows_filtered_no_metadata, '#'] = df_tagged.loc[rows_filtered_no_metadata, '#'].apply(
                            lambda x: eval(modification['value'])
                            )
                    
                    #modify filtered data
                    else:
                        df.loc[:, :] = _apply_modification_df(df, rows_filtered_no_metadata, cols_filtered_no_metadata, modification, verbosity).loc[:, :]


        rows_filtered = pd.Index(['#', *df.index[rows_filtered]])
        cols_filtered = pd.Index(['#', *df.columns[cols_filtered]])
        
        df_tagged.iloc[1:, 1:] = df.loc[:,:]
        df_filtered = df_tagged.loc[rows_filtered, cols_filtered]
        df_filtered.qp = self.df.qp
        
        if inplace is True:
            self.df.loc[:,:] = df_tagged.loc[:,:]



        ##########################################
        #            display settings            #
        ##########################################
   
            
        if diff is None:
            pd.set_option('display.max_columns', max_cols)
            pd.set_option('display.max_rows', max_rows)

            cols_num = len(df_filtered.columns)
            rows_num = len(df_filtered.index)

            if verbosity >= 2:
                if max_cols is not None and max_cols < cols_num:
                    log(f'showing {max_cols} out of {cols_num} columns', level='warning', source='df.q', input=df.qp._input)
                if max_rows is not None and max_rows < rows_num:
                    log(f'showing {max_rows} out of {rows_num} rows', level='warning', source='df.q', input=df.qp._input)
 
            return df_filtered
        
        else:
            #show difference before and after filtering
            result = qp.diff(
                df_filtered, self.df, show=diff,
                max_cols=max_cols, max_rows=max_rows,
                verbosity=verbosity)  
            return  result



def _parse_expression(pd_object, expression, mode, verbosity=3):

    ast = {
        'expression': expression,
        'mode': mode,
        'connector': None,
        'conditions': [],
        }
    
    if expression[:2] not in ['&&', '//', '>>']:
        expression = '>>' + expression

    ast['connector'] = expression[:2]


    expressions = re.split('(&&|//|>>)', expression)
    for ind in range(len(expressions)):

        condition = {'expression': expressions[ind], 'mode': mode}
        condition_str = expressions[ind].strip()

        if len(condition_str) == 0 or condition_str in ['&&', '//', '>>']:
            continue

        
        if expressions[ind-1] == '&&':  #and
            condition['condition_connector'] = '&&'
        elif expressions[ind-1] == '//':  #inclusive or
            condition['condition_connector'] = '//'
        elif expressions[ind-1] == '>>':  #reset
            condition['condition_connector'] = '>>'

        
        #row conditions also specify in which cols the condition is applied. default: any
        if mode == 'row':
            if condition_str.startswith('any'):
                condition['which_cols'] = 'any'
                condition_str = condition_str[3:].lstrip()
            elif condition_str.startswith('all'):
                condition['which_cols'] = 'all'
                condition_str = condition_str[3:].lstrip()
            else:
                condition['which_cols'] = 'any'


        #rows and cols can be filtered by metadata
        if condition_str.startswith('#'):
            condition['by_tag'] = 'row or col metadata'
            condition_str = condition_str[1:].lstrip()
        else:
            condition['by_tag'] = False

        #wip
        # if condition_str.startswith('x#'):
        #     condition['by_tag_value'] = 'value metadata'
        #     condition_str = condition_str[1:].lstrip()
        # else:
        #     condition['by_tag_value'] = False

        


        #should the condition be negated. default: False
        if condition_str.startswith('!'):
            condition['negate'] = True
            condition_str = condition_str[1:].lstrip()
        else:
            condition['negate'] = False


        #operator for condition. default: =
        for operator in _operators.values_flat():
            if condition_str.startswith(operator):
                condition['operator'] = operator
                condition_str = condition_str[len(operator):].lstrip()
                break
        
        if 'operator' not in condition:
            if verbosity >= 3:
                log(f'no operator found in condition "{condition_str}". Using default operator "="',
                    level='info', source='_parse_expression', input=pd_object.qp._input)
            condition['operator'] = '='


        if mode in ['filter series', 'filter index'] and condition['operator'] in _operators_only_df.values_flat():
            if verbosity >= 1:
                operator_temp = condition['operator']
                log(f'operator "{operator_temp}" only works with dataframes. Ignoring condition "{condition_str}"',
                    level='error', source='_parse_expression', input=pd_object.qp._input)
            continue
        
        if condition['operator'] in _operators2.values_flat() and len(condition_str) > 0:
            if verbosity >= 2:
                operator_temp = condition['operator']
                log(f'operator "{operator_temp}" does not need a value for comparison. Ignoring value "{condition_str}"',
                    level='warning', source='_parse_expression', input=pd_object.qp._input)


        #value for comparison
        condition['value'] = condition_str.strip()
    
        ast['conditions'].append(condition)

    if verbosity >= 4:
        display(f'abstract syntax tree for expression "{expression}":', ast)
        
    return ast

def _parse_expression_modify(pd_object, expression, mode, verbosity=3):

    ast = {
        'expression': expression,
        'mode': mode,
        'modifications': [],
        }
    

    expressions = re.split('(&&)', expression)
    for ind in range(len(expressions)):

        condition = {}
        condition_str = expressions[ind].strip()

        if len(condition_str) == 0 or condition_str in ['&&']:
            continue


        #modifier to use. default: "set x:"
        for modifier in _modifiers.values_flat():
            if condition_str.startswith(modifier):
                condition['modifier'] = modifier
                condition_str = condition_str[len(modifier):].lstrip()
                break
        
        if 'modifier' not in condition:
            if verbosity > 3:
                log(f'no modifier found in condition "{condition_str}". Using default modifier "x="',
                    level='info', source='_parse_expression', input=pd_object.qp._input)
            condition['modifier'] = 'x='

        if mode in ['modify series', 'modify index'] and condition['modifier'] in _modifiers_only_df.values_flat():
            if verbosity >= 1:
                modifier_temp = condition['modifier']
                log(f'modifier "{modifier_temp}" only works with dataframes. Ignoring condition "{condition_str}"',
                    level='error', source='_parse_expression', input=pd_object.qp._input)
            continue

        if condition['modifier'] in _modifiers2.values_flat() and len(condition_str) > 0:
            if verbosity >= 2:
                modifier_temp = condition['modifier']
                log(f'modifier "{modifier_temp}" does not need a value for comparison. Ignoring value "{condition_str}"',
                    level='warning', source='_parse_expression', input=pd_object.qp._input)


        #expression used for modification
        condition['value'] = condition_str.strip()
    
        ast['modifications'].append(condition)
        
    if verbosity >= 4:
        display(f'abstract syntax tree for expression "{expression}":', ast)
    
    return ast


def _apply_condition(pd_object, condition, operators, verbosity=3, df=None, series=None, index=None):
    """
    filters a pandas object using a query condition
    """

    value = condition['value']

    if isinstance(pd_object, pd.Index):
        pd_object = pd_object.to_series()

    
    match condition['operator']:
        #numeric comparison
        case operators.bigger_equal:
            filtered = pd.to_numeric(pd_object, errors='coerce') >= pd.to_numeric(value)
        case operators.smaller_equal:
            filtered = pd.to_numeric(pd_object, errors='coerce') <= pd.to_numeric(value)
        case operators.bigger:
            filtered = pd.to_numeric(pd_object, errors='coerce') > pd.to_numeric(value)
        case operators.smaller:
            filtered = pd.to_numeric(pd_object, errors='coerce') < pd.to_numeric(value)
        
        
        #regex comparison
        case operators.regex_match:
            filtered = pd_object.astype(str).str.fullmatch(value) 
        case operators.regex_search:
            filtered = pd_object.astype(str).str.contains(value)


        #string equality comparison
        case operators.strict_equal:
            filtered = pd_object.astype(str) == value
        case operators.equal:
            value_lenient = [value]
            try:
                value_lenient.append(str(float(value)))
                value_lenient.append(str(int(float(value))))
            except:
                value_lenient.append(value.lower())
            filtered = pd_object.astype(str).str.lower().isin(value_lenient)
        
        #substring comparison
        case operators.strict_contains:
            filtered = pd_object.astype(str).str.contains(value, case=True, regex=False)
        case operators.contains:
            filtered = pd_object.astype(str).str.contains(value, case=False, regex=False)



        #lambda function
        case operators.lambda_condition:
            filtered = pd_object.apply(lambda x, df=df, series=series, index=index, pd=pd, np=np: eval(value))
        case operators.lambda_condition_col:
            filtered = eval(value, {'col': pd_object, 'df': df, 'pd': pd, 'np': np, 'qp': qp})


        #type checks
        case operators.is_bool:
            filtered = pd_object.apply(lambda x: isinstance(x, bool))
        case operators.is_str:
            filtered = pd_object.apply(lambda x: isinstance(x, str))
        case operators.is_int:
            filtered = pd_object.apply(lambda x: isinstance(x, int))
        case operators.is_float:
            filtered = pd_object.apply(lambda x: isinstance(x, float))
        case operators.is_num:
            filtered = pd_object.apply(lambda x: qp_num(x, errors='ERROR')) != 'ERROR'

        case operators.is_date:
            filtered = pd_object.apply(lambda x: qp_date(x, errors='ERROR')) != 'ERROR'
        case operators.is_datetime:
            filtered = pd_object.apply(lambda x: qp_datetime(x, errors='ERROR')) != 'ERROR'

        case operators.is_any:
            filtered = pd_object.apply(lambda x: True)
        case operators.is_na:
            filtered = pd_object.apply(lambda x: qp_na(x, errors='ERROR')) != 'ERROR'
        case operators.is_nk:
            filtered = pd_object.apply(lambda x: qp_nk(x, errors='ERROR')) != 'ERROR'
        case operators.is_yn:
            filtered = pd_object.apply(lambda x: qp_yn(x, errors='ERROR')) != 'ERROR'
        case operators.is_yes:
            filtered = pd_object.apply(lambda x: qp_yn(x, errors='ERROR', yes=1)) == 1
        case operators.is_no:
            filtered = pd_object.apply(lambda x: qp_yn(x, errors='ERROR', no=0)) == 0

        case _:
            if verbosity >= 1:
                operator_temp = condition['operator']
                log(f'operator "{operator_temp}" is not implemented', level='error', source='_apply_condition()', input=pd_object.qp._input)
            filtered = None


    if condition['negate']:
        filtered = ~filtered

    return filtered

def _apply_modification(pd_object, indices, modification, verbosity=3, series=None, index=None):
    modifiers = pd_object.qp._modifiers

    #data modification
    if modification['modifier'] in modifiers['set_x']:
        pd_object[indices] = pd_object[indices].map(lambda x, pd=pd, np=np, qp=qp: eval(modification['value']))

    #type conversion
    elif modification['modifier'] == modifiers['to_str']:
        pd_object[indices] = pd_object[indices].map(str)
    elif modification['modifier'] == modifiers['to_int']:
        pd_object[indices] = pd_object[indices].map(qp_int)
    elif modification['modifier'] == modifiers['to_float']:
        pd_object[indices] = pd_object[indices].map(qp_float)
    elif modification['modifier'] == modifiers['to_num']:
        pd_object[indices] = pd_object[indices].map(qp_num)
    elif modification['modifier'] == modifiers['to_bool']:
        pd_object[indices] = pd_object[indices].map(qp_bool)
    
    elif modification['modifier'] == modifiers['to_date']:
        pd_object[indices] = pd_object[indices].map(qp_date)
    elif modification['modifier'] == modifiers['to_datetime']:
        pd_object[indices] = pd_object[indices].map(qp_datetime)
    elif modification['modifier'] == modifiers['to_na']:
        pd_object[indices] = pd_object[indices].map(qp_na)
    elif modification['modifier'] == modifiers['to_nk']:
        pd_object[indices] = pd_object[indices].map(qp_nk)
    elif modification['modifier'] == modifiers['to_yn']:
        pd_object[indices] = pd_object[indices].map(qp_yn)

    return pd_object

def _apply_modification_df(df, rows, cols, modification, verbosity=3):
    modifiers = df.qp._modifiers

    if pd.__version__ >= '2.1.0':
        #data modification
        if modification['modifier'] in modifiers['set_x']:
            df.loc[rows, cols] = df.loc[rows, cols].map(lambda x, df=df, pd=pd, np=np, qp=qp: eval(modification['value']))
                

        elif modification['modifier'] in modifiers['set_col']:
            df.loc[:, cols] = df.loc[:, cols].apply(lambda x, df=df, pd=pd, np=np, qp=qp: eval(modification['value']), axis=0)

        elif modification['modifier'] in modifiers['set_row']:
            df.loc[rows, :] = df.loc[rows, :].apply(lambda x, df=df, pd=pd, np=np, qp=qp: eval(modification['value']), axis=1)


        #type conversion
        elif modification['modifier'] == modifiers['to_str']:
            df.loc[rows, cols] = df.loc[rows, cols].map(str)
        elif modification['modifier'] == modifiers['to_int']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_int)
        elif modification['modifier'] == modifiers['to_float']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_float)
        elif modification['modifier'] == modifiers['to_num']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_num)
        elif modification['modifier'] == modifiers['to_bool']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_bool)
        
        elif modification['modifier'] == modifiers['to_date']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_date)
        elif modification['modifier'] == modifiers['to_datetime']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_datetime)

        elif modification['modifier'] == modifiers['to_na']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_na)
        elif modification['modifier'] == modifiers['to_nk']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_nk)
        elif modification['modifier'] == modifiers['to_yn']:
            df.loc[rows, cols] = df.loc[rows, cols].map(qp_yn)

    else:
        #data modification
        if modification['modifier'] in modifiers['set_x']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(lambda x, df=df, pd=pd, np=np, qp=qp: eval(modification['value']))
                

        elif modification['modifier'] in modifiers['set_col']:
            df.loc[:, cols] = df.loc[:, cols].apply(lambda x, df=df, pd=pd, np=np, qp=qp: eval(modification['value']), axis=0)

        elif modification['modifier'] in modifiers['set_row']:
            df.loc[rows, :] = df.loc[rows, :].apply(lambda x, df=df, pd=pd, np=np, qp=qp: eval(modification['value']), axis=1)


        #type conversion
        elif modification['modifier'] == modifiers['to_str']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(str)
        elif modification['modifier'] == modifiers['to_int']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_int)
        elif modification['modifier'] == modifiers['to_float']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_float)
        elif modification['modifier'] == modifiers['to_num']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_num)
        elif modification['modifier'] == modifiers['to_bool']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_bool)
        
        elif modification['modifier'] == modifiers['to_date']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_date)
        elif modification['modifier'] == modifiers['to_datetime']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_datetime)

        elif modification['modifier'] == modifiers['to_na']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_na)
        elif modification['modifier'] == modifiers['to_nk']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_nk)
        elif modification['modifier'] == modifiers['to_yn']:
            df.loc[rows, cols] = df.loc[rows, cols].applymap(qp_yn) 

    return df
    

def _update_index(values, values_new, connector, i, verbosity=3):
    if i == 0:
        values = values_new
    elif connector == '>>':
        values = values_new
    elif connector in ['&&', 'all']:
        values &= values_new
    elif connector in ['//', 'any']:
        values |= values_new
    else:
        if verbosity >= 1:
            log(f'connector "{connector}" is not implemented', level='error', source='_update_index()')
    return values



@pd.api.extensions.register_dataframe_accessor('qi')
class DataFrameQueryInteractiveMode:
    """
    Wrapper for df.q() for interactive use in Jupyter notebooks.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, num_filters=5):
        kwargs = {'df': fixed(self.df)}


        ###########################################
        #            tab0 queries&diff            #
        ###########################################

        ui_label_filter_cols = widgets.Label(value='filter columns')
        ui_label_filter_rows = widgets.Label(value='filter rows')
        ui_label_modify_data = widgets.Label(value='modify data')
        ui_expressions = [ui_label_filter_cols, ui_label_filter_rows, ui_label_modify_data]



        for i in range(num_filters):
            if i == 0:
                placeholder_col = '=name'
                placeholder_row = '()john'
                placeholder_modify = 'x= x.upper() '
            elif i == 1:
                placeholder_col = '// =age'
                placeholder_row = '&& <0 // >120'
                placeholder_modify = 'row#= "implausible age" '
            elif i == 2:
                placeholder_col = '// =weight // =height'
                placeholder_row = '!is num'
                placeholder_modify = 'to num'
            else:
                placeholder_col = ''
                placeholder_row = ''
                placeholder_modify = ''
            

            col = widgets.Combobox(
                value='',
                placeholder=placeholder_col,
                options=['=' + col for col in self.df.columns if col != '#'],
                )
            kwargs[f'col_expression{i}'] = col
            ui_expressions.append(col)
            
            row = widgets.Combobox(
                value='',
                placeholder=placeholder_row,
                options=[
                    '>0', '<0', '>=0', '<=0',  #numerical comparison
                    '=abc', '!=abc', '==AbC', '!==AbC',  #equality checks
                    '()abc', '(())AbC',  #substring checks
                    '~*a.c', '~~*a.c',  #regex checks
                    
                    '? len(x) > 3',  #lambda function condition for each value
                    'col? col > df["age"]',  #lambda function condition for whole columns

                    #type checks
                    'is str', 'is int', 'is float', 'is num', 'is bool',
                    'is date', 'is datetime',
                    'is any', 'is na', 'is nk',
                    'is yn', 'is yes', 'is no'
                    ]
                )
            kwargs[f'row_expression{i}'] = row
            ui_expressions.append(row)
            
            modify = widgets.Combobox(
                value='',
                placeholder=placeholder_modify,
                options=[
                    'x= x.upper() ', 'col= df["ID"]', 'row= df.loc[0]',  #change data
                    'col#= "tag" ', 'col#= x + "tag" ', 'row#= "tag" ', 'row#= x + "tag"',  #change metadata

                    #change type
                    'to str', 'to int', 'to float', 'to num', 'to bool',
                    'to date', 'to datetime',
                    'to na', 'to nk', 'to yn',
                    ]
                )
            kwargs[f'modify_expression{i}'] = modify
            ui_expressions.append(modify)
        

        #show differences
        ui_diff = widgets.ToggleButtons(
            options=['mix', 'old', 'new', 'new+', None],
            description='show differences mode:',
            tooltips=[
                'show new (filtered) dataframe plus all the removed (filtered) values from the old dataframe. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'show old (unfiltered) dataframe. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'show new (filtered) dataframe. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'show new (filtered) dataframe but also adds metadata columns with the prefix "#". If a value changed, the metadata column contains the old value. values affected by the filters are marked green (newly added), yellow (modified), red (deleted)',
                'dont show differences, just show the new (filtered) dataframe.',
                ],
            )
        kwargs['diff'] = ui_diff



        ###########################################
        #              tab1 settings              #
        ###########################################

        ui_inplace = widgets.ToggleButtons(
            options=[True, False],
            value=False,
            description='make modifications inplace:',
            tooltips=[
                'make modifications inplace, e.g. change the original dataframe',
                'return a new dataframe with the modifications',
                ],
            )
        kwargs['inplace'] = ui_inplace

        ui_verbosity = widgets.ToggleButtons(
            options=[0, 1, 2, 3, 4],
            value=3,
            description='verbosity level:',
            tooltips=[
                'no logging',
                'only errors',
                'errors and warnings',
                'errors, warnings and info',
                'errors, warnings, info and debug',
                ],
            )
        kwargs['verbosity'] = ui_verbosity



        cols_num = len(self.df.columns)
        rows_num = len(self.df.index)
        if '#' not in self.df.columns:
            cols_num = len(self.df.columns) + 1
        if '#' not in self.df.index:
            rows_num = len(self.df.index) + 1

        ui_max_cols = widgets.IntSlider(
            value=200,
            min=0,
            max=cols_num*2-1,  #*2 because of metadata columns which get added by diff='new+'
            description='columns',
            )
        kwargs['max_cols'] = ui_max_cols

        ui_max_rows = widgets.IntSlider(
            value=20,
            min=0,
            max=rows_num,
            description='rows',
            )
        kwargs['max_rows'] = ui_max_rows



        ###########################################
        #                tab2 info                #
        ###########################################


        ui_gridbox = widgets.GridBox(ui_expressions, layout=widgets.Layout(grid_template_columns="repeat(3, 330px)"))   
        tab0 = VBox([ui_gridbox, ui_diff])
        tab1 = VBox([ui_inplace, ui_verbosity, HBox([ui_max_cols, ui_max_rows])])
        ui_tab = widgets.Tab(
            children=[tab0, tab1],
            titles=['queries', 'settings'],
            )
        ui = VBox([ui_tab])
        display(ui)


        out = HBox([interactive_output(_interactive_mode, kwargs)], layout=Layout(overflow_y='auto'))

        display(out)

def _interactive_mode(**kwargs):

    df = kwargs.pop('df')
    expressions = [val for key, val in kwargs.items() if 'expression' in key]


    result = df.q(
        *expressions,
        diff=kwargs['diff'],
        max_cols=kwargs['max_cols'],
        max_rows=kwargs['max_rows'],
        inplace=kwargs['inplace'],
        verbosity=kwargs['verbosity'],
        )


    
    display(result)
    print('input code: ', df.qp._input)
    return result


