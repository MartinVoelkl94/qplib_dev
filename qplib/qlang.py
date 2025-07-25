
import numpy as np
import pandas as pd
import re
import qplib as qp

from IPython.display import display
from ipywidgets import widgets, interactive_output, HBox, VBox, fixed, Layout
from importlib.resources import files

from .util import log
from .types import _dict, _int, _float, _num, _bool, _datetime, _date, _na, _nk, _yn, _type
from .pandas import _diff



#####################     settings     #####################


VERBOSITY = 3
DIFF = None
INPLACE = False

TYPES_INT = (
    int,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    pd.Int64Dtype,
    pd.Int32Dtype,
    pd.Int16Dtype,
    pd.Int8Dtype,
    )
TYPES_FLOAT = (
    float,
    np.float64,
    np.float32,
    np.float16,
    pd.Float64Dtype,
    pd.Float32Dtype,
    )
TYPES_NUM = (
    int,
    float,
    np.int64,
    np.float64,
    np.int32,
    np.float32,
    np.int16,
    np.float16,
    np.int8,
    np.number,
    pd.Int64Dtype,
    pd.Float64Dtype,
    pd.Int32Dtype,
    pd.Float32Dtype,
    pd.Int16Dtype,
    pd.Int8Dtype,
    )
TYPES_BOOL = (
    bool,
    np.bool_,
    pd.BooleanDtype,
    )



##################     syntax symbols     ##################

class Symbol:
    """
    A Symbol used in the query languages syntax.
    """
    def __init__(self, name, glyph, symbol_type, description, traits):
        self.name = name
        self.glyph = glyph
        self.type = symbol_type
        self.description = description
        self.traits = traits

    def details(self):
        traits = '\n\t\t'.join(self.traits)
        return f'symbol:\n\tname: {self.name}\n\tsymbol: {self.glyph}\n\tdescription: {self.description}\n\ttraits:\n\t\t{traits}\n\t'

    def __repr__(self):
        return f'<"{self.glyph}": {self.name}>'
 
    def __str__(self):
        return self.__repr__()
    
    def __lt__(self, value):
        return self.glyph < value
    
    def __gt__(self, value):
        return self.glyph > value


class Symbols:
    """
    Multiple Symbols of the same category are collected in a Symbols object.
    """
    def __init__(self, name, *symbols):
        self.name = name
        self.by_name = {symbol.name: symbol for symbol in symbols}
        self.by_glyph = {symbol.glyph: symbol for symbol in symbols}
        self.by_trait = {}
        for symbol in symbols:
            for trait in symbol.traits:
                if trait not in self.by_trait:
                    self.by_trait[trait] = set()
                self.by_trait[trait].add(symbol)

    def __getattribute__(self, value):
        if value == 'by_name':
            return super().__getattribute__(value)
        elif value == 'by_glyph':
            return super().__getattribute__(value)
        elif value == 'by_trait':
            return super().__getattribute__(value)
        elif value in self.by_glyph:
            return self.by_glyph[value] #pragma: no cover (not possible with current symbols)
        elif value in self.by_name:
            return self.by_name[value]
        elif value in self.by_trait:
            return self.by_trait[value]
        else:
            return super().__getattribute__(value)

    def __getitem__(self, key):
        if key in self.by_glyph:
            return self.by_glyph[key]
        elif key in self.by_name:
            return self.by_name[key]
        elif key in self.by_trait:
            return self.by_trait[key]
        else:
            log(f'error: symbol "{key}" not found in "{self.name}"', 'qp.qlang.Symbols.__getitem__', VERBOSITY)
            return None

    def __iter__(self):
        return iter(self.by_name.values())

    def __repr__(self):
        return f'{self.name}:\n\t' + '\n\t'.join([str(val) for key,val in self.by_name.items()])

    def __str__(self):
        return self.__repr__()


class Instruction:
    """
    An instruction to select/filter or modify data/metadata.
    """
    def __init__(self, code='', line_num=None):
        #initial values
        self.line_num = line_num
        self.code = code

        #determined by tokenize()
        self.connector = None
        self.flags = set()
        self.operator = None
        self.value = None

        #determined by parse()
        self.function = None


    def __repr__(self):
        string = f'Instruction:\n\tline_num: {self.line_num}\n\tcode: {self.code}\n\tconnector: {self.connector}'
        for flag in self.flags:
            string += f'\n\tflag: {flag}'
        string += f'\n\toperator: {self.operator}\n\tvalue: {self.value}'
        if self.function:
            string += f'\n\tfunction: {self.function.__name__}'

        return string

    def __str__(self):
        return self.__repr__()



def get_symbols():
    path_symbols = files('qplib').joinpath('data/symbols.csv')
    definitions = pd.read_csv(path_symbols, index_col=0)
    definitions.drop(index=['type', 'glyph', 'description'], inplace=True)
    definitions['glyph'] = definitions['glyph'].str.strip('"')
    definitions.iloc[:, 3:] = definitions.iloc[:, 3:].fillna(0).astype('int')
    compatible = definitions.iloc[:, 3:].astype(bool)

    traits_all = definitions.loc[definitions['type'] == 'trait', :].index
    connectors = []
    operators = []
    flags = []

    for ind in definitions.index:
        name = definitions.loc[ind, :].name
        glyph = definitions.loc[ind, 'glyph']
        symbol_type = definitions.loc[ind, 'type']
        description = definitions.loc[ind, 'description']
        traits = [trait for trait in traits_all if definitions.loc[ind, trait] == 2]

        symbol = Symbol(name, glyph, symbol_type, description, traits)

        if symbol.type == 'connector':
            connectors.append(symbol)
        elif symbol.type == 'operator':
            operators.append(symbol)
        elif symbol.type == 'flag':
            flags.append(symbol)

    connectors = Symbols('CONNECTORS', *connectors)
    operators = Symbols('OPERATORS', *operators)
    flags = Symbols('FLAGS', *flags)

    return definitions, connectors, operators, flags, compatible



DEFINITIONS, CONNECTORS, OPERATORS, FLAGS, compatible = get_symbols()
COMMENT = Symbol('COMMENT', '#', 'syntax', 'comments out the rest of the line', [])
ESCAPE = Symbol('ESCAPE', '´',  'syntax', 'escape the next character', [])




#################     main query logic     #################


def query(df_old, code=''):
    """
    Used by the dataframe accessors df.q() (DataFrameQuery) and df.qi() (DataFrameQueryInteractive).
    """

    #setup

    check_df(df_old, VERBOSITY)

    df_new = df_old  #df_old will be copied later, only if any modifications are applied

    settings = _dict()

    settings.cols = pd.Series([True for col in df_new.columns])
    settings.cols.index = df_new.columns

    settings.rows = pd.Series([True for row in df_new.index])
    settings.rows.index = df_new.index

    settings.vals = pd.DataFrame(
        np.ones(df_new.shape, dtype=bool),
        columns=df_new.columns,
        index=df_new.index
        )
    
    settings.vals_blank = pd.DataFrame(
        np.zeros(df_new.shape, dtype=bool),
        columns=df_new.columns,
        index=df_new.index
        )
    
    settings.saved = {}
    settings.verbosity = VERBOSITY
    settings.diff = DIFF
    settings.style = None
    settings.copy_df = False
    settings.df_copied = False


    #apply instructions

    instructions_raw, settings = scan(code, settings)

    for instruction_raw in instructions_raw:

        instruction_tokenized, settings = tokenize(instruction_raw, settings)
        instruction, settings = parse(instruction_tokenized, settings)
        instruction = validate(instruction, settings)

        if settings.copy_df and not settings.df_copied:
            df_new = df_old.copy()
            settings.df_copied = True

        log(f'debug: applying instruction:\n{instruction}', 'qp.qlang.query', settings.verbosity)
        df_new, settings  = instruction.function(instruction, df_new, settings)
        log(f'trace: instruction applied', 'qp.qlang.query', settings.verbosity)



    #results

    df_filtered = df_new.loc[settings.rows, settings.cols]

    if settings.diff is not None and settings.style is not None:
        log('warning: diff and style formatting are not compatible. formatting will be ignored',
            'qp.qlang.query', settings.verbosity)
        settings.style = None

    if settings.diff is not None:
        #show difference before and after filtering
        if 'meta' in df_old.columns and 'meta' not in df_filtered.columns:
            df_filtered.insert(0, 'meta', df_old.loc[settings.rows, 'meta'])

        result = _diff(
            df_filtered, df_old,
            mode=settings.diff,
            verbosity=settings.verbosity
            )

    elif settings.style is not None:
        rows_shared = df_filtered.index.intersection(settings.style.index)
        cols_shared = df_filtered.columns.intersection(settings.style.columns)
        def f(x, style): #pragma: no cover
            return style
        result = df_filtered.style.apply(lambda x: f(x, settings.style.loc[rows_shared, cols_shared]), axis=None, subset=(rows_shared,  cols_shared))

    else:
        result = df_filtered

    return result



def check_df(df, verbosity=3):
    """
    Checks dataframe for issues which could interfere with the query language.
    Query language uses '%', &', '/' and '$' for expression syntax.
    """
    problems_found = False

    if len(df.index) != len(df.index.unique()):
        log('error: index is not unique', 'qp.qlang.check_df', verbosity)
        problems_found = True

    if len(df.columns) != len(df.columns.unique()):
        log('error: cols are not unique', 'qp.qlang.check_df', verbosity)
        problems_found = True

    syntax_conflicts = {
        '"%"': [],
        '"&"': [],
        '"/"': [],
        '"$"': [],
        }
    whitespace = {
        'leading whitespace': [],
        'trailing whitespace': [],
        }

    for col in df.columns:
        if isinstance(col, str):
            if '%' in col:
                syntax_conflicts['"%"'].append(col)
            if '&' in col:
                syntax_conflicts['"&"'].append(col)
            if '/' in col:
                syntax_conflicts['"/"'].append(col)
            if '$' in col:
                syntax_conflicts['"$"'].append(col)
            if col.startswith(' '):
                whitespace['leading whitespace'].append(col)
            if col.endswith(' '):
                whitespace['trailing whitespace'].append(col)

    for problem, cols in syntax_conflicts.items():
        if len(cols) > 0:
            log(f'warning: the following col headers contain {problem} which is used by the query syntax, use a tick (´) to escape such characters:\n\t{cols}',
                'qp.qlang.check_df', verbosity)
            problems_found = True

    for problem, cols in whitespace.items():
        if len(cols) > 0:
            log(f'warning: the following col headers contain {problem} which should be removed:\n\t{cols}',
                'qp.qlang.check_df', verbosity)
            problems_found = True


    symbol_conflicts = []

    for col in df.columns:
        if str(col).startswith(tuple(OPERATORS.by_glyph.keys())):
            symbol_conflicts.append(col)

    if len(symbol_conflicts) > 0:
        log(f'warning: the following col headers start with a character sequence that can be read as a query instruction symbol when the default instruction operator is inferred:\n{symbol_conflicts}\nexplicitely use a valid operator to avoid conflicts.',
            'qp.qlang.check_df', verbosity)
        problems_found = True


    if problems_found is False:
        log('debug: df was checked. no problems found', 'qp.qlang.check_df', verbosity)



def scan(code, settings):
    """
    Turns the plain text input string into a list of raw instructions.
    """
    verbosity = settings.verbosity
    instructions_raw = []

    for line_num, line in enumerate(code.split('\n')):

        line = line.strip()
        if line == '':
            continue
        elif line.startswith(COMMENT.glyph):
            continue
        elif line[0] not in CONNECTORS.by_glyph:
            log(f'trace: line "{line}" does not start with a connector, adding NEW_SELECT_COLS connector to the beginning of the line',
                'qp.qlang.tokenize', verbosity)
            line = CONNECTORS.NEW_SELECT_COLS.glyph + line

        while True:

            if line == '':
                break

            elif line.startswith(ESCAPE.glyph):
                instructions_raw[-1].code += line[1]
                line = line[2:]

            elif line.startswith(COMMENT.glyph):
                break

            elif len(line) > 2 and line[:3] in CONNECTORS.by_glyph:
                instructions_raw.append(Instruction(line[:3], line_num))
                line = line[3:]

            elif len(line) > 1 and line[:2] in CONNECTORS.by_glyph:
                instructions_raw.append(Instruction(line[:2], line_num))
                line = line[2:]

            elif line[0] in CONNECTORS.by_glyph:
                instructions_raw.append(Instruction(line[0], line_num))
                line = line[1:]

            else:
                instructions_raw[-1].code += line[0]
                line = line[1:]


    log('trace: transformed code into raw instructions:\n' + '\n'.join([str(instruction) for instruction in instructions_raw]),
        'qp.qlang.tokenize', verbosity)

    return instructions_raw, settings



def tokenize(instruction_raw, settings):
    """
    extracts syntax symbols from raw instruction strings.
    """
    verbosity = settings.verbosity
    instruction_tokenized = instruction_raw
    code = instruction_raw.code

    instruction_tokenized.connector, code = extract_symbol(code, CONNECTORS, verbosity)

    while True:
        flag, code = extract_symbol(code, symbols=FLAGS, verbosity=verbosity)
        if flag is None:
            break
        else:
            instruction_tokenized.flags.add(flag)

    instruction_tokenized.operator, code = extract_symbol(code, OPERATORS, verbosity)
    instruction_tokenized.value = code

    return instruction_tokenized, settings



def extract_symbol(string, symbols, verbosity=3):

    for symbol in symbols:
        if string.startswith(symbol.glyph):
            log(f'trace: found "{symbols.name}.{symbol.name}" in "{string}"',
                'qp.qlang.extract_symbol', verbosity)
            return symbol, string[len(symbol.glyph):].strip()

    return None, string



def parse(instruction_tokenized, settings):
    """
    translates syntax symbols into actual instruction behaviour.
    """
    verbosity = settings.verbosity
    instruction = instruction_tokenized
    code = instruction.code
    flags = instruction.flags


    #set defaults

    if instruction.operator is None:
        if FLAGS.TAG_METADATA in flags:
            log(f'trace: no operator found in "{code}". using default "{OPERATORS.ADD}" for tagging metadata', 'qp.qlang.parse', verbosity)
            instruction.operator = OPERATORS.ADD
        else:
            log(f'trace: no operator found in "{code}". using default "{OPERATORS.SET}"', 'qp.qlang.parse', verbosity)
            instruction.operator = OPERATORS.SET

    if FLAGS.SAVE_SELECTION in flags or FLAGS.LOAD_SELECTION in flags:
        pass #no defaults needed here
    
    elif instruction.connector in CONNECTORS.by_trait['select_rows'] \
        and not flags.intersection(FLAGS.by_trait['select_rows_scope']):
        log(f'trace: no row selection scope flag found in "{code}". using default "{FLAGS.ANY}"', 'qp.qlang.parse', verbosity)
        instruction.flags.add(FLAGS.ANY)

    elif instruction.connector == CONNECTORS.MODIFY \
        and not flags.intersection(FLAGS.by_trait['modify']):
        log(f'trace: no modification flag found in "{code}". using default "{FLAGS.VAL}"', 'qp.qlang.parse', verbosity)
        instruction.flags.add(FLAGS.VAL)



    #set function

    if FLAGS.SAVE_SELECTION in flags:
        instruction.function = _save_selection
    elif FLAGS.LOAD_SELECTION in flags:
        instruction.function = _load_selection

    elif instruction.connector in CONNECTORS.by_trait['select']:

        if instruction.operator == OPERATORS.SET:
            log(f'trace:"{OPERATORS.SET}" is interpreted as "{OPERATORS.EQUALS}" for selection instruction',
                'qp.qlang.parse', verbosity)
            instruction.operator = OPERATORS.EQUALS

        if instruction.connector in CONNECTORS.by_trait['select_cols']:
            instruction.function = _select_cols

        elif instruction.connector in CONNECTORS.by_trait['select_rows']:
            instruction.function = _select_rows

        elif instruction.connector in CONNECTORS.by_trait['select_vals']:
            instruction.function = _select_vals

    else:

        if flags.intersection(FLAGS.by_trait['settings']):
            instruction.function = _modify_settings

        elif flags.intersection(FLAGS.by_trait['metadata']):
            instruction.function = _modify_metadata

        elif flags.intersection(FLAGS.by_trait['format']):
            instruction.function = _modify_format

        elif FLAGS.HEADER in flags:
            instruction.function = _modify_headers

        elif FLAGS.NEW_COL in flags:
            instruction.function = _new_col

        else:
            instruction.function = _modify_vals


    #general checks
    if instruction.operator in OPERATORS.by_trait['unary'] and len(instruction.value) > 0:
        log(f'warning: value {instruction.value} will be ignored for unary operator {instruction.operator}',
            'qp.qlang.parse', verbosity)
        instruction.value = ''
    if flags.intersection(FLAGS.by_trait['copy_df']) and not INPLACE:
        log(f'debug: df will be copied since instruction "{instruction.code}" modifies data',
            'qp.qlang.parse', verbosity)
        settings.copy_df = True


    log(f'trace: parsed instruction: "{instruction.code}"',
        'qp.qlang.parse', verbosity)

    return instruction, settings



def validate(instruction, settings):
    symbols_list = [instruction.connector.name, instruction.operator.name]
    symbols_list.extend([flag.name for flag in instruction.flags])
    symbols = compatible.loc[symbols_list, symbols_list]

    #current approach is fast but not very specific in which symbols are not compatible
    if symbols.all().all():
        log(f'trace: instruction "{instruction.code}" is valid', 'qp.qlang.validate', settings.verbosity)
    else:
        incompatible = list(symbols.index[~symbols.all()])
        log(f'warning: the following symbols are not compatible: {incompatible}',
            'qp.qlang.validate', settings.verbosity)  #wip: switch to error?

    return instruction




##############     selection instructions     ##############


def _select_cols(instruction, df_new, settings):
    """
    An Instruction to select cols fulfilling a condition.
    """
    verbosity = settings.verbosity
    cols = settings.cols
    cols_all = df_new.columns.to_series()
    rows = settings.rows


    if instruction.operator == OPERATORS.TRIM:
        cols_new = settings.vals.any()
        if FLAGS.NEGATE in instruction.flags:
            cols_new = ~cols_new
    elif instruction.operator == OPERATORS.INVERT:
        cols_new = ~cols
    else:
        cols_new = _filter_series(cols_all, instruction, settings, df_new)


    if cols_new.any() == False:
        log(f'warning: no cols fulfill the condition in "{instruction.code}"',
            'qp.qlang._select_cols', verbosity)

    if cols is None: #pragma: no cover (should not happen)
        cols = cols_new
    elif instruction.connector == CONNECTORS.NEW_SELECT_COLS:
        cols = cols_new
    elif instruction.connector == CONNECTORS.AND_SELECT_COLS:
        cols &= cols_new
    elif instruction.connector == CONNECTORS.OR_SELECT_COLS:
        cols |= cols_new


    if cols.any() == False and instruction.connector == CONNECTORS.AND_SELECT_COLS:
        log(f'warning: no cols fulfill the condition in "{instruction.code}" and the previous condition(s)',
            'qp.qlang._select_cols', verbosity)

    settings.cols = cols
    return df_new, settings



def _select_rows(instruction, df_new, settings):
    """
    An Instruction to select rows fulfilling a condition.
    """

    verbosity = settings.verbosity
    flags = instruction.flags
    value = instruction.value
    cols = settings.cols
    rows = settings.rows
    rows_all = df_new.index.to_series()

    if cols.any() == False:
        log(f'error: row selection cannot be applied when no cols where selected', 'qp.qlang._select_rows', verbosity)
        return df_new, settings


    if value.startswith('@'):
        col = value[1:]
        if col in df_new.columns:
            instruction.value = df_new[col]
        else:
            log(f'error: col "{col}" not found in dataframe. cannot use "@{col}" for row selection',
                'qp.qlang._select_rows', verbosity)
            return df_new, settings

    if instruction.operator == OPERATORS.TRIM:
        rows_new = settings.vals.loc[:,cols].any(axis=1)
        if FLAGS.NEGATE in instruction.flags:
            rows_new = ~rows_new
    elif instruction.operator == OPERATORS.INVERT:
        rows_new = ~rows
    elif FLAGS.IDX in flags:
        rows_new = _filter_series(rows_all, instruction, settings, df_new)
    else:
        rows_new = None
        for col in df_new.columns[cols]:
            rows_temp = _filter_series(df_new[col], instruction, settings, df_new)
            if rows_new is None:
                rows_new = rows_temp
            elif FLAGS.ANY in flags:
                rows_new = rows_new | rows_temp
            elif FLAGS.ALL in flags:
                rows_new = rows_new & rows_temp


    if instruction.connector == CONNECTORS.AND_SELECT_ROWS:
        rows = rows & rows_new
    elif instruction.connector == CONNECTORS.OR_SELECT_ROWS:
        rows = rows | rows_new
    elif instruction.connector == CONNECTORS.NEW_SELECT_ROWS:
        rows = rows_new

    settings.rows = rows
    return df_new, settings




def _select_vals(instruction, df_new, settings):
    """
    An Instruction to select vals fulfilling a condition.
    """

    verbosity = settings.verbosity
    flags = instruction.flags
    value = instruction.value

    cols = settings.cols
    rows = settings.rows
    vals = settings.vals
    selection = df_new.loc[rows, cols]
    vals_blank = settings.vals_blank

    if cols.any() == False:
        log(f'error: val selection cannot be applied when no cols where selected', 'qp.qlang._select_vals', verbosity)
        return df_new, settings

    if rows.any() == False:
        log(f'warning: val selection cannot be applied when no rows where selected', 'qp.qlang._select_vals', verbosity)
        return df_new, settings


    if value.startswith('@'):
        col = value[1:]
        if col in df_new.columns:
            instruction.value = df_new.loc[rows, col]
        else:
            log(f'error: col "{col}" not found in dataframe. cannot use "@{col}" for val selection',
                'qp.qlang._select_vals', verbosity)
            return df_new, settings
    

    if instruction.operator == OPERATORS.INVERT:
        vals_new = ~vals
    else:
        vals_new = vals_blank.copy()
        for col in df_new.columns[cols]:
            vals_new.loc[rows, col] = _filter_series(selection[col], instruction, settings, selection)

    if instruction.connector == CONNECTORS.AND_SELECT_VALS:
        vals = vals & vals_new
    elif instruction.connector == CONNECTORS.OR_SELECT_VALS:
        vals = vals | vals_new
    elif instruction.connector == CONNECTORS.NEW_SELECT_VALS:
        vals = vals_new


    settings.cols = cols
    settings.rows = rows
    settings.vals = vals
    return df_new, settings



def _filter_series(series, instruction, settings, df_new=None):
    """
    Filters a pandas series by applying a condition.
    Conditions are made up of a comparison operator and for binary operators a value to compare to.
    FLAGS.NEGATE inverts the result of the condition.
    """
    verbosity = settings.verbosity
    flags = instruction.flags
    operator = instruction.operator
    value = instruction.value
    filtered = None
    

    #regex comparisone
    if FLAGS.REGEX in flags:
        if operator == OPERATORS.EQUALS:
            filtered = series.astype(str).str.fullmatch(value)
        elif operator == OPERATORS.CONTAINS:
            filtered = series.astype(str).str.contains(value)
        else: #pragma: no cover (covered by validate())
            log(f'error: operator "{operator}" is not compatible with regex flag', 'qp.qlang._filter_series', verbosity)


    #eval python expression
    elif operator == OPERATORS.EVAL:
        if FLAGS.COL_EVAL in flags:
            filtered = eval(value, {'col': series, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
        else:
            filtered = series.apply(lambda x: eval(value, {'x': x, 'col': series, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp}))


    #substring comparison
    elif operator == OPERATORS.CONTAINS:
        if FLAGS.STRICT in flags:
            filtered = series.astype(str).str.contains(value, case=True, regex=False)
        else:
            filtered = series.astype(str).str.contains(value, case=False, regex=False)


    #type checks
    elif operator in OPERATORS.by_trait['is_type']:
        if FLAGS.STRICT in flags:
            # if operator == OPERATORS.IS_STR:
            #     filtered = series.apply(lambda x: isinstance(x, str))
            if operator == OPERATORS.IS_INT:
                filtered = series.apply(lambda x: isinstance(x, TYPES_INT))
            elif operator == OPERATORS.IS_FLOAT:
                filtered = series.apply(lambda x: isinstance(x, TYPES_FLOAT))
            elif operator == OPERATORS.IS_NUM:
                filtered = series.apply(lambda x: isinstance(x, TYPES_NUM))
            elif operator == OPERATORS.IS_BOOL:
                filtered = series.apply(lambda x: isinstance(x, TYPES_BOOL))

            # elif operator == OPERATORS.IS_DATETIME:
            #     filtered = series.apply(lambda x: _datetime(x, errors='ERROR')) != 'ERROR'
            # elif operator == OPERATORS.IS_DATE:
            #     filtered = series.apply(lambda x: _date(x, errors='ERROR')) != 'ERROR'

            elif operator == OPERATORS.IS_NA:
                filtered = series.isna()

        else:
            if operator == OPERATORS.IS_STR:
                filtered = series.apply(lambda x: isinstance(x, str))
            elif operator == OPERATORS.IS_INT:
                unrounded = series.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                rounded = unrounded.round(0)
                filtered = rounded == unrounded
            elif operator == OPERATORS.IS_FLOAT:
                filtered = series.apply(lambda x: _float(x, errors='ERROR')) != 'ERROR'
            elif operator == OPERATORS.IS_NUM:
                filtered = series.apply(lambda x: _num(x, errors='ERROR')) != 'ERROR'
            elif operator == OPERATORS.IS_BOOL:
                filtered = series.apply(lambda x: _bool(x, errors='ERROR')) != 'ERROR'

            elif operator == OPERATORS.IS_DATETIME:
                filtered = series.apply(lambda x: _datetime(x, errors='ERROR')) != 'ERROR'
            elif operator == OPERATORS.IS_DATE:
                filtered = series.apply(lambda x: _date(x, errors='ERROR')) != 'ERROR'

            elif operator == OPERATORS.IS_NA:
                filtered = series.apply(lambda x: _na(x, errors='ERROR')) != 'ERROR'


    #categorical checks
    elif operator == OPERATORS.IS_ANY:
        filtered = series.apply(lambda x: True)
    elif operator == OPERATORS.IS_NK:
        filtered = series.apply(lambda x: _nk(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_YN:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR')) != 'ERROR'
    elif operator == OPERATORS.IS_YES:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR', yes='yes')) == 'yes'
    elif operator == OPERATORS.IS_NO:
        filtered = series.apply(lambda x: _yn(x, errors='ERROR', no='no')) == 'no'

    elif operator == OPERATORS.IS_UNIQUE:
        filtered = series.duplicated(keep=False) == False
    elif operator == OPERATORS.IS_FIRST:
        filtered = series.duplicated(keep='first') == False
    elif operator == OPERATORS.IS_LAST:
        filtered = series.duplicated(keep='last') == False


    #type dependant comparison
    elif operator in [
        OPERATORS.BIGGER_EQUAL,
        OPERATORS.SMALLER_EQUAL,
        OPERATORS.BIGGER,
        OPERATORS.SMALLER,
        OPERATORS.EQUALS,
        ]:
        value_type = _type(value)
        log(f'trace: value "{value}" is treated as type "{value_type}" for comparison', 'qp.qlang._filter_series', verbosity)
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
        string_dtypes = ['string']

        if value_type in ('int', 'float', 'num'):
            value = pd.to_numeric(value, errors='coerce')
            if series.dtype not in numeric_dtypes:
                series = pd.to_numeric(series, errors='coerce')

        elif value_type in ('date', 'datetime'):
            value = _datetime(value, errors='ignore')
            if series.dtype not in datetime_dtypes:
                log(f'warning: series "{series.name}" is not a datetime series, consider converting it with "$to datetime;"',
                    'qp.qlang._filter_series', verbosity)
                series = pd.to_datetime(series, errors='coerce')

        elif value_type == 'str':
            if series.dtype not in string_dtypes:
                series = series.astype(str)
            if FLAGS.STRICT not in flags and operator == OPERATORS.EQUALS:
                value = value.lower()
                series = series.str.lower()

        elif isinstance(value, pd.Series):
            if FLAGS.STRICT not in flags and operator == OPERATORS.EQUALS:
                if series.dtype in string_dtypes or value.dtype in string_dtypes:
                    series = series.str.lower()
                    value = value.str.lower()


        if operator == OPERATORS.BIGGER_EQUAL:
            filtered = series >= value
        elif operator == OPERATORS.SMALLER_EQUAL:
            filtered = series <= value
        elif operator == OPERATORS.BIGGER:
            filtered = series > value
        elif operator == OPERATORS.SMALLER:
            filtered = series < value
        elif operator == OPERATORS.EQUALS:
            filtered = series == value

    else: #pragma: no cover  (should not happen)
        log(f'error: could not apply filter condition',
            'qp.qlang._filter_series', verbosity)

    if FLAGS.NEGATE in flags:
        filtered = ~filtered

    return filtered


def _save_selection(instruction, df_new, settings):
    """
    Save the current col, row, val selections as boolean masks.
    """

    value = instruction.value
    if value in settings.saved.keys():
        log(f'warning: a selection was already saved as "{value}". overwriting it',
            'qp.qlang._save_selection', settings.verbosity)

    selection = {
        'rows': settings.rows.copy(),
        'cols': settings.cols.copy(),
        'vals': settings.vals.copy()
        }
    settings.saved[value] = selection

    return df_new, settings


def _load_selection(instruction, df_new, settings):
    """
    loads a previously saved col, row, valselection.
    """

    verbosity = settings.verbosity
    saved = settings.saved
    value = instruction.value
    connector = instruction.connector

    if value in saved.keys():
        selection = saved[value]
    else:
        log(f'error: selection "{value}" is not in saved selections',
            'qp.qlang._load_selection', verbosity)
        return None, settings

    if connector == CONNECTORS.NEW_SELECT_COLS:
        settings.cols = selection['cols']
    elif connector == CONNECTORS.AND_SELECT_COLS:
        settings.cols = settings.cols & selection['cols']
    elif connector == CONNECTORS.OR_SELECT_COLS:
        settings.cols = settings.cols | selection['cols']

    elif connector == CONNECTORS.NEW_SELECT_ROWS:
        settings.rows = selection['rows']
    elif connector == CONNECTORS.AND_SELECT_ROWS:
        settings.rows = settings.rows & selection['rows']
    elif connector == CONNECTORS.OR_SELECT_ROWS:
        settings.rows = settings.rows | selection['rows']
    
    elif connector == CONNECTORS.NEW_SELECT_VALS:
        settings.vals = selection['vals']
    elif connector == CONNECTORS.AND_SELECT_VALS:
        settings.vals = settings.vals & selection['vals']
    elif connector == CONNECTORS.OR_SELECT_VALS:
        settings.vals = settings.vals | selection['vals']

    settings.saved = saved
    return df_new, settings



#############     modification instructions     #############

def _modify_settings(instruction, df_new, settings):
    """
    An instruction to change the query settings.
    """

    verbosity = settings.verbosity
    diff = settings.diff
    value = instruction.value

    if FLAGS.VERBOSITY in instruction.flags:
        if value in ['0', '1', '2', '3', '4', '5']:
            verbosity = int(value)
        else:
            log(f'warning: verbosity must be an integer between 0 and 5. "{value}" is not valid',
                'qp.qlang._modify_settings', verbosity)

    elif FLAGS.DIFF in instruction.flags:
        if value.lower() in ['none', '0', 'false']:
            diff = None
        elif value.lower() in ['mix', 'new', 'old', 'new+']:
            diff = value.lower()
        else:
            log(f'warning: diff must be one of [None, mix, old, new, new+]. "{value}" is not valid',
                'qp.qlang._modify_settings', verbosity)
    else: #pragma: no cover (covered by validate())
        log(f'error: no setting flag found in "{instruction.code}"',
            'qp.qlang._modify_settings', verbosity)

    settings.verbosity = verbosity
    settings.diff = diff
    return df_new, settings


def _modify_metadata(instruction, df_new, settings):
    """
    An Instruction for metadata modification.
    """

    verbosity = settings.verbosity
    cols = settings.cols
    rows = settings.rows
    operator = instruction.operator
    value = instruction.value


    if 'meta' not in df_new.columns:
        log(f'info: no metadata col found in dataframe. creating new col named "meta"',
            'qp.qlang._modify_metadata', verbosity)
        df_new['meta'] = ''
        cols = pd.concat([cols, pd.Series([True])])
        cols.index = df_new.columns


    if FLAGS.TAG_METADATA in instruction.flags:
        tag = ''
        for col in df_new.columns[cols]:
            tag += f'@{col}'
        if operator == OPERATORS.SET:
            df_new.loc[rows, 'meta'] = f'\n{tag}: {value}'
        elif operator == OPERATORS.ADD:
            df_new.loc[rows, 'meta'] += f'\n{tag}: {value}'
        else: #pragma: no cover (covered by validate())
            log(f'error: operator "{operator}" is not compatible with TAG_METADATA flag', 'qp.qlang._modify_metadata', verbosity)

    elif operator == OPERATORS.SET:
        df_new.loc[rows, 'meta'] = value

    elif operator == OPERATORS.ADD:
        df_new.loc[rows, 'meta'] += value

    elif operator == OPERATORS.EVAL:
        if FLAGS.COL_EVAL in instruction.flags:
            df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].apply(lambda x: eval(value, {'col': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
        elif pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
            if 'x' in value:  #needs to be evaluated for each value
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].map(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:  #only needs to be evaluated once
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].map(lambda x: eval_result)
        else: #pragma: no cover (tests only run with pandas >= 2.1.0)
            if 'x' in value:
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].applymap(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                df_new.loc[rows, 'meta'] = df_new.loc[rows, 'meta'].applymap(lambda x: eval_result)

    else: #pragma: no cover (covered by validate())
        log(f'error: operator "{operator}" is not compatible with metadata modification',
            'qp.qlang._modify_metadata', verbosity)


    settings.cols = cols
    return df_new, settings


def _modify_format(instruction, df_new, settings):
    """
    changes visual formatting of the current selection.
    """

    verbosity = settings.verbosity
    cols = settings.cols
    rows = settings.rows
    vals = settings.vals
    vals_temp = settings.vals_blank.copy()
    vals_temp.loc[rows, cols] = vals.loc[rows, cols]

    style = settings.style
    flags = instruction.flags
    value = instruction.value



    if not isinstance(style, pd.DataFrame):
        style = pd.DataFrame('', columns=df_new.columns, index=df_new.index)

    for col in df_new.columns[cols]:
        if col not in style.columns:
            style[col] = ''


    if FLAGS.COLOR in flags:
        style[vals_temp] += f'color: {value};'

    elif FLAGS.BACKGROUND_COLOR in flags: #pragma: no cover (visual changes are currently not tested)
        style[vals_temp] += f'background-color: {value};'

    elif FLAGS.ALIGN in flags: #pragma: no cover (visual changes are currently not tested)
        if value in ['left', 'right', 'center', 'justify']:
            style[vals_temp] += f'text-align: {value};'
        elif value in ['top', 'middle', 'bottom']:
            style[vals_temp] += f'vertical-align: {value};'
        else:
            log(f'warning: alignment "{value}" is not valid. must be one of [left, right, center, justify, top, middle, bottom]',
                'qp.qlang._modify_format', verbosity)

    elif FLAGS.WIDTH in flags: #pragma: no cover (visual changes are currently not tested)
        if not value.endswith(('px', 'cm', 'mm', 'in', 'pt', 'pc', 'em', 'ex', 'ch', 'rem', 'vw', 'vh', 'vmin', 'vmax', '%')):
            log(f'info: no unit for col width was used. defaulting to "px". other options: [cm, mm, in, pt, pc, em, ex, ch, rem, vw, vh, vmin, vmax, %]',
                'qp.qlang._modify_format', verbosity)
            value += 'px'
        style[vals_temp] += f'width: {value};'


    elif FLAGS.CSS in flags: #pragma: no cover (visual changes are currently not tested)
        if not value.endswith(';'):
            value += ';'
        style[vals_temp] += value

    else: #pragma: no cover (should not happen)
        log(f'error: no format flag found in "{instruction.code}"',
            'qp.qlang._modify_format', verbosity)


    settings.style = style
    return df_new, settings


def _modify_headers(instruction, df_new, settings):
    """
    An Instruction to modify the headers of the selected col(s).
    """

    verbosity = settings.verbosity
    cols = settings.cols
    vals = settings.vals
    vals_blank = settings.vals_blank.copy()
    operator = instruction.operator
    value = instruction.value

    if cols.any() == False:
        log(f'error: header modification cannot be applied when no cols where selected', 'qp.qlang._modify_headers', verbosity)
        return df_new, settings
    

    if operator == OPERATORS.SET:
        col_mapping = {col: value for col in df_new.columns[cols]}

    elif operator == OPERATORS.ADD:
        col_mapping = {col: col + value for col in df_new.columns[cols]}

    elif operator == OPERATORS.EVAL:
        col_mapping = {
                col: eval(value, {'x': col, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp})
                for col in df_new.columns[cols]
                }

    else: #pragma: no cover (covered by validate())
        log(f'error: operator "{operator}" is not compatible with header modification',
            'qp.qlang._modify_headers', verbosity)


    df_new.rename(columns=col_mapping, inplace=True)
    cols.index = df_new.columns
    vals.rename(columns=col_mapping, inplace=True)
    vals_blank.rename(columns=col_mapping, inplace=True)
    for selection in settings.saved.values():
        selection['cols'].index = df_new.columns
        selection['vals'].rename(columns=col_mapping, inplace=True)
    

    settings.cols = cols
    settings.vals = vals
    settings.vals_blank = vals_blank
    return df_new, settings


def _modify_vals(instruction, df_new, settings):
    """
    An Instruction to modify the selected values.
    """

    verbosity = settings.verbosity
    cols = settings.cols
    rows = settings.rows
    vals = settings.vals
    vals_temp = settings.vals_blank.copy()
    vals_temp.loc[rows, cols] = vals.loc[rows, cols]

    if vals_temp.any().any() == False:
        log(f'warning: value modification cannot be applied when no values where selected', 'qp.qlang._modify_vals', verbosity)
        return df_new, settings

    operator = instruction.operator
    value = instruction.value

    if value.startswith('@'):
        col = value[1:]
        if col in df_new.columns:
            value = df_new.loc[rows, col]
        else:
            log(f'error: col "{col}" not found in dataframe. cannot use "@{col}" for val modification',
                'qp.qlang._modify_vals', verbosity)
            return df_new, settings

    type_conversions = {
        OPERATORS.TO_STR: str,
        OPERATORS.TO_INT: _int,
        OPERATORS.TO_FLOAT: _float,
        OPERATORS.TO_NUM: _num,
        OPERATORS.TO_BOOL: _bool,
        OPERATORS.TO_DATETIME: _datetime,
        OPERATORS.TO_DATE: _date,
        OPERATORS.TO_NA: _na,
        OPERATORS.TO_NK: _nk,
        OPERATORS.TO_YN: _yn,
        }
    dtype_conversions = {
        OPERATORS.TO_STR: str,
        OPERATORS.TO_INT: 'Int64',
        OPERATORS.TO_FLOAT: 'Float64',
        OPERATORS.TO_NUM: 'object',
        OPERATORS.TO_BOOL: 'bool',
        OPERATORS.TO_DATETIME: 'datetime64[ns]',
        OPERATORS.TO_DATE: 'datetime64[ns]',
        OPERATORS.TO_NK: 'object',
        OPERATORS.TO_YN: 'object',
        }
    

    #data modification  
    if operator == OPERATORS.SET:
        if isinstance(value, pd.Series):
            df_temp = pd.DataFrame({colname: value for colname in df_new.columns})
            df_new[vals_temp] = df_temp
        else:
            df_new[vals_temp] = value

    elif operator == OPERATORS.ADD:
        if isinstance(value, pd.Series):
            df_temp = pd.DataFrame({colname: value for colname in df_new.columns})
            df_new[vals_temp] = df_new.astype(str) + df_temp.astype(str)
        else:
            df_new[vals_temp] = df_new[vals_temp].astype(str) + value

    elif FLAGS.COL_EVAL in instruction.flags:
        if operator == OPERATORS.EVAL:
            changed = df_new.loc[rows, cols].apply(lambda x: eval(value, {'col': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}), axis=0)
            df_new = df_new.mask(vals_temp, changed)
        else: #pragma: no cover (covered by validate())
            log(f'error: operator "{operator}" is not compatible with COL_EVAL flag', 'qp.qlang._modify_vals', verbosity)

    #wip: reintroduce/change regex flag?
    # elif FLAGS.REGEX in instruction.flags:
    #     if operator == OPERATORS.SET:
    #         rows = mask_temp.any(axis=1)
    #         for col in df_new.columns[cols]:
    #             df_new.loc[rows, col] = df_new.loc[rows, col].str.extract(value).loc[rows, 0]
    #     else: #pragma: no cover (covered by validate())
    #         log(f'error: operator "{operator}" is not compatible with regex flag', 'qp.qlang._modify_vals', verbosity)

    elif operator == OPERATORS.SORT:
        if FLAGS.NEGATE in instruction.flags:
            df_new.sort_values(by=list(df_new.columns[cols]), axis=0, ascending=False, inplace=True)
        else:
            df_new.sort_values(by=list(df_new.columns[cols]), axis=0, inplace=True)


    elif pd.__version__ >= '2.1.0':  #map was called applymap before 2.1.0
        #data modification
        if operator == OPERATORS.EVAL:
            if 'x' in value:  #needs to be evaluated for each value
                changed = df_new.loc[rows, cols].map(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:  #only needs to be evaluated once
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                changed = df_new.loc[rows, cols].map(lambda x: eval_result)  #setting would be faster but map is dtype compatible
            df_new = df_new.mask(vals_temp, changed)

        #type conversion
        elif operator in type_conversions:
            changed = df_new.loc[rows, cols].map(lambda x: type_conversions[operator](x))
            df_new.loc[rows, cols] = changed
            if operator in dtype_conversions:
                for col in df_new.columns[cols]:
                    df_new[col] = df_new[col].astype(dtype_conversions[operator])

    else: #pragma: no cover (tests only run with pandas >= 2.1.0)
        #data modification
        if operator == OPERATORS.EVAL:
            if 'x' in value:  #needs to be evaluated for each value
                changed = df_new.loc[rows, cols].applymap(lambda x: eval(value, {'x': x, 'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re}))
            else:  #only needs to be evaluated once
                eval_result = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp, 're': re})
                changed = df_new.loc[rows, cols].applymap(lambda x: eval_result)  #setting would be faster but map is dtype compatible
            df_new = df_new.mask(vals_temp, changed)

        #type conversion
        elif operator in type_conversions:
            changed = df_new.loc[rows, cols].applymap(lambda x: type_conversions[operator](x))
            df_new.loc[rows, cols] = changed
            if operator in dtype_conversions:
                for col in df_new.columns[cols]:
                    df_new[col] = df_new[col].astype(dtype_conversions[operator])

        else:
            log(f'error: operator "{operator}" is not compatible with value modification',
                'qp.qlang._modify_vals', verbosity)

    return df_new, settings


def _new_col(instruction, df_new, settings):
    """
    An Instruction to add a new col.
    """

    verbosity = settings.verbosity
    cols = settings.cols
    rows = settings.rows
    vals = settings.vals
    vals_blank = settings.vals_blank
    operator = instruction.operator
    value = instruction.value

    if value.startswith('@'):
        col = value[1:]
        if col in df_new.columns:
            value = df_new[col]
        else:
            log(f'error: col "{col}" not found in dataframe. cannot add a new col thats a copy of it',
                'qp.qlang._new_col', verbosity)

    for i in range(1, 1001):
        if i == 1000: #pragma: no cover (unlikely to happen)
            log(f'error: could not add new col. too many cols named "new<x>"',
                'qp.qlang._new_col', verbosity)
            return df_new, settings
        header = 'new' + str(i)
        if header not in df_new.columns:
            df_new[header] = pd.NA
            break

    if operator == OPERATORS.SET:
        pass

    elif operator == OPERATORS.EVAL:
        value = eval(value, {'df': df_new, 'pd': pd, 'np': np, 'qp': qp})

    if isinstance(value, pd.Series):
        df_new[header] = value
    else:
        df_new.loc[rows, header] = value

    cols = pd.Series([True if col == header else False for col in df_new.columns])
    cols.index = df_new.columns
    vals[header] = False
    vals_blank[header] = False
    for selection in settings.saved.values():
        selection['cols'] = pd.concat((selection['cols'], pd.Series({header: False})))
        selection['vals'][header] = False

    settings.cols = cols
    settings.vals = vals
    settings.vals_blank = vals_blank
    return df_new, settings





###################     df accessors     ###################


@pd.api.extensions.register_dataframe_accessor('check')
class DataFrameCheck:
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def __call__(self, verbosity=3):
        check_df(self.df, verbosity=verbosity)
        return self.df


@pd.api.extensions.register_dataframe_accessor('q')
class DataFrameQuery:
    """
    A query language for pandas data exploration/analysis/modification.
    df.qi() without any args can be used to interactively build a query in Jupyter notebooks.


    examples:

    #select col
    df.q('id')
    df.q('%id')  #equivalent
    df.q('%=id') #equivalent
    df.q('%==id') #equivalent
    df.q('% == id') #equivalent

    #select multiple cols
    df.q('id  /name')

    #select rows in a col which fullfill a condition
    df.q('id  %%>20000')

    #select rows fullfilling multiple conditions in the same col
    df.q('id  %%>20000   &&<30000')

    #select rows fullfilling both conditions in different cols
    df.q('id  %%>20000    %name   &&?john')

    #select rows fullfilling either condition in different cols
    df.q('id   %%>20000   %name   //?john')

    #select values
    df.q('%%%>0;  &&&<100')

    #highlight values using background color
    df.q('%%%is na;  $bg=red')
    """

    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return 'docstring of dataframe accessor pd_object.q():\n' + str(self.__doc__)

    def __call__(self, code=''):
        return query(self.df, code)


@pd.api.extensions.register_dataframe_accessor('qi')
class DataFrameQueryInteractiveMode: #pragma: no cover (dynamic UI is not tested)
    """
    Interactive version of df.q() for building queries in Jupyter notebooks.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self):
        kwargs = {'df': fixed(self.df), 'code': ''}

        #code input
        readme = '$verbosity=3\n$diff=None\n\n#Enter query code here,\n' \
                + '#or use the buttons to the right to create a query.\n' \
                + '#hover over buttons for tooltips.\n\n'
        ui_code = widgets.Textarea(
            value=readme,
            layout=Layout(width='35%', height='97%')
            )



        #syntax


        def get_symbols_compatible():
            if ui_connectors.value is None:
                symbols_compatible_with_connector = set(compatible.index)
            else:
                compatible_with_connector = compatible[ui_connectors.value.name]
                symbols_compatible_with_connector = set(compatible[compatible_with_connector].index)

            if ui_operators.value is None:
                symbols_compatible_with_operator = set(compatible.index)
            else:
                compatible_with_operator = compatible[ui_operators.value.name]
                symbols_compatible_with_operator = set(compatible[compatible_with_operator].index)

            return symbols_compatible_with_connector.intersection(symbols_compatible_with_operator)


        def build_instruction(ui_connectors, ui_operators, ui_flags):
            if ui_connectors.value is None:
                connector = ''
            else:
                connector = ui_connectors.value.glyph + ' '

            if ui_operators.value is None:
                operator = ''
            else:
                operator = ui_operators.value.glyph + ' '

            if ui_flags.value is None:
                flag = ''
            else:
                flag = ui_flags.value.glyph + ' '

            return f'{connector}{flag}{operator}'
     

        def new_connector(value):

            ui_operators.value = None
            ui_flags.value = None
            ui_instruction.value = ui_connectors.value.glyph

            symbols_compatible = get_symbols_compatible()

            operators_compatible = sorted([operator for operator in OPERATORS if operator.name in symbols_compatible])
            ui_operators.options = [(symbol.glyph, symbol) for symbol in operators_compatible]
            ui_operators.tooltips = [symbol.description for symbol in operators_compatible]

            flags_compatible = sorted([flag for flag in FLAGS if flag.name in symbols_compatible])
            ui_flags.options = [(symbol.glyph, symbol) for symbol in flags_compatible]
            ui_flags.tooltips = [symbol.description for symbol in flags_compatible]

        
        def new_operator(value):

            symbols_compatible = get_symbols_compatible()

            flag_old = ui_flags.value
            flags_compatible = sorted([flag for flag in FLAGS if flag.name in symbols_compatible])
            ui_flags.options = [(symbol.glyph, symbol) for symbol in flags_compatible]
            ui_flags.tooltips = [symbol.description for symbol in flags_compatible]
            if flag_old in flags_compatible:
                ui_flags.value = flag_old
            else:
                ui_flags.value = None
            
            ui_instruction.value = build_instruction(ui_connectors, ui_operators, ui_flags)

            if ui_operators.value in OPERATORS.by_trait['unary']:
                ui_instruction_value.value = ''
                ui_instruction_value.disabled = True
                ui_instruction_value.layout.visibility = 'hidden'
                ui_instruction_value.layout.width = '0px'
            else:
                ui_instruction_value.disabled = False
                ui_instruction_value.layout.visibility = 'visible'
                ui_instruction_value.layout.width = 'auto'

    
        def new_flag(value):
            ui_instruction.value = build_instruction(ui_connectors, ui_operators, ui_flags)


        connectors = sorted(list(CONNECTORS))
        ui_connectors = widgets.ToggleButtons(
            options=[(symbol.glyph, symbol) for symbol in connectors],
            description='connectors:',
            value=None,
            tooltips=[symbol.description for symbol in connectors],
            layout=Layout(width='auto', height='auto'),
            )
        ui_connectors.style.button_width = 'auto'
        ui_connectors.observe(new_connector, names='value')


        operators = sorted(list(OPERATORS))
        ui_operators = widgets.ToggleButtons(
            options=[(symbol.glyph, symbol) for symbol in operators],
            description='operators:',
            value=None,
            tooltips=[symbol.description for symbol in operators],
            layout=Layout(width='auto', height='auto'),
            )
        ui_operators.style.button_width = 'auto'
        ui_operators.observe(new_operator, names='value')


        flags = sorted(list(FLAGS))
        ui_flags = widgets.ToggleButtons(
            options=[(symbol.glyph, symbol) for symbol in flags],
            description='flags:',
            value=None,
            tooltips=[symbol.description for symbol in flags],
            layout=Layout(width='auto', height='auto'),
            )
        ui_flags.style.button_width = 'auto'
        ui_flags.observe(new_flag, names='value')
        

        ui_instruction = widgets.HTML(value='',)
        
        ui_instruction_value = widgets.Text(
            value='',
            disabled=True,
            layout=Layout(width='0px', visibility='hidden')
            )
        
        def add_instruction(test):
            ui_code.value += f'\n{ui_instruction.value} {ui_instruction_value.value}'
        ui_button = widgets.Button(
            description='add',
            button_style='success',
            tooltip='add instruction to query code',
            )
        ui_button.on_click(add_instruction)


        ui_builder = widgets.VBox([
            widgets.HBox([
                ui_button,
                ui_instruction,
                ui_instruction_value,
                ]),
            ui_connectors,
            ui_operators,
            ui_flags,
            ])


        #some general info and statistics about the df
        mem_usage = self.df.memory_usage().sum() / 1024
        ui_details = widgets.HTML(
            value=f"""
            <b>rows:</b> {len(self.df.index)}<br>
            <b>cols:</b> {len(self.df.columns)}<br>
            <b>memory usage:</b> {mem_usage:,.3f}kb<br>
            <b>unique values:</b> {self.df.nunique().sum()}<br>
            <b>missing values:</b> {self.df.isna().sum().sum()}<br>
            <b>cols:</b><br> {'<br>'.join([f'{col} ({dtype})' for col, dtype in list(zip(self.df.columns, self.df.dtypes))])}<br>
            """
            )


        ui_tabs = widgets.Tab(
            children=[
                ui_builder,
                ui_details,
                widgets.HTML(value=DataFrameQuery.__doc__.replace('\n', '<br>').replace('    ', '&emsp;')),
                ],
            titles=['query builder', 'df details', 'readme'],
            layout=Layout(width='50%', height='94%')
            )
        ui = HBox([ui_code, ui_tabs], layout=Layout(width='100%', height='330px'))

        kwargs['code'] = ui_code
        display(ui)
        out = HBox([interactive_output(_interactive_mode, kwargs)], layout=Layout(overflow_y='auto'))
        display(out)

def _interactive_mode(**kwargs): #pragma: no cover (dynamic UI is not tested)
    df = kwargs.pop('df')
    code = kwargs.pop('code')
    result = query(df, code)
    display(result)
    return result



