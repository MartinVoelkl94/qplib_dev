import re
import pandas as pd
import numpy as np
import datetime
from pandas import isna

#Mostly wrappers for pandas functions but with some additional functionality and generally more lenient handling of edge cases.


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

VALUES_NA = (
    '',
    'na',
    'n/a',
    'n.a',
    'n.a.',
    'na.',
    'n.a',
    'nan',
    'n.a.n',
    'n.a.n.',
    'not available',
    'not applicable',
    'not a number',
    'missing',
    'missing.',
    'null',
    'nil',
    'none',
    'void',
    'blank',
    'empty',
    )
VALUES_NK = (
    'unk',
    'unknown',
    'not known',
    'not known.',
    'nk',
    'n.k.',
    'n.k',
    'n/k',
    'not specified',
    'not specified.',
    )


def _int(x, errors='coerce', na=np.nan):
    if isinstance(x, TYPES_INT):
        return x
    try:
        return round(float(x))  #float first to handle strings like '1.0'
    except:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to integer.
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns np.nan
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


def _float(x, errors='coerce', na=np.nan):
    if isinstance(x, TYPES_FLOAT):
        return x
    try:
        return float(x)
    except:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to float.
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns np.nan
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


def _num(x, errors='coerce', na=np.nan):
    if isinstance(x, TYPES_NUM):
        return x
    try:
        return pd.to_numeric(x)
    except:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to numeric.
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns np.nan
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


def _bool(x, errors='coerce', na=None):
    if isinstance(x, TYPES_BOOL):
        return x
    elif str(x).lower() in ['y', 'yes', 'true', '1', '1.0', 'positive', 'pos']:
        return True
    elif str(x).lower() in ['n', 'no', 'false', '0', '0.0', 'negative', 'neg']:
        return False
    else:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to boolean.
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns None
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


months_txt = 'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December'
def _date(x, errors='coerce', na=pd.NaT):
    """
    recognizes and converts potential dates between 0001-01-01 and 2999-12-31 in various formats.
    """
    result = _datetime(x, errors=errors, na=na)
    if result is na:
        return na
    elif isinstance(result, datetime.datetime) or isinstance(result, pd.Timestamp):
        return result.date()
    
    #raise and coerce are handled in _datetime and should not be reached here
    elif errors == 'raise': #pragma: no cover
        raise ValueError(f"""could not convert "{x}" to datetime.
            Error handling:
            errors='raise': raises a ValueError
            errors='ignore': returns the original value
            errors='coerce': returns pd.NaT
            errors=<any other value>: returns <any other value>
            """)
    elif errors == 'ignore':
        return x
    elif errors == 'coerce':
        return na #pragma: no cover
    else:
        return errors


def _datetime(x, errors='coerce', na=pd.NaT):
    if isinstance(x, datetime.datetime):
        return x
    elif isinstance(x, datetime.date):
        return pd.to_datetime(x)
    elif isinstance(x, str):
        x = x.replace('.', '-')
        x = x.replace('/', '-')
        x = x.replace('\\', '-')
        x = x.replace('_', '-')
    try:
        
        if re.fullmatch(r'[012]\d\d\d[-\d\s:]+.*', x):
            result = pd.to_datetime(x, dayfirst=False)
        
        elif re.match(f'(\\d\\d\\d\\d)\\D?({months_txt})\\D?(\\d\\d)', x, flags=re.IGNORECASE):
            x = re.sub(f'(\\d\\d\\d\\d)\\D?({months_txt})\\D?(\\d\\d)(.*)', r'\3-\2-\1\4', x, flags=re.IGNORECASE)
            result = pd.to_datetime(x, dayfirst=True)
        
        elif re.match(f'(\\d\\d)\\D?({months_txt})\\D?(\\d\\d\\d\\d)', x, flags=re.IGNORECASE):
            x = re.sub(f'(\\d\\d)\\D?({months_txt})\\D?(\\d\\d\\d\\d)(.*)', r'\1-\2-\3\4', x, flags=re.IGNORECASE)
            result = pd.to_datetime(x, dayfirst=True)
        
        elif re.match(f'({months_txt})\\D?(\\d\\d)\\D?(\\d\\d\\d\\d)', x, flags=re.IGNORECASE):
            x = re.sub(f'({months_txt})\\D?(\\d\\d)\\D?(\\d\\d\\d\\d)(.*)', r'\2-\1-\3\4', x, flags=re.IGNORECASE)
            result = pd.to_datetime(x, dayfirst=True)
        
        else:
            result = pd.to_datetime(x, dayfirst=True)
        
        if result is pd.NaT:
            raise ValueError(f'could not convert "{x}" to date')
        else:
            return result
    except:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to datetime.
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns pd.NaT
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


def _na(x, errors='ignore', na=None):

    if str(x).lower().strip() in VALUES_NA:
        return na
    elif isna(x):
        return na
    else:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to "{na}".
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns None
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return None
        else:
            return errors


def _nk(x, errors='ignore', nk='unknown', na=None):
    if str(x).lower().strip() in VALUES_NK:
        return nk
    else:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to "{nk}".
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns None
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


def _yn(x, errors='coerce', yes='yes', no='no', na=None):
    if str(x).lower() in ['y', 'yes', 'true', '1', '1.0', 'positive', 'pos']:
        return yes
    elif str(x).lower() in ['n', 'no', 'false', '0', '0.0', 'negative', 'neg']:
        return no
    else:
        if errors == 'raise':
            raise ValueError(f"""could not convert "{x}" to "{yes}" or "{no}".
                Error handling:
                errors='raise': raises a ValueError
                errors='ignore': returns the original value
                errors='coerce': returns NaN
                errors=<any other value>: returns <any other value>
                """)
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors



def _type(x):
    """
    Returns what type something "should" be. e.g.: qp.type('1') == 'int'
    """
    
    if isinstance(x, bool):
        return 'bool'
    elif isinstance(x, TYPES_INT):  #type: ignore  (turns of pylance for this line)
        return 'int'
    elif isinstance(x, TYPES_FLOAT):  #type: ignore  (turns of pylance for this line)
        return 'float'
    
    elif isinstance(x, str):
        if re.fullmatch(r'(true|false)', x.strip(), re.IGNORECASE):
            return 'bool'
        elif re.fullmatch(r'\d+', x.strip()):
            return 'int'
        elif re.fullmatch(r'\d+\.\d+', x.strip()):
            return 'float'
        

        elif re.search(r'\d{2}:\d{2}:\d[\d\.:]', x.strip(), re.IGNORECASE) \
            and _datetime(x) is not pd.NaT:
            return 'datetime'
        elif _date(x) is not pd.NaT:
            return 'date'
     
        else:
            try:
                x = pd.to_numeric(x)
                return 'num'
            except:
                return 'str'
                     
    else:
        return type(x).__name__


def _convert(value, errors='coerce', na=None):
    """
    Converts to the type something "should" be according to qp.type().
    e.g.: qp.convert('1') == 1
    """
    mapping = {
        'int': _int,
        'float': _float,
        'num': _num,
        'bool': _bool,
        'date': _date,
        'datetime': _datetime,
        'na': _na,
        'nk': _nk,
        'yn': _yn,
        'NoneType': lambda x, errors, na: na,
        }
    type_qp = _type(value)
    if type_qp == 'str':
        result = str(value)
    elif type_qp in mapping:
        result = mapping[type_qp](value, errors, na)
    else:
        result = value
    return result


class _dict(dict):
    """
    Dictionary with some extra features:
    - attributes can be set (if not needed to preserve regular dict functionality)
    - qp.dict().values_flat() will unpack nested iterables
    - qp.dict().invert() will invert the key:value pairs to value:key pairs
    """

    def __setattr__(self, name, value):
        if name in dict().__dir__():
            msg = f'Attribute "{name}" is needed for regular dict functionality and cannot be modified'
            raise AttributeError(msg)
        else:
            super().__setattr__(name, value)

    def values_flat(self):
        values_flat = []
        for val in self.values():
            if isinstance(val, dict):
                values_flat.extend(_dict(val).values_flat())
            elif hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                for item in val:
                    if isinstance(item, dict):
                        values_flat.extend(_dict(item).values_flat())
                    else:
                        values_flat.append(item)
            else:
                values_flat.append(val)
        return values_flat
    
    def invert(self):
        return _dict({val:key for key,val in self.items()})
