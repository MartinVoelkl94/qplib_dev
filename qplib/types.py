import re
import pandas as pd
import numpy as np
import datetime
from pandas import isna

#Mostly wrappers for pandas functions but with some additional
#functionality and generally more lenient handling of edge cases.


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
    except Exception as e:
        if errors == 'raise':
            raise ValueError(
                f'could not convert "{x}" to integer.\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                f'original error:\n{e}'
                )
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
    except Exception as e:
        if errors == 'raise':
            raise ValueError(
                f'could not convert "{x}" to float.\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                f'original error:\n{e}'
                )
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
    except Exception as e:
        if errors == 'raise':
            raise ValueError(
                f'could not convert "{x}" to numeric value.\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                f'original error:\n{e}'
                )
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
            raise ValueError(
                f'could not convert "{x}" to numeric boolean.\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                )
        elif errors == 'ignore':
            return x
        elif errors == 'coerce':
            return na
        else:
            return errors


months_txt = (
    'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
    'January|February|March|April|May|June|July|'
    'August|September|October|November|December'
    )
def _date(x, errors='coerce', na=pd.NaT):
    """
    recognizes and converts potential dates
    between 0001-01-01 and 2999-12-31 in various formats.
    """
    result = _datetime(x, errors=errors, na=na)
    if result is na:
        return na
    elif isinstance(result, datetime.datetime) or isinstance(result, pd.Timestamp):
        return result.date()

    #raise and coerce are handled in _datetime and should not be reached here
    elif errors == 'raise':  #pragma: no cover
        raise ValueError(
            f'could not convert "{x}" to datetime.\n'
            'Error handling:\n'
            'errors="raise": raises a ValueError\n'
            'errors="ignore": returns the original value\n'
            'errors="coerce": returns np.nan\n'
            'errors=<any other value>: returns <any other value>\n'
            )
    elif errors == 'ignore':
        return x
    elif errors == 'coerce':
        return na  #pragma: no cover
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

        elif re.match(
                f'(\\d\\d\\d\\d)\\D?({months_txt})\\D?(\\d\\d)',
                x,
                flags=re.IGNORECASE,
                ):
            x = re.sub(
                f'(\\d\\d\\d\\d)\\D?({months_txt})\\D?(\\d\\d)(.*)',
                r'\3-\2-\1\4',
                x,
                flags=re.IGNORECASE,
                )
            result = pd.to_datetime(x, dayfirst=True)

        elif re.match(
                f'(\\d\\d)\\D?({months_txt})\\D?(\\d\\d\\d\\d)',
                x,
                flags=re.IGNORECASE,
                ):
            x = re.sub(
                f'(\\d\\d)\\D?({months_txt})\\D?(\\d\\d\\d\\d)(.*)',
                r'\1-\2-\3\4',
                x,
                flags=re.IGNORECASE,
                )
            result = pd.to_datetime(x, dayfirst=True)

        elif re.match(
                f'({months_txt})\\D?(\\d\\d)\\D?(\\d\\d\\d\\d)',
                x,
                flags=re.IGNORECASE,
                ):
            x = re.sub(
                f'({months_txt})\\D?(\\d\\d)\\D?(\\d\\d\\d\\d)(.*)',
                r'\2-\1-\3\4',
                x,
                flags=re.IGNORECASE,
                )
            result = pd.to_datetime(x, dayfirst=True)

        else:
            result = pd.to_datetime(x, dayfirst=True)

        if result is pd.NaT:
            raise ValueError(f'could not convert "{x}" to date')
        else:
            return result
    except Exception as e:
        if errors == 'raise':
            raise ValueError(
                f'could not convert "{x}" to datetime.\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                f'original error:\n{e}'
                )
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
            raise ValueError(
                f'could not convert "{x}" to "{na}".\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                )
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
            raise ValueError(
                f'could not convert "{x}" to "{nk}".\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                )
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
            raise ValueError(
                f'could not convert "{x}" to "{yes}" or "{no}".\n'
                'Error handling:\n'
                'errors="raise": raises a ValueError\n'
                'errors="ignore": returns the original value\n'
                'errors="coerce": returns np.nan\n'
                'errors=<any other value>: returns <any other value>\n'
                )
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
        elif (
                re.search(r'\d{2}:\d{2}:\d[\d\.:]', x.strip(), re.IGNORECASE)
                and _datetime(x) is not pd.NaT
                ):
            return 'datetime'
        elif _date(x) is not pd.NaT:
            return 'date'

        else:
            try:
                x = pd.to_numeric(x)
                return 'num'
            except Exception:
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



class Container:
    """
    lightweight attribute-centric datastructure:
    - stores data as attributes
    - prevents collisions with class methods and reserved attributes
    - provides dict-like introspection via keys(), values(), items()
    - debug-friendly self referential representation
    """

    #essential methods
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __setattr__(self, name, value):
        if name in dir(self.__class__):
            raise AttributeError(f"{name!r} is a reserved attribute/method name.")
        super().__setattr__(name, value)

    #dict-like interface
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f"Container keys must be strings, not {type(key)}.")
        self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    #list/iterator interface
    def append(self, value):
        counter = 0
        key = 'i' + str(counter)
        while key in self.__dict__:
            counter += 1
            key = 'i' + str(counter)
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    #utility
    def __repr__(self):
        content = []
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                v = str(v).replace('\n', '\n    ')
                content.append(f'  {k} = {v},')
            else:
                content.append(f'  {k} = {v!r},')
        return 'Container(\n' + '\n'.join(content) + '\n  )'

    def __eq__(self, value):
        if not isinstance(value, self.__class__):
            return False
        else:
            return self.__dict__ == value.__dict__

    def clear(self):
        for key in list(self.__dict__.keys()):
            del self.__dict__[key]


msg_reserved_attr = (
    ' is a reserved attribute and cannot be'
    ' modified or used as a key or attribute name.'
    'view all using ".reserved_attributes".'
    )
class _dict(dict):
    """
    Dictionary with some extra features:
    - attributes can be set (if not needed to preserve regular dict functionality)
    - qp.dict().values_flat() will unpack nested iterables
    - qp.dict().invert() will invert the key:value pairs to value:key pairs
    """

    #handling equivalence of attributes and items

    _reserved_attributes = dict().__dir__() + [
        #not included in dict().__dir__() but still present
        '__dict__',
        '__weakref__',

        #custom
        '_reserved_attributes',
        'values_flat',
        'invert',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self.__setitemattr__(key, value)

    def __setitemattr__(self, key, value):
        if key in self._reserved_attributes:
            raise AttributeError(f'"{key}"' + msg_reserved_attr)
        elif not isinstance(key, str):
            raise AttributeError('keys and attribute names must be strings')
        else:
            super().__setitem__(key, value)
            super().__setattr__(key, value)

    def __setitem__(self, key, value):
        self.__setitemattr__(key, value)

    def __setattr__(self, name, value):
        self.__setitemattr__(name, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        super().__delattr__(key)

    def __delattr__(self, name):
        super().__delattr__(name)
        super().__delitem__(name)

    def pop(self, key):
        super().__delattr__(key)
        return super().pop(key)

    def popitem(self):
        key, value = super().popitem()
        super().__delattr__(key)
        return key, value

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        for key, value in self.items():
            self.__setitemattr__(key, value)

    def fromkeys(self, *args, **kwargs):
        return _dict(super().fromkeys(*args, **kwargs))

    def clear(self):
        for key in self.keys():
            super().__delattr__(key)
        super().clear()

    def copy(self):
        return _dict(super().copy())

    #extra methods

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
        try:
            inverted = _dict({val: key for key, val in self.items()})
        except AttributeError as e:
            if str(e).endswith(msg_reserved_attr):
                msg = (
                    'Cannot invert dictionary with reserved'
                    ' attributes as values.'
                    )
                raise AttributeError(msg)
            else:
                raise e
        return inverted
