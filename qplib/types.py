import pandas as pd
import numpy as np
import re

#these are mostly wrappers type conversions with extra features for dealing with errors

def _int(x, errors='coerce', na=np.nan):
    try:
        return int(float(x))  #float first to handle strings like '1.0'
    except:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to integer.
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns np.nan
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors

def _float(x, errors='coerce', na=np.nan):
    try:
        return float(x)
    except:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to float.
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns np.nan
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors
            
def _num(x, errors='coerce', na=np.nan):
    try:
        return pd.to_numeric(x)
    except:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to numeric.
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns np.nan
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors
            
def _bool(x, errors='coerce', na=None):
    if str(x).lower() in ['y', 'yes', 'true', '1', '1.0', 'positive', 'pos']:
        return True
    elif str(x).lower() in ['n', 'no', 'false', '0', '0.0', 'negative', 'neg']:
        return False
    else:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to boolean.
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns None
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors


def _date(x, errors='coerce', na=pd.NaT):
    if isinstance(x, str):
        x = x.replace('_', '-')
    try:
        if re.match(r'\D*(1|2)\d\d\d', x):
            return pd.to_datetime(x, dayfirst=False).date()
        else:
            return pd.to_datetime(x, dayfirst=True).date()
    except:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to date.
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns pd.NaT
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors
       
def _datetime(x, errors='coerce', na=pd.NaT):
    if isinstance(x, str):
        x = x.replace('_', '-')
    try:
        if re.match(r'\D*(1|2\d\d\d)', x):
            return pd.to_datetime(x, dayfirst=False)
        else:
            return pd.to_datetime(x, dayfirst=True)
    except:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to datetime.
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns pd.NaT
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors


def _na(x, errors='ignore', na=None):
    possible_nas = [
        'not available', 'na', 'n/a', 'n.a', 'n.a.', 'na.', 'n.a',
        'not a number', 'nan',
        'null', 'nil',
        'none',
        '',
        ]
    
    if pd.isna(x) or str(x).lower() in possible_nas:
        return na
    else:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to "{na}".
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns None
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return None
            case _:
                return errors

def _nk(x, errors='ignore', nk='unknown', na=None):
    possible_nks = [
        'unk', 'unknown', 'not known', 'not known.',
        'nk', 'n.k.', 'n.k', 'n/k',
        'not specified', 'not specified.',
        ]
    
    if str(x).lower() in possible_nks:
        return nk
    else:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to "{nk}".
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns None
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors

def _yn(x, errors='coerce', yes='yes', no='no', na=None):
    if str(x).lower() in ['y', 'yes', 'true', '1', '1.0', 'positive', 'pos']:
        return yes
    elif str(x).lower() in ['n', 'no', 'false', '0', '0.0', 'negative', 'neg']:
        return no
    else:
        match errors:
            case 'raise':
                raise ValueError(f"""could not convert "{x}" to "{yes}" or "{no}".
                    Error handling:
                    errors='raise': raises a ValueError
                    errors='ignore': returns the original value
                    errors='coerce': returns NaN
                    errors=<any other value>: returns <any other value>
                    """)
            case 'ignore':
                return x
            case 'coerce':
                return na
            case _:
                return errors




class qpDict(dict):
    """
    qp.dict().values_flat() will unpack nested iterables
    qp.dict().invert() will invert the key:value pairs to value:key pairs
    """

    def values_flat(self):
        values_flat = []
        for val in self.values():
            if isinstance(val, dict):
                values_flat.extend(val.values())
            elif isinstance(val, qpDict):
                values_flat.extend(val.values_flat())
            elif hasattr(val, '__iter__') and not isinstance(val, str):
                values_flat.extend(val)
            else:
                values_flat.append(val)
        return values_flat
    
    def invert(self):
        return qpDict({val:key for key,val in self.items()})



