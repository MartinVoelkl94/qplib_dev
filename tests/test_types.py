import pytest
import datetime
import pandas as pd
import numpy as np
import qplib as qp
from faker import Faker



qp_types = [
    qp.int,
    qp.float,
    qp.num,
    qp.bool,
    qp.date,
    qp.datetime,
    qp.na,
    qp.nk,
    qp.yn,
    ]

invalid_values = [
    None,
    np.nan,
    pd.NaT,
    pd.NA,
    '',
    ' ',
    'nan',
    'NaN',
    'NAN',
    'na',
    'NA',
    'n/a',
    'N/A',
    'none',
    'None',
    'NONE',
    'null',
    'Null',
    'NULL',
    'nil',
    'Nil',
    'NIL',
    ]

def test_error_raising():
    for func in qp_types:
        with pytest.raises(ValueError):
            func('abc', errors='raise')

def test_error_ignoring():
    for func in qp_types:
        assert func('abc', errors='ignore') == 'abc'

def test_error_coercing():
    for invalid in invalid_values:
        assert qp.int('abc', errors='coerce') is np.nan
        assert qp.float('abc', errors='coerce') is np.nan
        assert qp.num('abc', errors='coerce') is np.nan
        assert qp.date('abc', errors='coerce') is pd.NaT
        assert qp.datetime('abc', errors='coerce') is pd.NaT
        assert qp.bool('abc', errors='coerce') is None
        assert qp.na('abc', errors='coerce') is None
        assert qp.nk('abc', errors='coerce') is None
        assert qp.yn('abc', errors='coerce') is None

def test_error_coercing_none():
    for func in qp_types:
        assert func('abc', errors='coerce', na=None) is None

def test_error_custom():
    for func in qp_types:
        text = f'Failed using custom error message for {func.__name__}'
        assert func('abc', errors='custom') == 'custom', text





@pytest.mark.parametrize("input, expected", [
    ('1', 1),
    ('1.0', 1),
    ('1.1', 1),
    ('1.9', 2),
    ('1.5', 2),

    ('0', 0),
    ('0.0', 0),
    ('0.1', 0),
    ('0.9', 1),
    ('0.5', 0),

    ('-1', -1),
    ('-1.0', -1),
    ('-1.1', -1),
    ('-1.9', -2),
    ('-1.5', -2),

    (1, 1),
    (1.0, 1),
    (1.1, 1),
    (1.9, 2),
    (1.5, 2),

    (0, 0),
    (0.0, 0),
    (0.1, 0),
    (0.9, 1),
    (0.5, 0),

    (-1, -1),
    (-1.0, -1),
    (-1.1, -1),
    (-1.9, -2),
    (-1.5, -2),

    ('1e0', 1),
    ('1_000', 1000),
    ])
def test_int(input, expected):
    result = qp.int(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text


@pytest.mark.parametrize("input, expected", [
    ('1', 1.0),
    ('1.0', 1.0),
    ('1.1', 1.1),

    ('0', 0.0),
    ('0.0', 0.0),
    ('0.1', 0.1),

    ('-1', -1.0),
    ('-1.0', -1.0),
    ('-1.1', -1.1),

    (1, 1.0),
    (1.0, 1.0),
    (1.1, 1.1),

    (0, 0.0),
    (0.0, 0.0),
    (0.1, 0.1),

    (-1, -1.0),
    (-1.0, -1.0),
    (-1.1, -1.1),

    ('1e0', 1.0),
    ('1_000', 1000.0),
    ])
def test_float(input, expected):
    result = qp.float(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text


@pytest.mark.parametrize("input, expected", [
    ('1', 1),
    ('1.0', 1.0),
    ('1.1', 1.1),

    ('0', 0),
    ('0.0', 0.0),
    ('0.1', 0.1),

    ('-1', -1),
    ('-1.0', -1.0),
    ('-1.1', -1.1),

    (1, 1),
    (1.0, 1.0),
    (1.1, 1.1),

    (0, 0),
    (0.0, 0.0),
    (0.1, 0.1),

    (-1, -1),
    (-1.0, -1.0),
    (-1.1, -1.1),

    ('1e0', 1),
    ])
def test_num(input, expected):
    result = qp.num(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text




@pytest.mark.parametrize("input, expected", [
    ('y', True),
    ('yes', True),
    ('true', True),
    ('1', True),
    ('1.0', True),
    ('positive', True),
    ('pos', True),

    ('n', False),
    ('no', False),
    ('false', False),
    ('0', False),
    ('0.0', False),
    ('negative', False),
    ('neg', False),

    ('Y', True),
    ('YES', True),
    ('TRUE', True),
    ('1', True),
    ('1.0', True),
    ('POSITIVE', True),
    ('POS', True),

    ('N', False),
    ('NO', False),
    ('FALSE', False),
    ('0', False),
    ('0.0', False),
    ('NEGATIVE', False),
    ('NEG', False),

    (0, False),
    (0.0, False),
    (1, True),
    (1.0, True),
    ])
def test_bool(input, expected):
    result = qp.bool(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text




@pytest.mark.parametrize("input, expected", [
    ('2020-01-01', (2020, 1, 1)),
    ('2020-01-01 00:00:00', (2020, 1, 1)),

    ('2020-01-01', (2020, 1, 1)),
    ('2020.01.01', (2020, 1, 1)),
    ('2020/01/01', (2020, 1, 1)),
    ('2020\\01\\01', (2020, 1, 1)),
    ('2020_01_01', (2020, 1, 1)),
    ('2020 01 01', (2020, 1, 1)),
    ('20200101', (2020, 1, 1)),

    ('2020 Jan 01', (2020, 1, 1)),
    ('2020 January 01', (2020, 1, 1)),
    ('2020 Jan 1', (2020, 1, 1)),
    ('2020 January 1', (2020, 1, 1)),

    ('Jan 01 2020', (2020, 1, 1)),
    ('January 01 2020', (2020, 1, 1)),
    ('Jan 1 2020', (2020, 1, 1)),
    ('January 1 2020', (2020, 1, 1)),

    ('01 Jan 2020', (2020, 1, 1)),
    ('01 January 2020', (2020, 1, 1)),
    ('1 Jan 2020', (2020, 1, 1)),
    ('1 January 2020', (2020, 1, 1)),

    ('01-01-2020', (2020, 1, 1)),
    ('01.01.2020', (2020, 1, 1)),
    ('01/01/2020', (2020, 1, 1)),
    ('01 01 2020', (2020, 1, 1)),

    ('02-01-20', (2020, 1, 2)),
    ('02.01.20', (2020, 1, 2)),
    ('02/01/20', (2020, 1, 2)),
    ('02 01 20', (2020, 1, 2)),

    ('2020-01-02', (2020, 1, 2)),
    ('2020.01.02', (2020, 1, 2)),
    ('2020/01/02', (2020, 1, 2)),
    ('2020 01 02', (2020, 1, 2)),

    ('2020-12-32', pd.NaT),
    ('2020.12.32', pd.NaT),
    ('2020/12/32', pd.NaT),
    ('2020\\12\\32', pd.NaT),
    ('2020_12_32', pd.NaT),
    ('2020 12 32', pd.NaT),
    ('20201232', pd.NaT),

    ('2020-13-30', pd.NaT),
    ('2020.13.30', pd.NaT),
    ('2020/13/30', pd.NaT),
    ('2020\\13\\30', pd.NaT),
    ('2020_13_30', pd.NaT),
    ('2020 13 30', pd.NaT),
    ('20201330', pd.NaT),

    ('3000-12-30', pd.NaT),
    ('3000.12.30', pd.NaT),
    ('3000/12/30', pd.NaT),
    ('3000\\12\\30', pd.NaT),
    ('3000_12_30', pd.NaT),
    ('3000 12 30', pd.NaT),
    ('30001230', pd.NaT),

    ])
def test_date(input, expected):
    result = qp.date(input)
    if expected is pd.NaT:
        text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
        assert result is expected, text
    else:
        expected = datetime.date(*expected)
        text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
        assert result == expected, text


def test_date_random():
    Faker.seed(1)
    fake = Faker()

    for i in range(1000):
        date_str = fake.date()
        result = qp.date(date_str)
        expected = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        assert result == expected, f'Expected {expected}, but got {result}'


@pytest.mark.parametrize("input, expected", [
    (0, pd.NaT),

    ('2020-02-01', (2020, 2, 1)),
    ('20200201', (2020, 2, 1)),
    ('2020-02-01 00:00:00', (2020, 2, 1)),
    ('2020-02-01 00:00:01', (2020, 2, 1, 0, 0, 1)),
    ('2020-02-01 00:01:00', (2020, 2, 1, 0, 1, 0)),
    ('2020-02-01 01:00:00', (2020, 2, 1, 1, 0, 0)),
    ('2020-02-01 01:01:01', (2020, 2, 1, 1, 1, 1)),
    ('20200201 01:01:01', (2020, 2, 1, 1, 1, 1)),

    ('2020.02.01', (2020, 2, 1)),
    ('2020/02/01', (2020, 2, 1)),
    ('2020 02 01', (2020, 2, 1)),

    ('2020 Feb 01', (2020, 2, 1)),
    ('2020 February 01', (2020, 2, 1)),
    ('2020 Feb 1', (2020, 2, 1)),
    ('2020 February 1', (2020, 2, 1)),


    ('Feb 01 2020', (2020, 2, 1)),
    ('February 01 2020', (2020, 2, 1)),
    ('Feb 1 2020', (2020, 2, 1)),
    ('February 1 2020', (2020, 2, 1)),

    ('01 Feb 2020', (2020, 2, 1)),
    ('01 February 2020', (2020, 2, 1)),
    ('1 Feb 2020', (2020, 2, 1)),
    ('1 February 2020', (2020, 2, 1)),

    ('01-02-2020', (2020, 2, 1)),
    ('01.02.2020', (2020, 2, 1)),
    ('01/02/2020', (2020, 2, 1)),

    ('01-02-20', (2020, 2, 1)),
    ('01.02.20', (2020, 2, 1)),
    ('01/02/20', (2020, 2, 1)),
    ('01 02 20', (2020, 2, 1)),

    ('2020-02-01', (2020, 2, 1)),
    ('2020.02.01', (2020, 2, 1)),
    ('2020/02/01', (2020, 2, 1)),
    ('2020 02 01', (2020, 2, 1)),
    ])
def test_datetime(input, expected):
    result = qp.datetime(input)
    if expected is pd.NaT:
        text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
        assert result is expected, text
    else:
        expected = datetime.datetime(*expected)
        text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
        assert result == expected, text


def test_datetime_random():
    Faker.seed(1)
    fake = Faker()

    for i in range(1000):
        date_str = fake.date_time().strftime('%Y-%m-%d %H:%M:%S')
        result = qp.datetime(date_str)
        expected = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        assert result == expected, f'Expected {expected}, but got {result}'



@pytest.mark.parametrize("input, expected", [
    (None, None),
    (np.nan, None),
    (pd.NaT, None),
    (pd.NA, None),
    ('', None),
    (' ', None),
    ('nan', None),
    ('NaN', None),
    ('NAN', None),
    ('na', None),
    ('NA', None),
    ('n/a', None),
    ('N/A', None),
    ('none', None),
    ('None', None),
    ('NONE', None),
    ('null', None),
    ('Null', None),
    ('NULL', None),
    ('nil', None),
    ('Nil', None),
    ('NIL', None),
    ('missing', None),
    ('Missing', None),
    ('MISSING', None),
    ('not available', None),
    ('Not available', None),
    ('NOT AVAILABLE', None),
    ('not a number', None),
    ('Not a number', None),
    ('NOT A NUMBER', None),
    ('not applicable', None),
    ('Not applicable', None),
    ('NOT APPLICABLE', None),
    ('not applicable', None),
    ('Not applicable', None),
    ('NOT APPLICABLE', None),
    ('not applicable', None),
    ('Not applicable', None),
    ('NOT APPLICABLE', None),
    ('void', None),
    ('Void', None),
    ('VOID', None),
    ('empty', None),
    ('Empty', None),
    ('EMPTY', None),
    ('blank', None),
    ('Blank', None),
    ('BLANK', None),
    ])
def test_na(input, expected):
    result = qp.na(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text


@pytest.mark.parametrize("input, expected", [
    ('unk', 'unknown'),
    ('unknown', 'unknown'),
    ('not known', 'unknown'),
    ('not known.', 'unknown'),
    ('nk', 'unknown'),
    ('n.k.', 'unknown'),
    ('n.k', 'unknown'),
    ('n/k', 'unknown'),
    ('not specified', 'unknown'),
    ('not specified.', 'unknown'),
    ('not specified', 'unknown'),
    ('not specified.', 'unknown'),
    ])
def test_nk(input, expected):
    result = qp.nk(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text


@pytest.mark.parametrize("input, expected", [
    ('y', 'yes'),
    ('yes', 'yes'),
    ('true', 'yes'),
    ('1', 'yes'),
    ('1.0', 'yes'),
    ('positive', 'yes'),
    ('pos', 'yes'),

    ('n', 'no'),
    ('no', 'no'),
    ('false', 'no'),
    ('0', 'no'),
    ('0.0', 'no'),
    ('negative', 'no'),
    ('neg', 'no'),

    ('Y', 'yes'),
    ('YES', 'yes'),
    ('TRUE', 'yes'),
    ('1', 'yes'),
    ('1.0', 'yes'),
    ('POSITIVE', 'yes'),
    ('POS', 'yes'),

    ('N', 'no'),
    ('NO', 'no'),
    ('FALSE', 'no'),
    ('0', 'no'),
    ('0.0', 'no'),
    ('NEGATIVE', 'no'),
    ('NEG', 'no'),

    (0, 'no'),
    (0.0, 'no'),
    (1, 'yes'),
    (1.0, 'yes'),
    ])
def test_yn(input, expected):
    result = qp.yn(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text




@pytest.mark.parametrize("input, expected", [
    (1, 'int'),
    (np.int8(1), 'int'),
    (np.int16(1), 'int'),
    (np.int32(1), 'int'),
    (np.int64(1), 'int'),

    (1.0, 'float'),
    (np.float16(1.0), 'float'),
    (np.float32(1.0), 'float'),
    (np.float64(1.0), 'float'),

    (True, 'bool'),
    (False, 'bool'),

    ('1', 'int'),
    ('1.0', 'float'),
    ('True', 'bool'),
    ('text', 'str'),
    ('20240411', 'int'),

    (complex(1, 1), 'complex'),
    (object(), 'object'),
    (None, 'NoneType'),
    ('123abc', 'str'),
    ('2024-13-32', 'str'),
    ('TrueFalse', 'str'),

    ('2024-04-11', 'date'),
    ('2024.04.11', 'date'),
    ('2024/04/11', 'date'),
    ('2024\\04\\11', 'date'),
    ('2024_04_11', 'date'),

    ('11-04-2024', 'date'),
    ('11.04.2024', 'date'),
    ('11/04/2024', 'date'),
    ('11\\04\\2024', 'date'),
    ('11_04_2024', 'date'),

    ('Apr-11-2024', 'date'),
    ('Apr-11-2024', 'date'),
    ('Apr.11.2024', 'date'),
    ('Apr/11/2024', 'date'),
    ('Apr\\11\\2024', 'date'),
    ('Apr_11_2024', 'date'),
    ('Apr112024', 'date'),
    ('11Apr2024', 'date'),
    ('2024Apr11', 'date'),

    ('November-11-2024', 'date'),
    ('November.11.2024', 'date'),
    ('November/11/2024', 'date'),
    ('November\\11\\2024', 'date'),
    ('November_11_2024', 'date'),
    ('November112024', 'date'),
    ('11November2024', 'date'),
    ('2024November11', 'date'),

    ('20240411 00:00:00', 'datetime'),
    ('2024-04-11 00:00:00', 'datetime'),
    ('2024-04-11 00:00:00', 'datetime'),
    ('11-04-2024 00:00:00', 'datetime'),
    ('Apr-11-2024 00:00:00', 'datetime'),

    ('2024-04-11 00:00:00', 'datetime'),
    ('2024.04.11 00:00:00', 'datetime'),
    ('2024/04/11 00:00:00', 'datetime'),
    ('2024\\04\\11 00:00:00', 'datetime'),
    ('2024_04_11 00:00:00', 'datetime'),

    ('11-04-2024 00:00:00', 'datetime'),
    ('11.04.2024 00:00:00', 'datetime'),
    ('11/04/2024 00:00:00', 'datetime'),
    ('11\\04\\2024 00:00:00', 'datetime'),
    ('11_04_2024 00:00:00', 'datetime'),

    ('Apr-11-2024 00:00:00', 'datetime'),
    ('Apr-11-2024 00:00:00', 'datetime'),
    ('Apr.11.2024 00:00:00', 'datetime'),
    ('Apr/11/2024 00:00:00', 'datetime'),
    ('Apr\\11\\2024 00:00:00', 'datetime'),
    ('Apr_11_2024 00:00:00', 'datetime'),
    ('Apr112024 00:00:00', 'datetime'),
    ('11Apr2024 00:00:00', 'datetime'),
    ('2024Apr11 00:00:00', 'datetime'),

    ('November-11-2024 00:00:00', 'datetime'),
    ('November.11.2024 00:00:00', 'datetime'),
    ('November/11/2024 00:00:00', 'datetime'),
    ('November\\11\\2024 00:00:00', 'datetime'),
    ('November_11_2024 00:00:00', 'datetime'),
    ('November112024 00:00:00', 'datetime'),
    ('11November2024 00:00:00', 'datetime'),
    ('2024November11 00:00:00', 'datetime'),
    ])
def test_type(input, expected):
    result = qp.type(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text




@pytest.mark.parametrize("input, expected", [
    (1, 1),
    (np.int8(1), 1),
    (np.int16(1), 1),
    (np.int32(1), 1),
    (np.int64(1), 1),

    (1.0, 1.0),
    (np.float16(1.0), 1.0),
    (np.float32(1.0), 1.0),
    (np.float64(1.0), 1.0),

    (True, True),
    (False, False),

    ('1', 1),
    ('1.0', 1.0),
    ('True', True),
    ('text', 'text'),
    ('20240411', 20240411),

    (complex(1, 1), complex(1, 1)),
    (None, None),
    ('123abc', '123abc'),
    ('2024-13-32', '2024-13-32'),
    ('TrueFalse', 'TrueFalse'),

    ('2024-04-11', datetime.datetime(2024, 4, 11).date()),
    ('2024.04.11', datetime.datetime(2024, 4, 11).date()),
    ('2024/04/11', datetime.datetime(2024, 4, 11).date()),
    ('2024\\04\\11', datetime.datetime(2024, 4, 11).date()),
    ('2024_04_11', datetime.datetime(2024, 4, 11).date()),

    ('11-04-2024', datetime.datetime(2024, 4, 11).date()),
    ('11.04.2024', datetime.datetime(2024, 4, 11).date()),
    ('11/04/2024', datetime.datetime(2024, 4, 11).date()),
    ('11\\04\\2024', datetime.datetime(2024, 4, 11).date()),
    ('11_04_2024', datetime.datetime(2024, 4, 11).date()),

    ('Apr-11-2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr-11-2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr.11.2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr/11/2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr\\11\\2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr_11_2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr112024', datetime.datetime(2024, 4, 11).date()),
    ('11Apr2024', datetime.datetime(2024, 4, 11).date()),
    ('2024Apr11', datetime.datetime(2024, 4, 11).date()),


    ('November-11-2024', datetime.datetime(2024, 11, 11).date()),
    ('November.11.2024', datetime.datetime(2024, 11, 11).date()),
    ('November/11/2024', datetime.datetime(2024, 11, 11).date()),
    ('November\\11\\2024', datetime.datetime(2024, 11, 11).date()),
    ('November_11_2024', datetime.datetime(2024, 11, 11).date()),
    ('November112024', datetime.datetime(2024, 11, 11).date()),
    ('11November2024', datetime.datetime(2024, 11, 11).date()),
    ('2024November11', datetime.datetime(2024, 11, 11).date()),

    ('20240411 00:00:00', datetime.datetime(2024, 4, 11)),
    ('2024-04-11 00:00:00', datetime.datetime(2024, 4, 11)),
    ('11-04-2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr-11-2024 00:00:00', datetime.datetime(2024, 4, 11)),

    ('2024-04-11 00:00:00', datetime.datetime(2024, 4, 11)),
    ('2024.04.11 00:00:00', datetime.datetime(2024, 4, 11)),
    ('2024/04/11 00:00:00', datetime.datetime(2024, 4, 11)),
    ('2024\\04\\11 00:00:00', datetime.datetime(2024, 4, 11)),
    ('2024_04_11 00:00:00', datetime.datetime(2024, 4, 11)),

    ('11-04-2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('11.04.2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('11/04/2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('11\\04\\2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('11_04_2024 00:00:00', datetime.datetime(2024, 4, 11)),

    ('Apr-11-2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr-11-2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr.11.2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr/11/2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr\\11\\2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr_11_2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('Apr112024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('11Apr2024 00:00:00', datetime.datetime(2024, 4, 11)),
    ('2024Apr11 00:00:00', datetime.datetime(2024, 4, 11)),
    ])
def test_convert(input, expected):
    result = qp.convert(input)
    text = f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'
    assert result == expected, text



def test_Container_set_get():
    c = qp.Container()
    c['name'] = 1
    assert c['name'] == 1
    assert c.name == 1

    c.name = 2
    assert c.name == 2
    assert c['name'] == 2

    c.__setitem__('name', 4)
    assert c.name == 4
    assert c['name'] == 4

    c.__setattr__('name', 3)
    assert c.name == 3
    assert c['name'] == 3


def test_Container_other_methods():
    c = qp.Container()
    c.a = 1
    c['b'] = 2
    c.__setitem__('c', 3)
    c.__setattr__('d', 4)
    c.append(5)
    c.append(6)

    assert c.i0 == 5
    assert c.i1 == 6
    assert c == eval(c.__repr__().replace('Container', 'qp.Container'))
    assert list(c.keys()) == ['a', 'b', 'c', 'd', 'i0', 'i1']
    assert list(c.values()) == [1, 2, 3, 4, 5, 6]
    assert list(c.items()) == [
        ('a', 1),
        ('b', 2),
        ('c', 3),
        ('d', 4),
        ('i0', 5),
        ('i1', 6),
        ]

    c.clear()
    assert c == qp.Container()



def test_dict_set_get():
    d = qp.dict()
    d['name'] = 1
    assert d['name'] == 1
    assert d.name == 1

    d.name = 2
    assert d.name == 2
    assert d['name'] == 2

    d.__setitem__('name', 4)
    assert d.name == 4
    assert d['name'] == 4

    d.__setattr__('name', 3)
    assert d.name == 3
    assert d['name'] == 3


def test_dict_keys_values_items():
    d = qp.dict()
    d.a = 1
    d['b'] = 2

    assert set(d.keys()) == {'a', 'b'}
    assert set(d.values()) == {1, 2}
    assert set(d.items()) == {('a', 1), ('b', 2)}


def test_dict_clear():
    d = qp.dict()

    d.a = 1
    d.clear()
    with pytest.raises(AttributeError):
        d.a

    d.a = 1
    d.clear()
    with pytest.raises(KeyError):
        d['a']

    d['a'] = 1
    d.clear()
    with pytest.raises(KeyError):
        d['a']

    d['a'] = 1
    d.clear()
    with pytest.raises(AttributeError):
        d.a


def test_dict_copy():
    d1 = qp.dict()
    d1.a = 1
    d1['b'] = 2

    d2 = d1.copy()
    assert d2.a == 1
    assert d2['a'] == 1
    assert d2.b == 2
    assert d2['b'] == 2

    d2.a = 3
    assert d1.a == 1
    assert d1['a'] == 1
    assert d2.a == 3
    assert d2['a'] == 3


def test_dict_update():
    d = qp.dict()
    d.a = 1
    d['b'] = 2

    d.update({'a': 3, 'c': 4})
    assert d.a == 3
    assert d['a'] == 3
    assert d.b == 2
    assert d['b'] == 2
    assert d.c == 4
    assert d['c'] == 4

    d.update(d=5, e=6)
    assert d.d == 5
    assert d['d'] == 5
    assert d.e == 6
    assert d['e'] == 6


def test_dict_nonstring():
    d = qp.dict()

    with pytest.raises(AttributeError):
        d[1] = 'one'

    with pytest.raises(AttributeError):
        d[1.0] = 'one'


def test_dict_reserved_attributes():
    reserved_attributes = dict().__dir__() + [
        '__dict__',
        '__weakref__',
        '_reserved_attributes',
        'values_flat',
        'invert',
        ]
    d = qp.dict()
    assert reserved_attributes == d._reserved_attributes
    for attr in reserved_attributes:
        with pytest.raises(AttributeError):
            d[attr] = 'should_fail'


def test_dict_values_flat():
    d = qp.dict({
        'a': 1,
        'b': [2, {'x': 3}],
        'c': (4, 5),
        'd': {6, 7},
        'e': {'y0': 8, 'y1': {'y2': 9}},
        'f': qp.dict({'z0': 10, 'z1': [11, 12]})
        })
    values_expected = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        ]
    assert d.values_flat() == values_expected, "Failed for nested structures"


def test_dict_invert():
    d = qp.dict({'a': '1', 'b': '2', 'c': '3'})
    assert d.invert() == qp.dict({'1': 'a', '2': 'b', '3': 'c'})

    d = qp.dict({'a': '1', 'b': '1', 'c': '3'})
    assert d.invert() == qp.dict({'1': 'b', '3': 'c'})
