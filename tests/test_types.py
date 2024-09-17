import pytest
import datetime
import pandas as pd
import numpy as np
import qplib as qp

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

def test_error_raising():
    for func in qp_types:
        with pytest.raises(ValueError):
            func('abc', errors='raise')

def test_error_ignoring():
    for func in qp_types:
        assert func('abc', errors='ignore') == 'abc'

def test_error_coercing():
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
        assert func('abc', errors='custom') == 'custom'





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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'




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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'




@pytest.mark.parametrize("input, expected", [
    ('2020-01-01', (2020, 1, 1)),
    ('2020-01-01 00:00:00', (2020, 1, 1)),

    ('2020.01.01', (2020, 1, 1)),
    ('2020/01/01', (2020, 1, 1)),
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
    ])

def test_date(input, expected):
    result = qp.date(input)
    expected = datetime.date(*expected)
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


@pytest.mark.parametrize("input, expected", [
    ('2020-01-01', (2020, 1, 1)),
    ('2020-01-01 00:00:00', (2020, 1, 1)),
    ('2020-01-01 00:00:01', (2020, 1, 1, 0, 0, 1)),
    ('2020-01-01 00:01:00', (2020, 1, 1, 0, 1, 0)),
    ('2020-01-01 01:00:00', (2020, 1, 1, 1, 0, 0)),
    ('2020-01-01 01:01:01', (2020, 1, 1, 1, 1, 1)),

    ('2020.01.01', (2020, 1, 1)),
    ('2020/01/01', (2020, 1, 1)),
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

    ('02-01-20', (2020, 1, 2)),
    ('02.01.20', (2020, 1, 2)),
    ('02/01/20', (2020, 1, 2)),
    ('02 01 20', (2020, 1, 2)),

    ('2020-01-02', (2020, 1, 2)),
    ('2020.01.02', (2020, 1, 2)),
    ('2020/01/02', (2020, 1, 2)),
    ('2020 01 02', (2020, 1, 2)),
    ])

def test_datetime(input, expected):
    result = qp.datetime(input)
    expected = datetime.datetime(*expected)
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'



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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'




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
    (np.bool(True), 'bool'),
    (np.bool_(True), 'bool'),


    ('text', 'str'),
    ('1', 'int'),
    ('1.0', 'float'),
    ('True', 'bool'),
    ('2024-04-11', 'date'),
    ('2024.04.11', 'date'),
    ('2024/04/11', 'date'),
    ('2024\\04\\11', 'date'),
    ('2024-04-11 00:00:001', 'datetime'),
    ('2024-04-11 00:00:001.001', 'datetime'),
    ('2024-04-11 00:00:001.001.001', 'datetime'),
    ])
def test_type(input, expected):
    result = qp.type(input)
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


