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

    ('1', 'int'),
    ('1.0', 'float'),
    ('True', 'bool'),
    ('text', 'str'),
    ('20240411', 'int'),

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

    ('20240411 00:00:001', 'str'),
    ('2024-04-11 00:00:0', 'datetime'),
    ('2024-04-11 00:00:00', 'datetime'),
    ('2024-04-11 00:00:001', 'datetime'),
    ('2024-04-11 00:00:001.001', 'datetime'),
    ('2024-04-11 00:00:001.001.001', 'datetime'),
    ('11-04-2024 00:00:001', 'datetime'),
    ('11-04-2024 00:00:001.001', 'datetime'),
    ('11-04-2024 00:00:001.001.001', 'datetime'),
    ('Apr-11-2024 00:00:001', 'datetime'),
    ('Apr-11-2024 00:00:001.001', 'datetime'),
    ('Apr-11-2024 00:00:001.001.001', 'datetime'),

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
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


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

    ('1', 1),
    ('1.0', 1.0),
    ('True', True),
    ('text', 'text'),
    ('20240411', 20240411),

    ('2024-04-11', datetime.datetime(2024, 4, 11).date()),
    ('2024.04.11', datetime.datetime(2024, 4, 11).date()),
    ('2024/04/11', datetime.datetime(2024, 4, 11).date()),
    ('2024\\04\\11', datetime.datetime(2024, 4, 11).date()),
    ('2024_04_11', datetime.datetime(2024, 4, 11).date()),

    ('11-04-2024', datetime.datetime(2024, 4, 11).date()),
    ('11.04.2024', datetime.datetime(2024, 4, 11).date()),
    ('11/04/2024', datetime.datetime(2024, 4, 11).date()),
    # ('11\\04\\2024', datetime.datetime(2024, 4, 11).date()), #wip: not yet recognized by qp.date()
    ('11_04_2024', datetime.datetime(2024, 4, 11).date()),

    ('Apr-11-2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr-11-2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr.11.2024', datetime.datetime(2024, 4, 11).date()),
    ('Apr/11/2024', datetime.datetime(2024, 4, 11).date()),
    # ('Apr\\11\\2024', datetime.datetime(2024, 4, 11).date()), #wip: not yet recognized by qp.date()
    ('Apr_11_2024', datetime.datetime(2024, 4, 11).date()),
    # ('Apr112024', datetime.datetime(2024, 4, 11).date()), #wip: not yet recognized by qp.date()
    ('11Apr2024', datetime.datetime(2024, 4, 11).date()),
    # ('2024Apr11', datetime.datetime(2024, 4, 11).date()), #wip: not yet recognized by qp.date()

    #wip: not yet recognized by qp.date()
    # ('November-11-2024', datetime.datetime(2024, 11, 1).date()),
    # ('November.11.2024', datetime.datetime(2024, 11, 1).date()),
    # ('November/11/2024', datetime.datetime(2024, 11, 1).date()),
    # ('November\\11\\2024', datetime.datetime(2024, 11, 1).date()),
    # ('November_11_2024', datetime.datetime(2024, 11, 1).date()),
    # ('November112024', datetime.datetime(2024, 11, 1).date()),
    # ('11November2024', datetime.datetime(2024, 11, 1).date()),
    # ('2024November11', datetime.datetime(2024, 11, 1).date()),

    #wip: implement:
    # ('20240411 00:00:001',),
    # ('2024-04-11 00:00:0', ),
    # ('2024-04-11 00:00:00', ),
    # ('2024-04-11 00:00:001', ),
    # ('2024-04-11 00:00:001.001', ),
    # ('2024-04-11 00:00:001.001.001', ),
    # ('11-04-2024 00:00:001', ),
    # ('11-04-2024 00:00:001.001', ),
    # ('11-04-2024 00:00:001.001.001', ),
    # ('Apr-11-2024 00:00:001', ),
    # ('Apr-11-2024 00:00:001.001', ),
    # ('Apr-11-2024 00:00:001.001.001', ),

    # ('2024-04-11 00:00:00', ),
    # ('2024.04.11 00:00:00', ),
    # ('2024/04/11 00:00:00', ),
    # ('2024\\04\\11 00:00:00', ),
    # ('2024_04_11 00:00:00', ),

    # ('11-04-2024 00:00:00', ),
    # ('11.04.2024 00:00:00', ),
    # ('11/04/2024 00:00:00', ),
    # ('11\\04\\2024 00:00:00', ),
    # ('11_04_2024 00:00:00', ),

    # ('Apr-11-2024 00:00:00', ),
    # ('Apr-11-2024 00:00:00', ),
    # ('Apr.11.2024 00:00:00', ),
    # ('Apr/11/2024 00:00:00', ),
    # ('Apr\\11\\2024 00:00:00', ),
    # ('Apr_11_2024 00:00:00', ),
    # ('Apr112024 00:00:00', ),
    # ('11Apr2024 00:00:00', ),
    # ('2024Apr11 00:00:00', ),

    # ('November-11-2024 00:00:00', ),
    # ('November.11.2024 00:00:00', ),
    # ('November/11/2024 00:00:00', ),
    # ('November\\11\\2024 00:00:00', ),
    # ('November_11_2024 00:00:00', ),
    # ('November112024 00:00:00', ),
    # ('11November2024 00:00:00', ),
    # ('2024November11 00:00:00', ),
    ])
def test_convert(input, expected):
    result = qp.convert(input)
    assert result == expected, f'\ninput: {input}\nRESULT: {result}\nEXPECTED: {expected}'


def test_dict_setattr():
    d = qp.dict()
    d['key'] = 'value'
    assert d['key'] == 'value'

    d.new_attr = 'new_value'
    assert d.new_attr == 'new_value'

    with pytest.raises(AttributeError):
        d.keys = 'should_fail'

def test_dict_values_flat():
    d = qp.dict({
        'a': 1,
        'b': [2, 3],
        'c': qp.dict({'d': 4, 'e': [5, 6]})
        })
    assert d.values_flat() == [1, 2, 3, 4, 5, 6]

def test_dict_invert():
    d = qp.dict({'a': 1, 'b': 2, 'c': 3})
    assert d.invert() == qp.dict({1: 'a', 2: 'b', 3: 'c'})

