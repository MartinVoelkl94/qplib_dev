import datetime
import pytest
import pandas as pd
import qplib as qp
from qplib import merge, log



start = qp.pandas.ROW_SIGNIFIER_START
stop = qp.pandas.ROW_SIGNIFIER_STOP



def check_message(expected_message):
    logs = qp.log()
    logs['text_full'] = logs['level'] + ': ' + logs['text']
    log_texts = logs['text_full'].to_list()
    assert expected_message in logs['text_full'].values, f'did not find expected message: {expected_message}\nin logs:\n{log_texts}'


def get_dfs():
    df1 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        })

    df2 = pd.DataFrame({
        'uid': [1, 1, 2],
        'age': [42, 42, 17],
        'term': ['headache', 'nausea', 'headache'],
        'start': [datetime.date(2023, 1, 1), None, datetime.date(2021, 12, 3)],
        })
    
    df3 = pd.DataFrame({
        'uid': [1, 2, 2],
        'age': [42, 17, 17],
        'term': ['mexalen', 'aspirin', 'ibuprofen'],
        'dose': ['1mg', None, 3],
        })

    return df1, df2, df3


def test_default():
    df1, df2, df3 = get_dfs()
    result = merge(df1, df2, on='uid', duplicates=True, prefix=None, verbosity=3)
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_age': [
            f'{start}1: 42{stop}{start}2: 42{stop}',
            17,
            '',
            ],
        '1_term': [
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            'headache',
            '',
            ],
        '1_start': [
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_prefix():
    df1, df2, df3 = get_dfs()
    result = merge(df1, df2, on='uid', duplicates=True, prefix='MH_', verbosity=3)
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH_age': [
            f'{start}1: 42{stop}{start}2: 42{stop}',
            17,
            '',
            ],
        'MH_term': [
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            'headache',
            '',
            ],
        'MH_start': [
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_duplicates():
    df1, df2, df3 = get_dfs()
    result = merge(df1, df2, on='uid', duplicates=False, prefix=None, verbosity=3)
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_term': [
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            'headache',
            '',
            ],
        '1_start': [
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_sequential():
    df1, df2, df3 = get_dfs()
    result1 = merge(df1, df2, on='uid', duplicates=False, prefix=None, verbosity=3)
    result2 = merge(result1, df3, on='uid', duplicates=False, prefix=None, verbosity=3)
    expected1 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_term': [
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            'headache',
            '',
            ],
        '1_start': [
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    expected2 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_term': [
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            'headache',
            '',
            ],
        '1_start': [
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            datetime.date(2021, 12, 3),
            '',
            ],
        '2_term': [
            'mexalen',
            f'{start}1: aspirin{stop}{start}2: ibuprofen{stop}',
            '',
            ],
        '2_dose': [
            '1mg',
            f'{start}1: {stop}{start}2: 3{stop}',
            '',
            ],
        })
    assert result1.equals(expected1), qp.diff(result1, expected1, output='str')
    assert result2.equals(expected2), qp.diff(result2, expected2, output='str')


def test_logging():
    log(clear=True)
    df1, df2, df3 = get_dfs()
    df1['uid'] = [1,1,3]
    result = merge(df1, df2, on='uid', duplicates=False, prefix=None, verbosity=3)
    expected = pd.DataFrame({
        'uid': [1, 1, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_term': [
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            f'{start}1: headache{stop}{start}2: nausea{stop}',
            '',
            ],
        '1_start': [
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            f'{start}1: 2023-01-01{stop}{start}2: {stop}',
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')
    check_message('ERROR: column "uid" is not unique in left dataframe')

    log(clear=True)
    df2.drop(columns=['uid'], inplace=True)
    with pytest.raises(KeyError):
        merge(df1, df2, on='uid', duplicates=False, prefix=None, verbosity=3)
    check_message('ERROR: column "uid" is not unique in left dataframe')
    check_message('ERROR: "uid" is not in right dataframe')

    log(clear=True)
    df1.drop(columns=['uid'], inplace=True)
    with pytest.raises(KeyError):
        merge(df1, df3, on='uid', duplicates=False, prefix=None, verbosity=3)
    check_message('ERROR: "uid" is not in left dataframe')
