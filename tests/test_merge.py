import datetime
import pytest
import pandas as pd
import qplib as qp
from qplib import merge, log



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
            '#1: 42 ;\n#2: 42 ;\n',
            17,
            '',
            ],
        '1_term': [
            '#1: headache ;\n#2: nausea ;\n',
            'headache',
            '',
            ],
        '1_start': [
            '#1: 2023-01-01 ;\n#2:  ;\n',
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
            '#1: 42 ;\n#2: 42 ;\n',
            17,
            '',
            ],
        'MH_term': [
            '#1: headache ;\n#2: nausea ;\n',
            'headache',
            '',
            ],
        'MH_start': [
            '#1: 2023-01-01 ;\n#2:  ;\n',
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
            '#1: headache ;\n#2: nausea ;\n',
            'headache',
            '',
            ],
        '1_start': [
            '#1: 2023-01-01 ;\n#2:  ;\n',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_line_start():
    df1, df2, df3 = get_dfs()
    result = merge(df1, df2, on='uid', duplicates=True, prefix=None, line_start='§', verbosity=3)
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_age': [
            '§1: 42 ;\n§2: 42 ;\n',
            17,
            '',
            ],
        '1_term': [
            '§1: headache ;\n§2: nausea ;\n',
            'headache',
            '',
            ],
        '1_start': [
            '§1: 2023-01-01 ;\n§2:  ;\n',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_line_stop():
    df1, df2, df3 = get_dfs()
    result = merge(df1, df2, on='uid', duplicates=True, prefix=None, line_stop='§', verbosity=3)
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_age': [
            '#1: 42§#2: 42§',
            17,
            '',
            ],
        '1_term': [
            '#1: headache§#2: nausea§',
            'headache',
            '',
            ],
        '1_start': [
            '#1: 2023-01-01§#2: §',
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
            '#1: headache ;\n#2: nausea ;\n',
            'headache',
            '',
            ],
        '1_start': [
            '#1: 2023-01-01 ;\n#2:  ;\n',
            datetime.date(2021, 12, 3),
            '',
            ],
        })
    expected2 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        '1_term': [
            '#1: headache ;\n#2: nausea ;\n',
            'headache',
            '',
            ],
        '1_start': [
            '#1: 2023-01-01 ;\n#2:  ;\n',
            datetime.date(2021, 12, 3),
            '',
            ],
        '2_term': [
            'mexalen',
            '#1: aspirin ;\n#2: ibuprofen ;\n',
            '',
            ],
        '2_dose': [
            '1mg',
            '#1:  ;\n#2: 3 ;\n',
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
            '#1: headache ;\n#2: nausea ;\n',
            '#1: headache ;\n#2: nausea ;\n',
            '',
            ],
        '1_start': [
            '#1: 2023-01-01 ;\n#2:  ;\n',
            '#1: 2023-01-01 ;\n#2:  ;\n',
            '',
            ],
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')
    check_message('WARNING: column "uid" is not unique in left dataframe')

    log(clear=True)
    df2.drop(columns=['uid'], inplace=True)
    with pytest.raises(KeyError):
        merge(df1, df2, on='uid', duplicates=False, prefix=None, verbosity=3)
    check_message('WARNING: column "uid" is not unique in left dataframe')
    check_message('ERROR: "uid" is not in right dataframe')

    log(clear=True)
    df1.drop(columns=['uid'], inplace=True)
    with pytest.raises(KeyError):
        merge(df1, df3, on='uid', duplicates=False, prefix=None, verbosity=3)
    check_message('ERROR: "uid" is not in left dataframe')

