import datetime
import numpy as np
import pandas as pd
import qplib as qp
from qplib import days_between



def check_message(expected_message):
    logs = qp.log()
    logs['text_full'] = logs['level'] + ': ' + logs['text']
    log_texts = logs['text_full'].to_list()
    text = f'did not find expected message: {expected_message}\nin logs:\n{log_texts}'
    assert expected_message in logs['text_full'].values, text


def get_df():
    df = pd.DataFrame({
        'date0': ['2023-01-01', '2023-09-07', '2024-12-03'],
        'date1': ['2024-01-01', '2024-02-01', None],
        'date2': ['2024-01-05', '2024-02-10', '2024-03-15'],
        'date3': ['nan', 'na', np.nan],
        'date4': [pd.NaT, pd.NA, pd.Timestamp('2024-04-01')],
        'date5': [
            datetime.datetime(2024, 5, 1),
            pd.Timestamp('2024-06-01'),
            datetime.datetime(2024, 7, 1),
            ],
        })
    return df




def test_all():
    result = days_between(
        get_df(),
        cols=get_df().columns,
        reference_col='date1',
        ).iloc[:, 6:].astype(object)
    expected = pd.DataFrame({
        'reference_date': [qp.date('2024-01-01'), qp.date('2024-02-01'), pd.NaT],
        'days_between_date1_and_date0': [-365, -147, np.nan],
        'days_between_date1_and_date1': [0, 0, np.nan],
        'days_between_date1_and_date2': [4, 9, np.nan],
        'days_between_date1_and_date3': [np.nan, np.nan, np.nan],
        'days_between_date1_and_date4': [np.nan, np.nan, np.nan],
        'days_between_date1_and_date5': [121, 121, np.nan],
        }).astype(object)
    assert result.equals(expected), qp.diff(result, expected)


def test_col():
    result = days_between(
        get_df(),
        cols='date0',
        reference_col='date1',
        ).iloc[:, 6:].astype(object)
    expected = pd.DataFrame({
        'reference_date': [qp.date('2024-01-01'), qp.date('2024-02-01'), pd.NaT],
        'days_between_date1_and_date0': [-365, -147, np.nan],
        }).astype(object)
    assert result.equals(expected), qp.diff(result, expected)


def test_cols():
    result = days_between(
        get_df(),
        cols=['date0', 'date2'],
        reference_col='date1',
        ).iloc[:, 6:].astype(object)
    expected = pd.DataFrame({
        'reference_date': [qp.date('2024-01-01'), qp.date('2024-02-01'), pd.NaT],
        'days_between_date1_and_date0': [-365, -147, np.nan],
        'days_between_date1_and_date2': [4, 9, np.nan],
        }).astype(object)
    assert result.equals(expected), qp.diff(result, expected)


def test_date():
    result = days_between(
        get_df(),
        cols=get_df().columns,
        reference_date='2024-01-01',
        ).iloc[:, 6:].astype(object)
    expected = pd.DataFrame({
        'reference_date': [
            qp.date('2024-01-01'),
            qp.date('2024-01-01'),
            qp.date('2024-01-01'),
            ],
        'days_between_2024-01-01_and_date0': [-365, -116, 337],
        'days_between_2024-01-01_and_date1': [0, 31, np.nan],
        'days_between_2024-01-01_and_date2': [4, 40, 74],
        'days_between_2024-01-01_and_date3': [np.nan, np.nan, np.nan],
        'days_between_2024-01-01_and_date4': [np.nan, np.nan, 91],
        'days_between_2024-01-01_and_date5': [121, 152, 182],
        }).astype(object)
    assert result.equals(expected), qp.diff(result, expected)


def test_logging():
    result = days_between(
        get_df(),
        cols=get_df().columns,
        )
    expected = get_df()
    assert result.equals(expected), qp.diff(result, expected)
    check_message('ERROR: no reference date or column provided')

    result = days_between(
        get_df(),
        cols=get_df().columns,
        reference_col='date1',
        reference_date='date1',
        )
    expected = get_df()
    assert result.equals(expected), qp.diff(result, expected)
    check_message('ERROR: both reference date and column provided')

    result = days_between(
        get_df(),
        cols='date6',
        reference_col='date1',
        )
    expected = get_df()
    expected['reference_date'] = [qp.date('2024-01-01'), qp.date('2024-02-01'), pd.NaT]
    assert result.equals(expected), qp.diff(result, expected)
    check_message('ERROR: both reference date and column provided')

    df = get_df()
    df['reference_date'] = [0, 0, 0]
    df['days_between_date1_and_date0'] = [0, 0, 0]
    result = days_between(
        df.copy(),
        cols='date0',
        reference_col='date1',
        ).iloc[:, 6:].astype(object)
    expected = pd.DataFrame({
        'reference_date': [qp.date('2024-01-01'), qp.date('2024-02-01'), pd.NaT],
        'days_between_date1_and_date0': [-365, -147, np.nan],
        }).astype(object)
    assert result.equals(expected), qp.diff(result, expected)
    check_message('WARNING: column "reference_col" already exists, overwriting')
    check_message(
        'WARNING: column "days_between_date1_and_date0" already exists, overwriting'
        )
