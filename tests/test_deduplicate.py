import pandas as pd
import qplib as qp
from qplib import deduplicate



def check_message(expected_message):
    logs = qp.log()
    logs['text_full'] = logs['level'] + ': ' + logs['text']
    log_texts = logs['text_full'].to_list()
    text = f'did not find expected message: {expected_message}\nin logs:\n{log_texts}'
    assert expected_message in logs['text_full'].values, text


#list tests

def test_deduplicate_list():
    obj = [1]
    expected = ['1']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_list1():
    obj = [1, 1]
    expected = ['1', '1_1']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_list2():
    obj = [1, 2, 2,]
    expected = ['1', '2', '2_1']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_list3():
    obj = [1, 2, 2, 3, 3, 3]
    expected = ['1', '2', '2_1', '3', '3_1', '3_2']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_list4():
    obj = [3, 1, 3, 2, 3, 2]
    expected = ['3', '1', '3_1', '2', '3_2', '2_1']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_list5():
    obj = [1, 1, '1_1']
    expected = ['1', '1_1', '1_1_1']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_list6():
    obj = ['1_1', 1, '1_1_1', 1, '1_1']
    expected = ['1_1', '1', '1_1_1', '1_1_1_1', '1_1_1_1_1']
    result = deduplicate(obj, name='test_list', verbosity=3)
    assert result == expected, f'EXPECTED: {expected}\nRESULT: {result}'


#series tests


def test_deduplicate_series():
    obj = pd.Series([1])
    expected = pd.Series(['1'])
    result = deduplicate(obj, name='test_series', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series1():
    obj = pd.Series([1, 1])
    expected = pd.Series(['1', '1_1'])
    result = deduplicate(obj, name='test_series1', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series2():
    obj = pd.Series([1, 2, 2])
    expected = pd.Series(['1', '2', '2_1'])
    result = deduplicate(obj, name='test_series2', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series3():
    obj = pd.Series([1, 2, 2, 3, 3, 3])
    expected = pd.Series(['1', '2', '2_1', '3', '3_1', '3_2'])
    result = deduplicate(obj, name='test_series3', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series4():
    obj = pd.Series([3, 1, 3, 2, 3, 2])
    expected = pd.Series(['3', '1', '3_1', '2', '3_2', '2_1'])
    result = deduplicate(obj, name='test_series4', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series5():
    obj = pd.Series([1, 1, '1_1'])
    expected = pd.Series(['1', '1_1', '1_1_1'])
    result = deduplicate(obj, name='test_series5', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series6():
    obj = pd.Series(['1_1', 1, '1_1_1', 1, '1_1'])
    expected = pd.Series(['1_1', '1', '1_1_1', '1_1_1_1', '1_1_1_1_1'])
    result = deduplicate(obj, name='test_series6', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series_index():
    obj = pd.Series([1, 1, '1_1'], index=['a', 'b', 'c'])
    expected = pd.Series(['1', '1_1', '1_1_1'], index=['a', 'b', 'c'])
    result = deduplicate(obj, name='test_series_index', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'


def test_deduplicate_series_index1():
    obj = pd.Series([1, 1, '1_1'], index=['a', 'b', 'b'])
    expected = pd.Series(['1', '1_1', '1_1_1'], index=['a', 'b', 'b_1'])
    result = deduplicate(obj, name='test_series_index1', verbosity=3)
    assert result.equals(expected), f'EXPECTED: {expected}\nRESULT: {result}'
