import pandas as pd
import qplib as qp
from qplib import embed



def check_message(expected_message):
    logs = qp.log()
    logs['text_full'] = logs['level'] + ': ' + logs['text']
    log_texts = logs['text_full'].to_list()
    text = f'did not find expected message: {expected_message}\nin logs:\n{log_texts}'
    assert expected_message in logs['text_full'].values, text


def get_dfs():
    df1 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [1, 2, 3],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    df2 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'term': ['headache', 'nausea', 'headache'],
        })

    df3 = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 42],
        'term': ['mexalen', 'aspirin', 'ibuprofen'],
        'dose': ['1mg', None, 3],
        })

    return df1, df2, df3



def test_default1():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        )
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            '2\nuid: 2 ;\nage: 17 ;\nterm: nausea ;',
            '3\nuid: 3 ;\nage: 55 ;\nterm: headache ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')



def test_default2():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MED1',
        df_src=df3,
        ).fillna('')

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [1, 2, 3],
        'MED1': [
            '1\nuid: 1 ;\nage: 42 ;\nterm: mexalen ;\ndose: 1mg ;',
            '2\nuid: 2 ;\nage: 17 ;\nterm: aspirin ;\ndose: None ;',
            '',
            ],
        'MED2': [3, None, None],
        }).fillna('')

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_default3():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MED2',
        df_src=df3,
        ).fillna('')

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [1, 2, 3],
        'MED1': [1, 2, None],
        'MED2': [
            '3\nuid: 3 ;\nage: 42 ;\nterm: ibuprofen ;\ndose: 3 ;',
            None,
            None,
            ],
        }).fillna('')

    assert result.equals(expected), qp.diff(result, expected, output='str')



def test_multiples():
    df1, df2, df3 = get_dfs()

    df1['MH'] = [1, 1, 1]
    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        )
    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_key_src():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        key_src='uid',
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            '2\nuid: 2 ;\nage: 17 ;\nterm: nausea ;',
            '3\nuid: 3 ;\nage: 55 ;\nterm: headache ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_key_src1():
    df1, df2, df3 = get_dfs()
    df1['MH'] = [3, 2, 1]
    df3['new_key'] = [1, 2, 3]

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        key_src='uid',
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '3\nuid: 3 ;\nage: 55 ;\nterm: headache ;',
            '2\nuid: 2 ;\nage: 17 ;\nterm: nausea ;',
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')



def test_include():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        include=['uid', 'age', 'term'],
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;\nage: 42 ;\nterm: headache ;',
            '2\nuid: 2 ;\nage: 17 ;\nterm: nausea ;',
            '3\nuid: 3 ;\nage: 55 ;\nterm: headache ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_include1():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        include=['uid', 'age'],
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;\nage: 42 ;',
            '2\nuid: 2 ;\nage: 17 ;',
            '3\nuid: 3 ;\nage: 55 ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_exclude():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        exclude=['term'],
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;\nage: 42 ;',
            '2\nuid: 2 ;\nage: 17 ;',
            '3\nuid: 3 ;\nage: 55 ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_exclude1():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        exclude=['uid', 'age', 'term'],
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1',
            '2',
            '3',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_include_exclude():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        include=['uid', 'age', 'term'],
        exclude=['uid', 'age', 'term'],
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1',
            '2',
            '3',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_include_exclude1():
    df1, df2, df3 = get_dfs()

    result = embed(
        df_dest=df1,
        key_dest='MH',
        df_src=df2,
        include=['uid', 'age', 'term'],
        exclude=['age', 'term'],
        )

    expected = pd.DataFrame({
        'uid': [1, 2, 3],
        'age': [42, 17, 55],
        'IC': ['y', 'n', 'y'],
        'MH': [
            '1\nuid: 1 ;',
            '2\nuid: 2 ;',
            '3\nuid: 3 ;',
            ],
        'MED1': [1, 2, None],
        'MED2': [3, None, None],
        })

    assert result.equals(expected), qp.diff(result, expected, output='str')
