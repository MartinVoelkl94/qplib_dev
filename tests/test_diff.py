import pandas as pd
import numpy as np
import qplib as qp
import openpyxl

def process_str(string):
    return string.replace('\n', '').replace('\t', '').replace(' ', '')


#prepare testing files

def setup(df_old, df_new, tmpdir):

    path_df_old = f'{tmpdir}/df_old.xlsx'
    df_old.to_excel(path_df_old, index=False)

    path_df_new = f'{tmpdir}/df_new.xlsx'
    df_new.to_excel(path_df_new, index=False)

    return path_df_old, path_df_new


def setup_csv(df_old, df_new, tmpdir):

    path_df_old = f'{tmpdir}/df_old.csv'
    df_old.to_csv(path_df_old, index=False)

    path_df_new = f'{tmpdir}/df_new.csv'
    df_new.to_csv(path_df_new, index=False)

    return path_df_old, path_df_new


def get_expected_new():
    expected = pd.DataFrame(
        columns=['diff', 'uid', 'd', 'b', 'a'],
        index=['y', 'x2', 'z']
        )

    expected['uid'] = expected.index

    expected.loc['y', 'diff'] = 'vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0

    expected.loc['x2', 'diff'] = 'row added'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1

    expected.loc['z', 'diff'] = 'vals added: 1\nvals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan
    return expected


def get_expected_newplus():
    expected = pd.DataFrame(
        columns=['diff', 'uid', 'd', 'b', 'b *old', 'a', 'a *old'],
        index=['y', 'x2', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'diff'] = 'vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'b *old'] = ''
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'a *old'] = 2

    expected.loc['x2', 'diff'] = 'row added'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'b *old'] = ''
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'a *old'] = ''

    expected.loc['z', 'diff'] = 'vals added: 1\nvals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'b *old'] = None
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'a *old'] = 3

    return expected


def get_expected_old():
    expected = pd.DataFrame(
        columns=['diff', 'uid', 'a', 'b', 'c'],
        index=['x', 'y', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['x', 'diff'] = 'row removed'
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'c'] = 1

    expected.loc['y', 'diff'] = 'vals changed: 1'
    expected.loc['y', 'a'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'c'] = 2

    expected.loc['z', 'diff'] = 'vals added: 1\nvals removed: 1'
    expected.loc['z', 'a'] = 3
    expected.loc['z', 'b'] = None
    expected.loc['z', 'c'] = 3

    return expected


def get_expected_mix():
    expected = pd.DataFrame(
        columns=['diff', 'uid', 'd', 'b', 'a', 'c'],
        index=['y', 'x2', 'z', 'x'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'diff'] = 'vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'c'] = 2

    expected.loc['x2', 'diff'] = 'row added'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'c'] = np.nan

    expected.loc['z', 'diff'] = 'vals added: 1\nvals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'c'] = 3

    expected.loc['x', 'diff'] = 'row removed'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1

    return expected


def test_mode_new(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)
    expected = get_expected_new()

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        verbosity=0,
        ).show('new', 'Sheet1').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501


def test_mode_newplus(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)
    expected = get_expected_newplus()

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501


def test_mode_old(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)
    expected = get_expected_old()

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_mix(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)
    expected = get_expected_mix()

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501







def test_summary(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ).summary().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == '2'
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ).summary().data
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == '2'
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == '2'


    #ignore a

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='a',
        ).summary().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == '1'
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='a',
        ).summary().data
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == '1'
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == '1'


    #ignore b

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['b'],
        ).summary().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == '1'
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['b'],
        ).summary().data
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == '1'
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == '1'


    #ignore a and b

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a', 'b'],
        ).summary().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == ''
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a', 'b'],
        ).summary().data
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == ''
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == ''


def test_details(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)


    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ).details().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == '2'
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''
    assert result.at['data', 'all cols added'] == "'d'"
    assert result.at['data', 'all cols removed'] == "'c'"
    assert result.at['data', 'all rows added'] == "'x2'"
    assert result.at['data', 'all rows removed'] == "'x'"
    assert result.at['data', 'all dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ).details().data
    dtypes_changed = "'a': 'int64' -> 'float64',\n'b': 'float64' -> 'int64'"
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == '2'
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == '2'
    assert result.at['Sheet1', 'all cols added'] == "'d'"
    assert result.at['Sheet1', 'all cols removed'] == "'c'"
    assert result.at['Sheet1', 'all rows added'] == "'x2'"
    assert result.at['Sheet1', 'all rows removed'] == "'x'"
    assert result.at['Sheet1', 'all dtypes changed'] == dtypes_changed


    #ignore a

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='a',
        ).details().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == '1'
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''
    assert result.at['data', 'all cols added'] == "'d'"
    assert result.at['data', 'all cols removed'] == "'c'"
    assert result.at['data', 'all rows added'] == "'x2'"
    assert result.at['data', 'all rows removed'] == "'x'"
    assert result.at['data', 'all dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='a',
        ).details().data
    dtypes_changed = "'b': 'float64' -> 'int64'"
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == '1'
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == '1'
    assert result.at['Sheet1', 'all cols added'] == "'d'"
    assert result.at['Sheet1', 'all cols removed'] == "'c'"
    assert result.at['Sheet1', 'all rows added'] == "'x2'"
    assert result.at['Sheet1', 'all rows removed'] == "'x'"
    assert result.at['Sheet1', 'all dtypes changed'] == dtypes_changed


    #ignore b

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['b'],
        ).details().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == '1'
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''
    assert result.at['data', 'all cols added'] == "'d'"
    assert result.at['data', 'all cols removed'] == "'c'"
    assert result.at['data', 'all rows added'] == "'x2'"
    assert result.at['data', 'all rows removed'] == "'x'"
    assert result.at['data', 'all dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['b'],
        ).details().data

    dtypes_changed = "'a': 'int64' -> 'float64'"

    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == '1'
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == '1'

    assert result.at['Sheet1', 'all cols added'] == "'d'"
    assert result.at['Sheet1', 'all cols removed'] == "'c'"
    assert result.at['Sheet1', 'all rows added'] == "'x2'"
    assert result.at['Sheet1', 'all rows removed'] == "'x'"
    assert result.at['Sheet1', 'all dtypes changed'] == dtypes_changed


    #ignore a and b

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a', 'b'],
        ).details().data
    assert result.at['data', 'uid'] == 'uid'
    assert result.at['data', 'in both datasets'] == 'yes'
    assert result.at['data', 'cols shared'] == ''
    assert result.at['data', 'rows shared'] == '2'
    assert result.at['data', 'cols added'] == '1'
    assert result.at['data', 'cols removed'] == '1'
    assert result.at['data', 'rows added'] == '1'
    assert result.at['data', 'rows removed'] == '1'
    assert result.at['data', 'dtypes changed'] == ''
    assert result.at['data', 'all cols added'] == "'d'"
    assert result.at['data', 'all cols removed'] == "'c'"
    assert result.at['data', 'all rows added'] == "'x2'"
    assert result.at['data', 'all rows removed'] == "'x'"
    assert result.at['data', 'all dtypes changed'] == ''

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a', 'b'],
        ).details().data
    dtypes_changed = ''
    assert result.at['Sheet1', 'uid'] == 'uid'
    assert result.at['Sheet1', 'cols shared'] == ''
    assert result.at['Sheet1', 'rows shared'] == '2'
    assert result.at['Sheet1', 'cols added'] == '1'
    assert result.at['Sheet1', 'cols removed'] == '1'
    assert result.at['Sheet1', 'rows added'] == '1'
    assert result.at['Sheet1', 'rows removed'] == '1'
    assert result.at['Sheet1', 'dtypes changed'] == ''
    assert result.at['Sheet1', 'all cols added'] == "'d'"
    assert result.at['Sheet1', 'all cols removed'] == "'c'"
    assert result.at['Sheet1', 'all rows added'] == "'x2'"
    assert result.at['Sheet1', 'all rows removed'] == "'x'"
    assert result.at['Sheet1', 'all dtypes changed'] == dtypes_changed




def test_str(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 2
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 0
          all cols shared:
            'a',
            'b'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ).str()
    expected = """
        Diffs summary:
          Diff of 'Sheet1':
            cols shared: 2
            cols added: 1
            cols removed: 1
            rows shared: 2
            rows added: 1
            rows removed: 1
            dtypes changed: 2
            all cols shared:
              'a',
              'b'
            all cols added:
              'd'
            all cols removed:
              'c'
            all rows shared:
              'y',
              'z'
            all rows added:
              'x2'
            all rows removed:
              'x'
            all dtypes changed:
              'a': 'int64' -> 'float64',
              'b': 'float64' -> 'int64'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 2
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 2
          all cols shared:
            'a',
            'b'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
            'a': 'object' -> 'float64',
            'b': 'object' -> 'int64'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 2
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 2
          all cols shared:
            'a',
            'b'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
            'a': 'int64' -> 'object',
            'b': 'float64' -> 'object'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #ignore a, in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='a',
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 1
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 0
          all cols shared:
            'b'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore a, files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='a',
        ).str()
    expected = """
        Diffs summary:
          Diff of 'Sheet1':
            cols shared: 1
            cols added: 1
            cols removed: 1
            rows shared: 2
            rows added: 1
            rows removed: 1
            dtypes changed: 1
            all cols shared:
              'b'
            all cols added:
              'd'
            all cols removed:
              'c'
            all rows shared:
              'y',
              'z'
            all rows added:
              'x2'
            all rows removed:
              'x'
            all dtypes changed:
              'b': 'float64' -> 'int64'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore a, in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols='a',
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 1
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 1
          all cols shared:
            'b'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
            'b': 'object' -> 'int64'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore a, file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols='a',
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 1
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 1
          all cols shared:
            'b'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
            'b': 'float64' -> 'object'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



    #ignore b, in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 1
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 0
          all cols shared:
            'a'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore b, files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'Sheet1':
            cols shared: 1
            cols added: 1
            cols removed: 1
            rows shared: 2
            rows added: 1
            rows removed: 1
            dtypes changed: 1
            all cols shared:
              'a'
            all cols added:
              'd'
            all cols removed:
              'c'
            all rows shared:
              'y',
              'z'
            all rows added:
              'x2'
            all rows removed:
              'x'
            all dtypes changed:
              'a': 'int64' -> 'float64'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore b, in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 1
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 1
          all cols shared:
            'a'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
            'a': 'object' -> 'float64'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #ignore b, file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 1
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 1
          all cols shared:
            'a'
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
            'a': 'int64' -> 'object'
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



    #ignore a and b, in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a', 'b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 0
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 0
          all cols shared:
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore a and b, files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a', 'b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'Sheet1':
            cols shared: 0
            cols added: 1
            cols removed: 1
            rows shared: 2
            rows added: 1
            rows removed: 1
            dtypes changed: 0
            all cols shared:
            all cols added:
              'd'
            all cols removed:
              'c'
            all rows shared:
              'y',
              'z'
            all rows added:
              'x2'
            all rows removed:
              'x'
            all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore a and b, in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['a', 'b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 0
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 0
          all cols shared:
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501

    #ignore a and b, file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['a', 'b'],
        ).str()
    expected = """
        Diffs summary:
          Diff of 'data':
          cols shared: 0
          cols added: 1
          cols removed: 1
          rows shared: 2
          rows added: 1
          rows removed: 1
          dtypes changed: 0
          all cols shared:
          all cols added:
            'd'
          all cols removed:
            'c'
          all rows shared:
            'y',
            'z'
          all rows added:
            'x2'
          all rows removed:
            'x'
          all dtypes changed:
        """
    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



def test_csv(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_csv, df_new_csv = setup_csv(df_old, df_new, tmpdir)

    result_df = qp.diff(
        df_old,
        df_new,
        ).show('new').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        ).show('new').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501

    result_df = qp.diff(
        df_old,
        df_new,
        ).show('new+').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        ).show('new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501

    result_df = qp.diff(
        df_old,
        df_new,
        ).show('new+').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        ).show('new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501

    result_df = qp.diff(
        df_old,
        df_new,
        ).show('new+').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        ).show('new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501


def test_excel(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_unique_cols = pd.DataFrame({
        'uid': [1],
        'a': [10],
        })
    df_new_unique_cols = pd.DataFrame({
        'uid': [1],
        'b': [10],
        })
    df_old_unique_rows = pd.DataFrame({
        'uid': [1],
        'a': [10],
        })
    df_new_unique_rows = pd.DataFrame({
        'uid': [2],
        'a': [20],
        })
    df_empty = pd.DataFrame()

    path_df_new = f'{tmpdir}/new.xlsx'
    with pd.ExcelWriter(path_df_new) as writer:
        df_new.to_excel(
            writer,
            sheet_name='df',
            index=False,
            index_label='uid',
            )
        df_new_unique_cols.to_excel(
            writer,
            sheet_name='unique_cols',
            index=False,
            index_label='uid',
            )
        df_new_unique_rows.to_excel(
            writer,
            sheet_name='unique_rows',
            index=False,
            index_label='uid',
            )
        df_empty.to_excel(
            writer,
            sheet_name='empty',
            index=False,
            index_label='uid',
            )
        df_empty.to_excel(
            writer,
            sheet_name='empty_new',
            index=False,
            index_label='uid',
            )

    path_df_old = f'{tmpdir}/old.xlsx'
    with pd.ExcelWriter(path_df_old) as writer:
        df_old.to_excel(
            writer,
            sheet_name='df',
            index=False,
            index_label='uid',
            )
        df_old_unique_cols.to_excel(
            writer,
            sheet_name='unique_cols',
            index=False,
            index_label='uid',
            )
        df_old_unique_rows.to_excel(
            writer,
            sheet_name='unique_rows',
            index=False,
            index_label='uid',
            )
        df_empty.to_excel(
            writer,
            sheet_name='empty',
            index=False,
            index_label='uid',
            )
        df_empty.to_excel(
            writer,
            sheet_name='empty_old',
            index=False,
            index_label='uid',
            )

    diff = qp.diff(
        path_df_old,
        path_df_new,
        uid='uid',
        verbosity=0,
        )
    diff.to_excel(f'{tmpdir}/diff.xlsx')

    s = pd.read_excel(
        f'{tmpdir}/diff.xlsx',
        sheet_name='summary',
        index_col=0,
        ).fillna('')

    col_val_mapping_df = {
        'in both datasets': 'yes',
        'cols shared': 2,
        'cols added': 1,
        'cols removed': 1,
        'rows shared': 2,
        'rows added': 1,
        'rows removed': 1,
        'dtypes changed': 2,
        }
    for col, val in col_val_mapping_df.items():
        expected = val
        result = s.loc['df', col]
        assert result == expected, f'For "df" and "{col}": RESULT: "{result}" EXPECTED: "{expected}"'  # noqa E501


    col_val_mapping_unique_cols = {
        'in both datasets': 'yes',
        'cols shared': '',
        'cols added': 1,
        'cols removed': 1,
        'rows shared': 1,
        'rows added': '',
        'rows removed': '',
        'dtypes changed': '',
        }
    for col, val in col_val_mapping_unique_cols.items():
        expected = val
        result = s.loc['unique_cols', col]
        assert result == expected, f'For "unique_cols" and "{col}": RESULT: "{result}" EXPECTED: "{expected}"'  # noqa E501


    col_val_mapping_unique_rows = {
        'in both datasets': 'yes',
        'cols shared': 1,
        'cols added': '',
        'cols removed': '',
        'rows shared': '',
        'rows added': 1,
        'rows removed': 1,
        'dtypes changed': '',
        }
    for col, val in col_val_mapping_unique_rows.items():
        expected = val
        result = s.loc['unique_rows', col]
        assert result == expected, f'For "unique_rows" and "{col}": RESULT: "{result}" EXPECTED: "{expected}"'  # noqa E501


    col_val_mapping_unique_cols = {
        'in both datasets': 'yes',
        'cols shared': '',
        'cols added': 1,
        'cols removed': 1,
        'rows shared': 1,
        'rows added': '',
        'rows removed': '',
        'dtypes changed': '',
        }
    for col, val in col_val_mapping_unique_cols.items():
        expected = val
        result = s.loc['unique_cols', col]
        assert result == expected, f'For "unique_cols" and "{col}": RESULT: "{result}" EXPECTED: "{expected}"'  # noqa E501


    assert s.loc['empty', 'in both datasets'] == 'yes'
    assert s.loc['empty_new', 'in both datasets'] == 'only in new'
    assert s.loc['empty_old', 'in both datasets'] == 'only in old'
    for col in s.columns:
        if col in ['uid', 'in both datasets']:
            continue
        assert s.loc['empty', col] == ''
        assert s.loc['empty_new', col] == ''
        assert s.loc['empty_old', col] == ''



def test_identical(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_xlsx, df_new_xlsx = setup(df_old, df_new, tmpdir)
    df_old_csv, df_new_csv = setup_csv(df_old, df_new, tmpdir)

    diff1 = qp.diff(
        df_new,
        df_new,
        )
    diff2 = qp.diff(
        df_new_xlsx,
        df_new_xlsx,
        )
    diff3 = qp.diff(
        df_new_csv,
        df_new_csv,
        )

    df1 = diff1.show('mix').data
    df1.index = df1['uid']
    df1['a'] = df1['a'].astype(float)
    df1 = df1.astype('object').fillna('')

    df2 = (
        diff2
        .show('mix')
        .data
        .astype('object')
        .fillna('')
        )

    df3 = (
        diff3
        .show('mix')
        .data
        .astype('object')
        .fillna('')
        )
    df3.index = df3['uid']

    assert df1.equals(df2), f"DataFrames are not equal:\n{df1}\n{df2}"
    assert df1.equals(df3), f"DataFrames are not equal:\n{df1}\n{df3}"

    str1 = diff1.str()
    str2 = diff2.str()
    str3 = diff3.str()

    expected = (
        'Diffs summary:\n'
        "  Diff of 'data':\n"
        '    datasets are identical\n  '
        )
    expected3 = (
        'Diffs summary:\n'
        "  Diff of 'Sheet1':\n"
        '    datasets are identical\n  '
        )

    assert str1 == str3 == expected, f"Strings are not equal:\n{str1}\n{str3}\n{expected}"  # noqa E501
    assert process_str(str2) == process_str(expected3), f"EXPECTED:\n{expected3}\nRESULT:\n{str2}"  # noqa E501

    for diff in [diff1, diff2, diff3]:
        summary = diff.summary().data
        summary.index = ['data']
        summary.index.name = 'dataset'

        assert summary.at['data', 'uid'] == 'uid'
        assert summary.at['data', 'in both datasets'] == 'yes'
        assert summary.at['data', 'cols shared'] == '3'
        assert summary.at['data', 'rows shared'] == '3'
        assert summary.at['data', 'cols added'] == ''
        assert summary.at['data', 'cols removed'] == ''
        assert summary.at['data', 'rows added'] == ''
        assert summary.at['data', 'rows removed'] == ''
        assert summary.at['data', 'dtypes changed'] == ''

        details = diff.details().data
        details.index = ['data']
        details.index.name = 'dataset'

        assert details.at['data', 'uid'] == 'uid'
        assert details.at['data', 'in both datasets'] == 'yes'
        assert details.at['data', 'cols shared'] == '3'
        assert details.at['data', 'rows shared'] == '3'
        assert details.at['data', 'cols added'] == ''
        assert details.at['data', 'cols removed'] == ''
        assert details.at['data', 'rows added'] == ''
        assert details.at['data', 'rows removed'] == ''
        assert details.at['data', 'dtypes changed'] == ''

        assert details.at['data', 'all cols added'] == ''
        assert details.at['data', 'all cols removed'] == ''
        assert details.at['data', 'all rows added'] == ''
        assert details.at['data', 'all rows removed'] == ''
        assert details.at['data', 'all dtypes changed'] == ''




def test_ignore_cols(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)


    #mode new, ignore a
    expected = get_expected_new()
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='a',
        verbosity=0,
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new, ignore b
    expected = get_expected_new()
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['b'],
        verbosity=0
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new, ignore a and b
    expected = get_expected_new()
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['a', 'b'],
        verbosity=0
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new+, ignore a
    expected = get_expected_newplus()
    expected.drop(columns=['a *old'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a'],
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new+, ignore b
    expected = get_expected_newplus()
    expected.drop(columns=['b *old'], inplace=True)
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='b',
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new+, ignore a and b
    expected = get_expected_newplus()
    expected.drop(columns=['a *old', 'b *old'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode old, ignore a
    expected = get_expected_old()
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='a',
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode old, ignore b
    expected = get_expected_old()
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='b',
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode old, ignore a and b
    expected = get_expected_old()
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode mix, ignore a
    expected = get_expected_mix()
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols='a',
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode mix, ignore b
    expected = get_expected_mix()
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['b'],
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode mix, ignore a and b
    expected = get_expected_mix()
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        ignore_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501




def test_remove_cols(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)


    #mode new, remove a
    expected = get_expected_new()
    expected.drop(columns=['a'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols='a',
        verbosity=0,
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new, remove b
    expected = get_expected_new()
    expected.drop(columns=['b'], inplace=True)
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['b'],
        verbosity=0
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new, remove a and b
    expected = get_expected_new()
    expected.drop(columns=['a', 'b'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('new').data
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['a', 'b'],
        verbosity=0
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new+, remove a
    expected = get_expected_newplus()
    expected.drop(columns=['a', 'a *old'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['a'],
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new+, remove b
    expected = get_expected_newplus()
    expected.drop(columns=['b', 'b *old'], inplace=True)
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols='b',
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode new+, remove a and b
    expected = get_expected_newplus()
    expected.drop(columns=['a', 'a *old', 'b', 'b *old'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode old, remove a
    expected = get_expected_old()
    expected.drop(columns=['a'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols='a',
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode old, remove b
    expected = get_expected_old()
    expected.drop(columns=['b'], inplace=True)
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols='b',
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode old, remove a and b
    expected = get_expected_old()
    expected.drop(columns=['a', 'b'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode mix, remove a
    expected = get_expected_mix()
    expected.drop(columns=['a'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = 'vals added: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols='a',
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode mix, remove b
    expected = get_expected_mix()
    expected.drop(columns=['b'], inplace=True)
    expected.loc['z', 'diff'] = 'vals removed: 1'

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['b'],
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



    #mode mix, remove a and b
    expected = get_expected_mix()
    expected.drop(columns=['a', 'b'], inplace=True)
    expected.loc['y', 'diff'] = ''
    expected.loc['z', 'diff'] = ''

    #in memory dfs
    result = qp.diff(
        df_old,
        df_new,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #files
    result = qp.diff(
        df_old_file,
        df_new_file,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #in memory df and file
    result = qp.diff(
        df_old,
        df_new_file,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        remove_cols=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501




def test_diff_of_diffs(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old.insert(1, 'notes', ['note1', 'note2', 'note3'])
    df_new.insert(1, 'notes', ['note3', 'note4', 'note5'])
    df_new1 = df_new.copy()
    df_new1.loc['y', 'a'] = 9

    path_df_old = f'{tmpdir}/df_old.xlsx'
    path_df_new = f'{tmpdir}/df_new.xlsx'
    path_df_new1 = f'{tmpdir}/df_new1.xlsx'
    path_diff1 = f'{tmpdir}/diff1.xlsx'
    path_diff2 = f'{tmpdir}/diff2.xlsx'
    df_old.to_excel(path_df_old, index=False)
    df_new.to_excel(path_df_new, index=False)
    df_new1.to_excel(path_df_new1, index=False)

    qp.diff(
        path_df_old,
        path_df_new,
        uid='uid',
        ignore_cols='notes',
        ).to_excel(path_diff1, index=False)

    qp.diff(
        path_diff1,
        path_df_new1,
        uid='uid',
        remove_cols='diff',
        remove_cols_by_suffix=' *old',
        retain_cols='notes',
        remove_sheets=['info', 'summary', 'details'],
        ).to_excel(path_diff2)

    sheets = openpyxl.load_workbook(path_diff2, read_only=True).sheetnames
    info = pd.read_excel(path_diff2, sheet_name='info', index_col=0)
    details = pd.read_excel(path_diff2, sheet_name='details', index_col=0).fillna('')

    assert sheets == ['info', 'summary', 'details', 'Sheet1']
    assert info.loc['old dataset', 'name'].split('/')[-1] == 'diff1.xlsx'
    assert info.loc['new dataset', 'name'].split('/')[-1] == 'df_new1.xlsx'

    col_val_mapping_details = {
        'in both datasets': 'yes',
        'cols shared': 3,
        'cols added': '',
        'cols removed': '',
        'rows shared': 3,
        'rows added': '',
        'rows removed': '',
        'dtypes changed': '',
        'all cols shared': "'a',\n'b',\n'd'",
        'all cols added': '',
        'all cols removed': '',
        'all rows shared': "'y',\n'x2',\n'z'",
        'all rows added': '',
        'all rows removed': '',
        'all dtypes changed': '',
        }
    for col, val in col_val_mapping_details.items():
        result = details.loc['Sheet1', col]
        expected = val
        assert result == expected, f'value in details: RESULT: {result}, EXPECTED: {expected}'  # noqa E501
