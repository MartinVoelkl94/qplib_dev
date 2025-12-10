import pandas as pd
import numpy as np
import qplib as qp
from numpy import dtype


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



def test_mode_new(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)


    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('new').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a'],
        index=['y', 'x2', 'z']
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1

    expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('new', 'Sheet1').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501


def test_mode_new_ignore(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore='a',
        verbosity=0,
        ).show('new').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a'],
        index=['y', 'x2', 'z']
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1

    expected.loc['z', 'meta'] = '<br>vals added: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_new_ignore1(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore=['b'],
        verbosity=0
        ).show('new').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a'],
        index=['y', 'x2', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1

    expected.loc['z', 'meta'] = '<br>vals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_new_ignore2(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore=['a', 'b'],
        verbosity=0,
        ).show('new').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a'],
        index=['y', 'x2', 'z']
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1

    expected.loc['z', 'meta'] = ''
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0
        ).show('new').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501








def test_mode_newplus(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        verbosity=0,
        ).show('new+').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'],
        index=['y', 'x2', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'old: d'] = ''
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'old: b'] = ''
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'old: a'] = 2

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'old: d'] = ''
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'old: b'] = ''
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'old: a'] = ''

    expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'old: d'] = ''
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'old: b'] = None
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'old: a'] = 3


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_newplus_ignore(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ignore=['a'],
        verbosity=0,
        ).show('new+').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'],
        index=['y', 'x2', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'old: d'] = ''
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'old: b'] = ''
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'old: a'] = ''

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'old: d'] = ''
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'old: b'] = ''
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'old: a'] = ''

    expected.loc['z', 'meta'] = '<br>vals added: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'old: d'] = ''
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'old: b'] = None
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'old: a'] = ''


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_newplus_ignore1(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('new+').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'],
        index=['y', 'x2', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'old: d'] = ''
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'old: b'] = ''
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'old: a'] = 2

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'old: d'] = ''
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'old: b'] = ''
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'old: a'] = ''

    expected.loc['z', 'meta'] = '<br>vals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'old: d'] = ''
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'old: b'] = ''
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'old: a'] = 3


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501




def test_mode_newplus_ignore2(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ignore=['b', 'a'],
        verbosity=0,
        ).show('new+').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'],
        index=['y', 'x2', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'old: d'] = ''
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'old: b'] = ''
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'old: a'] = ''

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'old: d'] = ''
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'old: b'] = ''
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'old: a'] = ''

    expected.loc['z', 'meta'] = ''
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'old: d'] = ''
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'old: b'] = ''
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'old: a'] = ''


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['b', 'a'],
        verbosity=0,
        ).show('new+').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_old(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('old').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'a', 'b', 'c'],
        index=['x', 'y', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'c'] = 1

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'a'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'c'] = 2

    expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    expected.loc['z', 'a'] = 3
    expected.loc['z', 'b'] = None
    expected.loc['z', 'c'] = 3


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501




def test_mode_old_ignore(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore='a',
        verbosity=0,
        ).show('old').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'a', 'b', 'c'],
        index=['x', 'y', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'c'] = 1

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'a'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'c'] = 2

    expected.loc['z', 'meta'] = '<br>vals added: 1'
    expected.loc['z', 'a'] = 3
    expected.loc['z', 'b'] = None
    expected.loc['z', 'c'] = 3


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501




def test_mode_old_ignore1(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore='b',
        verbosity=0,
        ).show('old').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'a', 'b', 'c'],
        index=['x', 'y', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'c'] = 1

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'a'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'c'] = 2

    expected.loc['z', 'meta'] = '<br>vals removed: 1'
    expected.loc['z', 'a'] = 3
    expected.loc['z', 'b'] = None
    expected.loc['z', 'c'] = 3


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore='b',
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501




def test_mode_old_ignore2(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore=['a', 'b'],
        verbosity=0,
        ).show('old').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'a', 'b', 'c'],
        index=['x', 'y', 'z'],
        )

    expected['uid'] = expected.index

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'c'] = 1

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'a'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'c'] = 2

    expected.loc['z', 'meta'] = ''
    expected.loc['z', 'a'] = 3
    expected.loc['z', 'b'] = None
    expected.loc['z', 'c'] = 3


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('old').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501





def test_mode_mix(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        verbosity=0,
        ).show('mix').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a', 'c'],
        index=['y', 'x2', 'z', 'x'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'c'] = 2

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'c'] = np.nan

    expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'c'] = 3

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_mix_ignore(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore='a',
        verbosity=0,
        ).show('mix').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a', 'c'],
        index=['y', 'x2', 'z', 'x'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'c'] = 2

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'c'] = np.nan

    expected.loc['z', 'meta'] = '<br>vals added: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'c'] = 3

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore='a',
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501





def test_mode_mix_ignore1(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore=['b'],
        verbosity=0,
        ).show('mix').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a', 'c'],
        index=['y', 'x2', 'z', 'x'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = '<br>vals changed: 1'
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'c'] = 2

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'c'] = np.nan

    expected.loc['z', 'meta'] = '<br>vals removed: 1'
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'c'] = 3

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_mode_mix_ignore2(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    result = qp.diff(
        df_old,
        df_new,
        ignore=['a', 'b'],
        verbosity=0,
        ).show('mix').data

    expected = pd.DataFrame(
        columns=['meta', 'uid', 'd', 'b', 'a', 'c'],
        index=['y', 'x2', 'z', 'x'],
        )

    expected['uid'] = expected.index

    expected.loc['y', 'meta'] = ''
    expected.loc['y', 'd'] = 2
    expected.loc['y', 'b'] = 2
    expected.loc['y', 'a'] = 0
    expected.loc['y', 'c'] = 2

    expected.loc['x2', 'meta'] = 'added row'
    expected.loc['x2', 'd'] = 1
    expected.loc['x2', 'b'] = 1
    expected.loc['x2', 'a'] = 1
    expected.loc['x2', 'c'] = np.nan

    expected.loc['z', 'meta'] = ''
    expected.loc['z', 'd'] = 3
    expected.loc['z', 'b'] = 3
    expected.loc['z', 'a'] = np.nan
    expected.loc['z', 'c'] = 3

    expected.loc['x', 'meta'] = 'removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501

    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['a', 'b'],
        verbosity=0,
        ).show('mix').data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'  # noqa E501



def test_summary_stats(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        )

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 2
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert not result.dtypes_changed[0]
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert not result.cols_ignored_new[0]
    assert not result.cols_ignored_old[0]


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        )

    dtypes = {
        'a': {
            'new': dtype('float64'),
            'old': dtype('int64'),
            },
        'b': {
            'new': dtype('int64'),
            'old': dtype('float64'),
            },
        }

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 2
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert result.dtypes_changed[0] == dtypes
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert not result.cols_ignored_new[0]
    assert not result.cols_ignored_old[0]



def test_summary_stats_ignore(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        ignore='a',
        )

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 1
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert not result.dtypes_changed[0]
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert result.cols_ignored_new[0] == ['a']
    assert result.cols_ignored_old[0] == ['a']


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='a',
        )

    dtypes = {
        'b': {
            'new': dtype('int64'),
            'old': dtype('float64'),
            },
        }

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 1
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert result.dtypes_changed[0] == dtypes
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert result.cols_ignored_new[0] == ['a']
    assert result.cols_ignored_old[0] == ['a']



def test_summary_stats_ignore1(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        ignore=['b'],
        )

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 1
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert not result.dtypes_changed[0]
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert result.cols_ignored_new[0] == ['b']
    assert result.cols_ignored_old[0] == ['b']


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['b'],
        )

    dtypes = {
        'a': {
            'new': dtype('float64'),
            'old': dtype('int64'),
            },
        }

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 1
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert result.dtypes_changed[0] == dtypes
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert result.cols_ignored_new[0] == ['b']
    assert result.cols_ignored_old[0] == ['b']



def test_summary_stats_ignore2(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        ignore=['a', 'b'],
        )

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 0
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert not result.dtypes_changed[0]
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert result.cols_ignored_new[0] == ['b', 'a']
    assert result.cols_ignored_old[0] == ['a', 'b']


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        )

    assert result.uid_cols[0] == 'uid'
    assert len(result.cols_shared[0]) == 0
    assert len(result.rows_shared[0]) == 2
    assert result.cols_added[0] == ['d']
    assert result.cols_removed[0] == ['c']
    assert result.rows_added[0] == ['x2']
    assert result.rows_removed[0] == ['x']
    assert not result.dtypes_changed[0]
    assert not result.cols_renamed_new[0]
    assert not result.cols_renamed_old[0]
    assert result.cols_ignored_new[0] == ['b', 'a']
    assert result.cols_ignored_old[0] == ['a', 'b']



def test_summary_str(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 2
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:

        cols ignored in old:

        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ).str()

    expected = """
        Diff between 2 excel files with 1 sheets

        Sheet: Sheet1
        in both files: yes
        uid col: uid
        cols shared: 2
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            a: int64 -> float64
            b: float64 -> int64

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:

        cols ignored in old:

        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 2
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            a: object -> float64
            b: object -> int64

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:

        cols ignored in old:

        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 2
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            a: int64 -> object
            b: float64 -> object

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:

        cols ignored in old:

        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



def test_summary_str_ignore(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ignore='a',
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            a
        cols ignored in old:
            a
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore='a',
        ).str()

    expected = """
        Diff between 2 excel files with 1 sheets

        Sheet: Sheet1
        in both files: yes
        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            b: float64 -> int64

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            a
        cols ignored in old:
            a
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore='a',
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            b: object -> int64

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            a
        cols ignored in old:
            a
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore='a',
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            b: float64 -> object

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            a
        cols ignored in old:
            a
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



def test_summary_str_ignore1(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ignore=['b'],
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b
        cols ignored in old:
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['b'],
        ).str()

    expected = """
        Diff between 2 excel files with 1 sheets

        Sheet: Sheet1
        in both files: yes
        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            a: int64 -> float64

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b
        cols ignored in old:
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['b'],
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            a: object -> float64

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b
        cols ignored in old:
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['b'],
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 1
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:
            a: int64 -> object

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b
        cols ignored in old:
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



def test_summary_str_ignore2(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_file, df_new_file = setup(df_old, df_new, tmpdir)

    #reading from in memory df
    result = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ignore=['a', 'b'],
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 0
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b;
            a
        cols ignored in old:
            a;
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from files
    result = qp.diff(
        df_old_file,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        ).str()

    expected = """
        Diff between 2 excel files with 1 sheets

        Sheet: Sheet1
        in both files: yes
        uid col: uid
        cols shared: 0
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b;
            a
        cols ignored in old:
            a;
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old,
        df_new_file,
        uid='uid',
        ignore=['a', 'b'],
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 0
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b;
            a
        cols ignored in old:
            a;
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501


    #reading from file and in memory df
    result = qp.diff(
        df_old_file,
        df_new,
        uid='uid',
        ignore=['a', 'b'],
        ).str()

    expected = """
        Diff between 2 dataframes

        uid col: uid
        cols shared: 0
        cols added:
            d
        cols removed:
            c
        rows shared: 2
        rows added:
            x2
        rows removed:
            x
        dtypes changed:

        cols renamed in new:

        cols renamed in old:

        cols ignored in new:
            b;
            a
        cols ignored in old:
            a;
            b
        """

    assert process_str(result) == process_str(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'  # noqa E501



def test_csv(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_csv, df_new_csv = setup_csv(df_old, df_new, tmpdir)

    result_df = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ).show('new').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        uid='uid',
        ).show('new').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501

    result_df = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ).show('new+').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        uid='uid',
        ).show('new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501

    result_df = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ).show('new+').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        uid='uid',
        ).show('new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501

    result_df = qp.diff(
        df_old,
        df_new,
        uid='uid',
        ).show('new+').data.astype('object')
    result_csv = qp.diff(
        df_old_csv,
        df_new_csv,
        uid='uid',
        ).show('new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'  # noqa E501



def test_identical(tmpdir):
    df_old, df_new = qp.get_dfs()
    df_old_xlsx, df_new_xlsx = setup(df_old, df_new, tmpdir)
    df_old_csv, df_new_csv = setup_csv(df_old, df_new, tmpdir)

    diff1 = qp.diff(
        df_new,
        df_new,
        uid='uid',
        )
    diff2 = qp.diff(
        df_new_xlsx,
        df_new_xlsx,
        uid='uid',
        )
    diff3 = qp.diff(
        df_new_csv,
        df_new_csv,
        uid='uid',
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

    expected = 'both dataframes are identical'
    expected3 = (
        'Diff between 2 excel files with 1 sheets'
        'Sheet "Sheet1" is identical in both files'
        )

    assert str1 == str3 == expected, f"Strings are not equal:\n{str1}\n{str3}"  # noqa E501
    assert process_str(str2) == process_str(expected3), f"EXPECTED:\n{expected3}\nRESULT:\n{str2}"  # noqa E501

    for diff in [diff1, diff2, diff3]:
        assert diff.uid_cols[0] == 'uid'
        assert len(diff.cols_shared[0]) == 3
        assert len(diff.rows_shared[0]) == 3
        assert len(diff.cols_added[0]) == 0
        assert len(diff.cols_removed[0]) == 0
        assert len(diff.rows_added[0]) == 0
        assert len(diff.rows_removed[0]) == 0
        assert not diff.dtypes_changed[0]
        assert not diff.cols_renamed_new[0]
        assert not diff.cols_renamed_old[0]
        assert not diff.cols_ignored_new[0]
        assert not diff.cols_ignored_old[0]
