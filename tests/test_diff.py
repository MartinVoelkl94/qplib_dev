import pandas as pd
import numpy as np
import qplib as qp

df_new, df_old = qp.get_dfs()


def test_mode_new():
    result = qp.diff(df_new, df_old, 'new', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nexpected:\n{result_expected}\nresult:\n{result}'


def test_mode_newplus():
    result = qp.diff(df_new, df_old, 'new+', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'old: d'] = ''
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'old: b'] = ''
    result_expected.loc['y', 'a'] = 0.0
    result_expected.loc['y', 'old: a'] = 2.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'old: d'] = ''
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'old: b'] = ''
    result_expected.loc['x2', 'a'] = 1.0
    result_expected.loc['x2', 'old: a'] = ''

    result_expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'old: d'] = ''
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'old: b'] = None
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'old: a'] = 3.0

    assert result.equals(result_expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nexpected:\n{result_expected}\nresult:\n{result}'


def test_mode_old():
    result = qp.diff(df_new, df_old, 'old', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'a'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    result_expected.loc['z', 'a'] = 3.0
    result_expected.loc['z', 'b'] = None
    result_expected.loc['z', 'c'] = 3.0

    assert result.equals(result_expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nexpected:\n{result_expected}\nresult:\n{result}'


def test_mode_mix():
    result = qp.diff(df_new, df_old, 'mix', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0
    result_expected.loc['x2', 'c'] = np.nan

    result_expected.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'c'] = 3.0

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'd'] = None
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    assert result.equals(result_expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nexpected:\n{result_expected}\nresult:\n{result}'





