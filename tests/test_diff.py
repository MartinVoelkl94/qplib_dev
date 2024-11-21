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

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'


def test_mode_new_ignore():
    result = qp.diff(df_new, df_old, 'new', ignore='a', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = '<br>vals added: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'


def test_mode_new_ignore1():
    result = qp.diff(df_new, df_old, 'new', ignore=['b'], verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = '<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'


def test_mode_new_ignore2():
    result = qp.diff(df_new, df_old, 'new', ignore=['a', 'b'], verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = ''
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'







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

    assert result.equals(result_expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT\n{result}'



def test_mode_newplus_ignore():
    result = qp.diff(df_new, df_old, 'new+', ignore=['a'], verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'old: d'] = ''
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'old: b'] = ''
    result_expected.loc['y', 'a'] = 0.0
    result_expected.loc['y', 'old: a'] = ''

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'old: d'] = ''
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'old: b'] = ''
    result_expected.loc['x2', 'a'] = 1.0
    result_expected.loc['x2', 'old: a'] = ''

    result_expected.loc['z', 'meta'] = '<br>vals added: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'old: d'] = ''
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'old: b'] = None
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'old: a'] = ''

    assert result.equals(result_expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT\n{result}'



def test_mode_newplus_ingnore1():
    result = qp.diff(df_new, df_old, 'new+', ignore='b', verbosity=0).data

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

    result_expected.loc['z', 'meta'] = '<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'old: d'] = ''
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'old: b'] = ''
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'old: a'] = 3.0

    assert result.equals(result_expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT\n{result}'



def test_mode_newplus_ignore2():
    result = qp.diff(df_new, df_old, 'new+', ignore=['b', 'a'], verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'old: d'] = ''
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'old: b'] = ''
    result_expected.loc['y', 'a'] = 0.0
    result_expected.loc['y', 'old: a'] = ''

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'old: d'] = ''
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'old: b'] = ''
    result_expected.loc['x2', 'a'] = 1.0
    result_expected.loc['x2', 'old: a'] = ''

    result_expected.loc['z', 'meta'] = ''
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'old: d'] = ''
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'old: b'] = ''
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'old: a'] = ''

    assert result.equals(result_expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT\n{result}'




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

    assert result.equals(result_expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'



def test_mode_old_ignore():
    result = qp.diff(df_new, df_old, 'old', ignore='a', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'a'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['z', 'meta'] = '<br>vals added: 1'
    result_expected.loc['z', 'a'] = 3.0
    result_expected.loc['z', 'b'] = None
    result_expected.loc['z', 'c'] = 3.0

    assert result.equals(result_expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'



def test_mode_old_ignore1():
    result = qp.diff(df_new, df_old, 'old', ignore='b', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'a'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['z', 'meta'] = '<br>vals removed: 1'
    result_expected.loc['z', 'a'] = 3.0
    result_expected.loc['z', 'b'] = None
    result_expected.loc['z', 'c'] = 3.0

    assert result.equals(result_expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'



def test_mode_old_ignore2():
    result = qp.diff(df_new, df_old, 'old', ignore=['a', 'b'], verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'a'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['z', 'meta'] = ''
    result_expected.loc['z', 'a'] = 3.0
    result_expected.loc['z', 'b'] = None
    result_expected.loc['z', 'c'] = 3.0

    assert result.equals(result_expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'




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

    assert result.equals(result_expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'


def test_mode_mix_ignore():
    result = qp.diff(df_new, df_old, 'mix', ignore='a', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0
    result_expected.loc['x2', 'c'] = np.nan

    result_expected.loc['z', 'meta'] = '<br>vals added: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'c'] = 3.0

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'd'] = None
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    assert result.equals(result_expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'




def test_mode_mix_ignore1():
    result = qp.diff(df_new, df_old, 'mix', ignore=['b'], verbosity=0).data

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

    result_expected.loc['z', 'meta'] = '<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'c'] = 3.0

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'd'] = None
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    assert result.equals(result_expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'


def test_mode_mix_ignore2():
    result = qp.diff(df_new, df_old, 'mix', ignore=['a','b'], verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0
    result_expected.loc['y', 'c'] = 2.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0
    result_expected.loc['x2', 'c'] = np.nan

    result_expected.loc['z', 'meta'] = ''
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan
    result_expected.loc['z', 'c'] = 3.0

    result_expected.loc['x', 'meta'] ='removed row'
    result_expected.loc['x', 'd'] = None
    result_expected.loc['x', 'b'] = 1.0
    result_expected.loc['x', 'a'] = 1.0
    result_expected.loc['x', 'c'] = 1.0

    assert result.equals(result_expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'




def test_returns_df():
    result = qp.diff(df_new, df_old, 'new', output='df', verbosity=0).data

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

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'



def test_returns_df_ignore():
    result = qp.diff(df_new, df_old, 'new', ignore='a', output='df', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = '<br>vals added: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'



def test_returns_df_ignore1():
    result = qp.diff(df_new, df_old, 'new', ignore=['b'], output='df', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = '<br>vals changed: 1'
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = '<br>vals removed: 1'
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'



def test_returns_df_ignore2():
    result = qp.diff(df_new, df_old, 'new', ignore=['b', 'a'], output='df', verbosity=0).data

    result_expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

    result_expected.loc['y', 'meta'] = ''
    result_expected.loc['y', 'd'] = 2.0
    result_expected.loc['y', 'b'] = 2.0
    result_expected.loc['y', 'a'] = 0.0

    result_expected.loc['x2', 'meta'] = 'added row'
    result_expected.loc['x2', 'd'] = 1.0
    result_expected.loc['x2', 'b'] = 1.0
    result_expected.loc['x2', 'a'] = 1.0

    result_expected.loc['z', 'meta'] = ''
    result_expected.loc['z', 'd'] = 3.0
    result_expected.loc['z', 'b'] = 3.0
    result_expected.loc['z', 'a'] = np.nan

    assert result.equals(result_expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_expected}\nRESULT:\n{result}'




def test_returns_summary():
    result = qp.diff(df_new, df_old, output='summary')
    expected = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 1,
        'vals changed': 1,
        }
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'


def test_returns_summary_ignore():
    result = qp.diff(df_new, df_old, ignore='a', output='summary')
    expected = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 0,
        'vals changed': 0,
        }
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'


def test_returns_summary_ignore1():
    result = qp.diff(df_new, df_old, ignore=['b'], output='summary')
    expected = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 0,
        'vals removed': 1,
        'vals changed': 1,
        }
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'


def test_returns_summary_ignore2():
    result = qp.diff(df_new, df_old, ignore=['a','b'], output='summary')
    expected = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 0,
        'vals removed': 0,
        'vals changed': 0,
        }
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_str():
    result = qp.diff(df_new, df_old, output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     b    a\ny  nan  0.0\nz  3.0  nan\n\ndifferent values in df_old:\n      b    a\ny   nan  2.0\nz  None  3.0\n".replace('\t','')
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_str_ignore():
    result = qp.diff(df_new, df_old, ignore='a', output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     b\ny  nan\nz  3.0\n\ndifferent values in df_old:\n      b\ny   nan\nz  None\n".replace('\t','')
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'


def test_returns_str_ignore1():
    result = qp.diff(df_new, df_old, ignore=['b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     a\ny  0.0\nz  nan\n\ndifferent values in df_old:\n      a\ny   2.0\nz  3.0\n".replace('\t','')
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'


def test_returns_str_ignore2():
    result = qp.diff(df_new, df_old, ignore=['a','b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     Empty DataFrame\nColumns: []\nIndex: [y, z]\n\ndifferent values in df_old:\n      Empty DataFrame\nColumns: []\nIndex: [y, z]\n".replace('\t','')
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'




def test_returns_all():
    result_df_styler, result_summary, result_str = qp.diff(df_new, df_old, output='all')
    result_df = result_df_styler.data

    expected_df = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

    expected_df.loc['y', 'meta'] = '<br>vals changed: 1'
    expected_df.loc['y', 'd'] = 2.0
    expected_df.loc['y', 'b'] = 2.0
    expected_df.loc['y', 'a'] = 0.0
    expected_df.loc['y', 'c'] = 2.0

    expected_df.loc['x2', 'meta'] = 'added row'
    expected_df.loc['x2', 'd'] = 1.0
    expected_df.loc['x2', 'b'] = 1.0
    expected_df.loc['x2', 'a'] = 1.0
    expected_df.loc['x2', 'c'] = np.nan

    expected_df.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    expected_df.loc['z', 'd'] = 3.0
    expected_df.loc['z', 'b'] = 3.0
    expected_df.loc['z', 'a'] = np.nan
    expected_df.loc['z', 'c'] = 3.0

    expected_df.loc['x', 'meta'] ='removed row'
    expected_df.loc['x', 'd'] = None
    expected_df.loc['x', 'b'] = 1.0
    expected_df.loc['x', 'a'] = 1.0
    expected_df.loc['x', 'c'] = 1.0

    expected_summary = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 1,
        'vals changed': 1,
        }
    
    expected_str = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     b    a\ny  nan  0.0\nz  3.0  nan\n\ndifferent values in df_old:\n      b    a\ny   nan  2.0\nz  None  3.0\n"

    assert result_df.equals(expected_df), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_df}\nRESULT\n{result_df}'
    assert result_summary == expected_summary, f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_summary}\nRESULT\n{result_summary}'
    assert result_str.replace(' ','') == expected_str.replace(' ',''), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_str}\nRESU\n{result_str}'
    

def test_returns_print():
    result = qp.diff(df_new, df_old, output='print')
    expected = None
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'




