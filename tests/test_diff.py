import pandas as pd
import numpy as np
import qplib as qp
import os
import shutil
import pytest




#prepare testing files

def setup(df_new, df_old, tmpdir):

    path_df_new = f'{tmpdir}/df_new.xlsx'
    df_new.to_excel(path_df_new, index=True)
    
    path_df_old = f'{tmpdir}/df_old.xlsx'
    df_old.to_excel(path_df_old, index=True)

    return path_df_new, path_df_old





def test_mode_new(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_new_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', ignore='a', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', ignore='a', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_new_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', ignore=['b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', ignore=['b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_new_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', ignore=['a', 'b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', ignore=['a', 'b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'








def test_mode_newplus(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new+', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new+', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_newplus_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new+', ignore=['a'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new+', ignore=['a'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_newplus_ingnore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new+', ignore='b', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new+', ignore='b', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_newplus_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new+', ignore=['b', 'a'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new+', ignore=['b', 'a'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_mode_old(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'old', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    expected.loc['x', 'meta'] ='removed row'
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
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'old', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_old_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'old', ignore='a', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    expected.loc['x', 'meta'] ='removed row'
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
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'old', ignore='a', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_old_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'old', ignore='b', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    expected.loc['x', 'meta'] ='removed row'
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
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'old', ignore='b', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_old_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'old', ignore=['a', 'b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=['x','y','z'])

    expected.loc['x', 'meta'] ='removed row'
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
    assert result.equals(expected), f'failed test for mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'old', ignore=['a', 'b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_mode_mix(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'mix', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'mix', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_mix_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'mix', ignore='a', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'mix', ignore='a', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_mode_mix_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'mix', ignore=['b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'mix', ignore=['b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_mix_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'mix', ignore=['a','b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'mix', ignore=['a','b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_returns_df(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_returns_df_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', ignore='a', output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', ignore='a', output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_returns_df_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', ignore=['b'], output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', ignore=['b'], output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_returns_df_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, 'new', ignore=['b', 'a'], output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, 'new', ignore=['b', 'a'], output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_returns_summary(tmpdir):
    df_new, df_old = qp.get_dfs()
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

    #reading from in memory df
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, output='summary')
    expected = pd.DataFrame({
        'sheet': ['Sheet1'],
        'cols added': [1],
        'cols removed': [1],
        'rows added': [1],
        'rows removed': [1],
        'vals added': [1],
        'vals removed': [1],
        'vals changed': [1],
        'is in "df_new.xlsx"': [True],
        'is in "df_old.xlsx"': [True],
        'index_col is unique in "df_new.xlsx"': [True],
        'index_col is unique in "df_old.xlsx"': [True],
        }).astype('object')
    assert result.equals(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_summary_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
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

    #reading from in memory df
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, ignore='a', output='summary')
    expected = pd.DataFrame({
        'sheet': ['Sheet1'],
        'cols added': [1],
        'cols removed': [1],
        'rows added': [1],
        'rows removed': [1],
        'vals added': [1],
        'vals removed': [0],
        'vals changed': [0],
        'is in "df_new.xlsx"': [True],
        'is in "df_old.xlsx"': [True],
        'index_col is unique in "df_new.xlsx"': [True],
        'index_col is unique in "df_old.xlsx"': [True],
        }).astype('object')
    assert result.equals(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_summary_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
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

    #reading from in memory df
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, ignore=['b'], output='summary')
    expected = pd.DataFrame({
        'sheet': ['Sheet1'],
        'cols added': [1],
        'cols removed': [1],
        'rows added': [1],
        'rows removed': [1],
        'vals added': [0],
        'vals removed': [1],
        'vals changed': [1],
        'is in "df_new.xlsx"': [True],
        'is in "df_old.xlsx"': [True],
        'index_col is unique in "df_new.xlsx"': [True],
        'index_col is unique in "df_old.xlsx"': [True],
        }).astype('object')
    assert result.equals(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_summary_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
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

    #reading from in memory df
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, ignore=['a','b'], output='summary')
    expected = pd.DataFrame({
        'sheet': ['Sheet1'],
        'cols added': [1],
        'cols removed': [1],
        'rows added': [1],
        'rows removed': [1],
        'vals added': [0],
        'vals removed': [0],
        'vals changed': [0],
        'is in "df_new.xlsx"': [True],
        'is in "df_old.xlsx"': [True],
        'index_col is unique in "df_new.xlsx"': [True],
        'index_col is unique in "df_old.xlsx"': [True],
        }).astype('object')
    assert result.equals(expected), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'




def test_returns_str(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     b    a\ny  nan  0\nz  3  nan\n\ndifferent values in df_old:\n      b    a\ny   nan  2\nz  None  3\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, output='str')
    assert result == 'no string version of differences available when comparing excel files'




def test_returns_str_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, ignore='a', output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     b\ny  nan\nz  3\n\ndifferent values in df_old:\n      b\ny   nan\nz  None\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, ignore='a', output='str')
    assert result == 'no string version of differences available when comparing excel files'



def test_returns_str_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, ignore=['b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     a\ny  0\nz  nan\n\ndifferent values in df_old:\n      a\ny   2\nz  3\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, ignore=['b'], output='str')
    assert result == 'no string version of differences available when comparing excel files'



def test_returns_str_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, ignore=['a','b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     Empty DataFrame\nColumns: []\nIndex: [y, z]\n\ndifferent values in df_old:\n      Empty DataFrame\nColumns: []\nIndex: [y, z]\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, ignore=['a','b'], output='str')
    assert result == 'no string version of differences available when comparing excel files'





def test_returns_all(tmpdir):
    df_new, df_old = qp.get_dfs()
    result_df_styler, result_summary, result_str = qp.diff(df_new, df_old, output='all')
    result_df = result_df_styler.data

    expected_df = pd.DataFrame(columns=['meta', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

    expected_df.loc['y', 'meta'] = '<br>vals changed: 1'
    expected_df.loc['y', 'd'] = 2
    expected_df.loc['y', 'b'] = 2
    expected_df.loc['y', 'a'] = 0
    expected_df.loc['y', 'c'] = 2

    expected_df.loc['x2', 'meta'] = 'added row'
    expected_df.loc['x2', 'd'] = 1
    expected_df.loc['x2', 'b'] = 1
    expected_df.loc['x2', 'a'] = 1
    expected_df.loc['x2', 'c'] = np.nan

    expected_df.loc['z', 'meta'] = '<br>vals added: 1<br>vals removed: 1'
    expected_df.loc['z', 'd'] = 3
    expected_df.loc['z', 'b'] = 3
    expected_df.loc['z', 'a'] = np.nan
    expected_df.loc['z', 'c'] = 3

    expected_df.loc['x', 'meta'] ='removed row'
    expected_df.loc['x', 'd'] = None
    expected_df.loc['x', 'b'] = 1
    expected_df.loc['x', 'a'] = 1
    expected_df.loc['x', 'c'] = 1

    expected_summary = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 1,
        'vals changed': 1,
        }
    
    expected_str = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n     b    a\ny  nan  0\nz  3  nan\n\ndifferent values in df_old:\n      b    a\ny   nan  2\nz  None  3\n"


    #reading from in memory df
    assert result_df.equals(expected_df), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_df}\nRESULT\n{result_df}'
    assert result_summary == expected_summary, f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_summary}\nRESULT\n{result_summary}'
    assert result_str.replace(' ','') == expected_str.replace(' ',''), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_str}\nRESU\n{result_str}'
    
    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result_dfs, result_summary, result_str = qp.diff(df_new, df_old, output='all')
    result_df = result_dfs['Sheet1'].data.astype('object')
    expected_summary = pd.DataFrame({
        'sheet': ['Sheet1'],
        'cols added': [1],
        'cols removed': [1],
        'rows added': [1],
        'rows removed': [1],
        'vals added': [1],
        'vals removed': [1],
        'vals changed': [1],
        'is in "df_new.xlsx"': [True],
        'is in "df_old.xlsx"': [True],
        'index_col is unique in "df_new.xlsx"': [True],
        'index_col is unique in "df_old.xlsx"': [True],
        }).astype('object')
    assert result_df.equals(expected_df), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_df}\nRESULT\n{result_df}'
    assert result_summary.equals(expected_summary)
    assert result_str == 'no string version of differences available when comparing excel files'
    




def test_returns_print(tmpdir):
    df_new, df_old = qp.get_dfs()
    result = qp.diff(df_new, df_old, output='print')
    expected = None

    #reading from in memory df
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    df_new, df_old = setup(df_new, df_old, tmpdir)
    result = qp.diff(df_new, df_old, output='print')
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'





