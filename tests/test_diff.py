import pandas as pd
import numpy as np
import qplib as qp




#prepare testing files

def setup(df_new, df_old, tmpdir):

    path_df_new = f'{tmpdir}/df_new.xlsx'
    df_new.to_excel(path_df_new, index=False)
    
    path_df_old = f'{tmpdir}/df_old.xlsx'
    df_old.to_excel(path_df_old, index=False)

    return path_df_new, path_df_old


def setup_csv(df_new, df_old, tmpdir):

    path_df_new = f'{tmpdir}/df_new.csv'
    df_new.to_csv(path_df_new, index=False)
    
    path_df_old = f'{tmpdir}/df_old.csv'
    df_old.to_csv(path_df_old, index=False)

    return path_df_new, path_df_old



def test_mode_new(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)


    result = qp.diff(df_new, df_old, 'new', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'


def test_mode_new_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', ignore='a', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', ignore='a', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', ignore='a', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
        
    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', ignore='a', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_new_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', ignore=['b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])
    
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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', ignore=['b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', ignore=['b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', ignore=['b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_new_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', ignore=['a', 'b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', ignore=['a', 'b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', ignore=['a', 'b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'
    
    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', ignore=['a', 'b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'








def test_mode_newplus(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new+', uid='uid', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new+', uid='uid', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new+', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new+', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_newplus_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new+', uid='uid', ignore=['a'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new+', uid='uid', ignore=['a'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new+', uid='uid', ignore=['a'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new+', uid='uid', ignore=['a'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_newplus_ingnore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new+', uid='uid', ignore='b', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new+', uid='uid', ignore='b', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new+', uid='uid', ignore='b', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new+', uid='uid', ignore='b', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_newplus_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new+', uid='uid', ignore=['b', 'a'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'old: d', 'b', 'old: b', 'a', 'old: a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new+', uid='uid', ignore=['b', 'a'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new+', uid='uid', ignore=['b', 'a'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new+', uid='uid', ignore=['b', 'a'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new+".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_old(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'old', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'a', 'b', 'c'], index=['x','y','z'])

    expected['uid'] = expected.index

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
    result = qp.diff(df_new_file, df_old_file, 'old', uid='uid', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'old', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'old', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_old_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'old', ignore='a', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'a', 'b', 'c'], index=['x','y','z'])

    expected['uid'] = expected.index

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
    result = qp.diff(df_new_file, df_old_file, 'old', uid='uid', ignore='a', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'old', uid='uid', ignore='a', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'old', uid='uid', ignore='a', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_old_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'old', ignore='b', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'a', 'b', 'c'], index=['x','y','z'])

    expected['uid'] = expected.index

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
    result = qp.diff(df_new_file, df_old_file, 'old', uid='uid', ignore='b', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'old', uid='uid', ignore='b', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'old', uid='uid', ignore='b', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_mode_old_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'old', ignore=['a', 'b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'a', 'b', 'c'], index=['x','y','z'])

    expected['uid'] = expected.index

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
    result = qp.diff(df_new_file, df_old_file, 'old', uid='uid', ignore=['a', 'b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'old', uid='uid', ignore=['a', 'b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'old', uid='uid', ignore=['a', 'b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "old".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_mode_mix(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'mix', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'mix', uid='uid', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'mix', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'mix', uid='uid', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_mix_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'mix', ignore='a', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'mix', uid='uid', ignore='a', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'mix', uid='uid', ignore='a', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'mix', uid='uid', ignore='a', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_mode_mix_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'mix', ignore=['b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'mix', uid='uid', ignore=['b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'mix', uid='uid', ignore=['b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'mix', uid='uid', ignore=['b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'



def test_mode_mix_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'mix', ignore=['a','b'], verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

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

    expected.loc['x', 'meta'] ='removed row'
    expected.loc['x', 'd'] = None
    expected.loc['x', 'b'] = 1
    expected.loc['x', 'a'] = 1
    expected.loc['x', 'c'] = 1


    #reading from in memory df
    assert result.equals(expected), f'failed test for mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'mix', uid='uid', ignore=['a','b'], verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'mix', uid='uid', ignore=['a','b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'mix', uid='uid', ignore=['a','b'], verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "mix".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_returns_df(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_returns_df_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', ignore='a', output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', ignore='a', output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', ignore='a', output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', ignore='a', output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_returns_df_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', ignore=['b'], output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', ignore=['b'], output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', ignore=['b'], output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', ignore=['b'], output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'




def test_returns_df_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, 'new', ignore=['b', 'a'], output='df', verbosity=0).data

    expected = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a'], index=['y','x2','z'])

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
    assert result.equals(expected), f'failed test for mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, 'new', uid='uid', ignore=['b', 'a'], output='df', verbosity=0)['Sheet1'].data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, 'new', uid='uid', ignore=['b', 'a'], output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, 'new', uid='uid', ignore=['b', 'a'], output='df', verbosity=0).data.astype('object')
    assert result.equals(expected), f'failed test for mode: "new".\nEXPECTED:\n{expected}\nRESULT:\n{result}'





def test_returns_summary(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

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
    result = qp.diff(df_new_file, df_old_file, uid='uid', output='summary')
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
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

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
    result = qp.diff(df_new_file, df_old_file, uid='uid', ignore='a', output='summary')
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
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

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
    result = qp.diff(df_new_file, df_old_file, uid='uid', ignore=['b'], output='summary')
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
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

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
    result = qp.diff(df_new_file, df_old_file, uid='uid', ignore=['a','b'], output='summary')
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
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, uid='uid', output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a    b\nuid          \ny      0  nan\nz    nan    3\n\ndifferent values in df_old:\n     a     b\nuid         \ny    2   nan\nz    3  None\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, uid='uid', output='str')
    assert result == 'no string version of differences available when comparing excel files'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, uid='uid', output='str')
    expected = "only in df_new:\ndtypes: {'a': dtype('float64'), 'b': dtype('int64')}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {'a': dtype('O'), 'b': dtype('O')}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a    b\nuid          \ny    0.0  nan\nz    nan  3.0\n\ndifferent values in df_old:\n     a     b\nuid         \ny    2   nan\nz    3  None\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, uid='uid', output='str')
    expected = "only in df_new:\ndtypes: {'a': dtype('O'), 'b': dtype('O')}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {'a': dtype('int64'), 'b': dtype('float64')}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a    b\nuid          \ny      0  nan\nz    nan    3\n\ndifferent values in df_old:\n     a    b\nuid        \ny    2  nan\nz    3  nan\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'


def test_returns_str_ignore(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, uid='uid', ignore='a', output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       b\nuid     \ny    nan\nz      3\n\ndifferent values in df_old:\n        b\nuid      \ny     nan\nz    None\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, uid='uid', ignore='a', output='str')
    assert result == 'no string version of differences available when comparing excel files'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, uid='uid', ignore='a', output='str')
    expected = "only in df_new:\ndtypes: {'b': dtype('int64')}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {'b': dtype('O')}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       b\nuid     \ny    nan\nz    3.0\n\ndifferent values in df_old:\n        b\nuid      \ny     nan\nz    None\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, uid='uid', ignore='a', output='str')
    expected = "only in df_new:\ndtypes: {'b': dtype('O')}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {'b': dtype('float64')}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       b\nuid     \ny    nan\nz      3\n\ndifferent values in df_old:\n       b\nuid     \ny    nan\nz    nan\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_str_ignore1(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, uid='uid', ignore=['b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a\nuid     \ny      0\nz    nan\n\ndifferent values in df_old:\n     a\nuid   \ny    2\nz    3\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, uid='uid', ignore=['b'], output='str')
    assert result == 'no string version of differences available when comparing excel files'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, uid='uid', ignore=['b'], output='str')
    expected = "only in df_new:\ndtypes: {'a': dtype('float64')}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {'a': dtype('O')}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a\nuid     \ny    0.0\nz    nan\n\ndifferent values in df_old:\n     a\nuid   \ny    2\nz    3\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, uid='uid', ignore=['b'], output='str')
    expected = "only in df_new:\ndtypes: {'a': dtype('O')}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {'a': dtype('int64')}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a\nuid     \ny      0\nz    nan\n\ndifferent values in df_old:\n     a\nuid   \ny    2\nz    3\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_returns_str_ignore2(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, uid='uid', ignore=['a','b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\nEmpty DataFrame\nColumns: []\nIndex: [y, z]\n\ndifferent values in df_old:\nEmpty DataFrame\nColumns: []\nIndex: [y, z]\n".replace('\t','')

    #reading from in memory df
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, uid='uid', ignore=['a','b'], output='str')
    assert result == 'no string version of differences available when comparing excel files'

    #reading from file and in memory df
    result = qp.diff(df_new_file, df_old, uid='uid', ignore=['a','b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\nEmpty DataFrame\nColumns: []\nIndex: [y, z]\n\ndifferent values in df_old:\nEmpty DataFrame\nColumns: []\nIndex: [y, z]\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from file and in memory df
    result = qp.diff(df_new, df_old_file, uid='uid', ignore=['a','b'], output='str')
    expected = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\nEmpty DataFrame\nColumns: []\nIndex: [y, z]\n\ndifferent values in df_old:\nEmpty DataFrame\nColumns: []\nIndex: [y, z]\n"
    assert result.replace(' ', '') == expected.replace(' ', ''), f'\nRESULT\n{result}\nEXPECTED:\n{expected}'





def test_returns_all(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result_df_styler, result_summary, result_str = qp.diff(df_new, df_old, uid='uid', output='all')
    result_df = result_df_styler.data

    expected_df = pd.DataFrame(columns=['meta', 'uid', 'd', 'b', 'a', 'c'], index=['y', 'x2', 'z', 'x'])

    expected_df['uid'] = expected_df.index
    
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
    
    expected_str = "only in df_new:\ndtypes: {}\nindices: ['x2']\nheaders: ['d']\nonly in df_old:\ndtypes: {}\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df_new:\n       a    b\nuid          \ny      0  nan\nz    nan    3\n\ndifferent values in df_old:\n     a     b\nuid         \ny    2   nan\nz    3  None\n"


    #reading from in memory df
    assert result_df.equals(expected_df), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_df}\nRESULT\n{result_df}'
    assert result_summary == expected_summary, f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_summary}\nRESULT\n{result_summary}'
    assert result_str.replace(' ','') == expected_str.replace(' ',''), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_str}\nRESU\n{result_str}'
    
    #reading from files
    result_dfs, result_summary, result_str = qp.diff(df_new_file, df_old_file, uid='uid', output='all')
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
    df_new_file, df_old_file = setup(df_new, df_old, tmpdir)

    result = qp.diff(df_new, df_old, output='print')
    expected = None

    #reading from in memory df
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'

    #reading from files
    result = qp.diff(df_new_file, df_old_file, uid='uid', output='print')
    assert result == expected, f'\nRESULT\n{result}\nEXPECTED:\n{expected}'



def test_csv(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_csv, df_old_csv = setup_csv(df_new, df_old, tmpdir)

    result_df = qp.diff(df_new, df_old, uid='uid', mode='new').data.astype('object')
    result_csv = qp.diff(df_new_csv, df_old_csv, uid='uid', mode='new').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'

    result_df = qp.diff(df_new, df_old, uid='uid', mode='new+').data.astype('object')
    result_csv = qp.diff(df_new_csv, df_old_csv, uid='uid', mode='new+').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "new+".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'

    result_df = qp.diff(df_new, df_old, uid='uid', mode='old').data.astype('object')
    result_csv = qp.diff(df_new_csv, df_old_csv, uid='uid', mode='old').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "old".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'

    result_df = qp.diff(df_new, df_old, uid='uid', mode='mix').data.astype('object')
    result_csv = qp.diff(df_new_csv, df_old_csv, uid='uid', mode='mix').data.astype('object')
    assert result_df.equals(result_csv), f'failed test for csv mode: "mix".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{result_df}\nRESULT:\n{result_csv}'




def test_identical(tmpdir):
    df_new, df_old = qp.get_dfs()
    df_new_xlsx, df_old_xlsx = setup(df_new, df_old, tmpdir)
    df_new_csv, df_old_csv = setup_csv(df_new, df_old, tmpdir)

    df1, summary1, str1 = qp.diff(df_new, df_new, uid='uid', mode='mix', output='all')
    dict2, summary2, str2 = qp.diff(df_new_xlsx, df_new_xlsx, uid='uid', mode='mix', output='all')
    df3, summary3, str3 = qp.diff(df_new_csv, df_new_csv, uid='uid', mode='mix', output='all')

    df1.index = df1['uid']
    df1['a'] = df1['a'].astype(float)
    df1 = df1.astype('object').fillna('')

    df2 = dict2['Sheet1'].data.drop('meta', axis=1).astype('object').fillna('')

    df3 = df3.astype('object').fillna('')
    df3.index = df3['uid']

    assert df1.equals(df2), f"DataFrames are not equal:\n{df1}\n{df2}"
    assert df1.equals(df3), f"DataFrames are not equal:\n{df1}\n{df3}"
    assert summary1 == summary3 == {}, f"Summaries are not equal:\n{summary1}\n{summary3}"
    assert str1 == str3 == 'both dataframes are identical', f"Strings are not equal:\n{str1}\n{str3}"
    assert str2 == 'no string version of differences available when comparing excel files'
