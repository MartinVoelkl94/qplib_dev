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


def test_returns_df():
    result = qp.diff(df_new, df_old, 'new', returns='df', verbosity=0).data

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


def test_returns_dict():
    result = qp.diff(df_new, df_old, returns='dict')
    expected = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 1,
        'vals changed': 1,
        }
    assert result == expected, f'\nRESULT:\n{result}\nEXPECTED:\n{expected}'


def test_returns_str():
    result = qp.diff(df_new, df_old, returns='str')
    expected = "only in df1:\nindices: ['x2']\nheaders: ['d']\nonly in df2:\nindices: ['x']\nheaders: ['c']\n"
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT:\n{result}\nEXPECTED:\n{expected.replace('\t','')}'


def test_returns_str_plus():
    result = qp.diff(df_new, df_old, returns='str+')
    expected = "only in df1:\nindices: ['x2']\nheaders: ['d']\nonly in df2:\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df1:\n    b    a\ny       0.0\nz  3.0     \n\ndifferent values in df2:\nb    a\ny    2.0\nz    3.0\n"
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT:\n{result}\nEXPECTED:\n{expected.replace('\t','')}'


def test_returns_all():
    result_df_styler, result_dict, result_str = qp.diff(df_new, df_old, returns='all')
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

    expected_dict = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 1,
        'vals changed': 1,
        }
    
    expected_str = "only in df1:\nindices: ['x2']\nheaders: ['d']\nonly in df2:\nindices: ['x']\nheaders: ['c']\n"

    assert result_df.equals(expected_df), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_df}\nRESULT:\n{result_df}'
    assert result_dict == expected_dict, f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_dict}\nRESULT:\n{result_dict}'
    assert result_str.replace(' ','') == expected_str.replace(' ',''), f'failed test for returns="all".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_str}\nRESULT:\n{result_str}'
    

def test_returns_all_plus():
    result_df_styler, result_dict, result_str = qp.diff(df_new, df_old, returns='all+')
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

    expected_dict = {
        'cols added': 1,
        'cols removed': 1,
        'rows added': 1,
        'rows removed': 1,
        'vals added': 1,
        'vals removed': 1,
        'vals changed': 1,
        }
    
    expected_str = "only in df1:\nindices: ['x2']\nheaders: ['d']\nonly in df2:\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df1:\n    b    a\ny       0.0\nz  3.0     \n\ndifferent values in df2:\nb    a\ny    2.0\nz    3.0\n"
    
    assert result_df.equals(expected_df), f'failed test for returns="all+": "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_df}\nRESULT:\n{result_df}'
    assert result_dict == expected_dict, f'failed test for returns="all+": "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_dict}\nRESULT:\n{result_dict}'
    assert result_str.replace(' ','') == expected_str.replace(' ',''), f'failed test for returns="all+": "new".\nold df:\n{df_old}\nnew df:{df_new}\nEXPECTED:\n{expected_str}\nRESULT:\n{result_str}'
    


def test_diff_str():
    result = qp.diff_str(df_new, df_old)
    expected = "only in df1:\nindices: ['x2']\nheaders: ['d']\nonly in df2:\nindices: ['x']\nheaders: ['c']\n"
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT:\n{result}\nEXPECTED:\n{expected.replace('\t','')}'


def test_diff_str_slow():
    result = qp.diff_str(df_new, df_old, fast=False)
    expected = "only in df1:\nindices: ['x2']\nheaders: ['d']\nonly in df2:\nindices: ['x']\nheaders: ['c']\n\ndifferent values in df1:\n    b    a\ny       0.0\nz  3.0     \n\ndifferent values in df2:\nb    a\ny    2.0\nz    3.0\n"
    assert result.replace(' ','') == expected.replace(' ',''), f'\nRESULT:\n{result}\nEXPECTED:\n{expected.replace('\t','')}'


