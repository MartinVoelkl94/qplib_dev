
import pytest
import pandas as pd
import numpy as np
import qplib as qp
from qplib import log


def get_df_simple_tagged():
    df = pd.DataFrame({
        'meta': ['', '', ''],
        'a': [-1, 0, 1],
        'b': [1, 2, 3]
        })
    df.index = [0, 1, 2]
    return df


def get_df():
    df = pd.DataFrame({
        'ID': [10001, 10002, 10003, 20001, 20002, 20003, 30001, 30002, 30003, 30004, 30005],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown', 'eva white', 'Frank miller', 'Grace TAYLOR', 'Harry Clark', 'IVY GREEN', 'JAck Williams', 'john Doe'],
        'date of birth': ['1995-01-02', '1990/09/14', '1985.08.23', '19800406', '05-11-2007', '06-30-1983', '28-05-1975', '1960Mar08', '1955-Jan-09', '1950 Sep 10', '1945 October 11'],
        'age': [-25, '30', np.nan, None, '40.0', 'forty-five', 'nan', 'unk', '', 'unknown', 35],
        'gender': ['M', 'F', 'Female', 'Male', 'Other', 'm', 'ff', 'NaN', None, 'Mal', 'female'],
        'height': [170, '175.5cm', None, '280', 'NaN', '185', '1', '6ft 1in', -10, '', 200],
        'weight': [70.2, '68', '72.5lb', 'na', '', '75kg', None, '80.3', '130lbs', '82', -65],
        'bp systole': ['20', 130, 'NaN', '140', '135mmhg', '125', 'NAN', '122', '', 130, '45'],
        'bp diastole': [80, '85', 'nan', '90mmHg', np.nan, '75', 'NaN', None, '95', '0', 'NaN'],
        'cholesterol': ['Normal', 'Highe', 'NaN', 'GOOD', 'n.a.', 'High', 'Normal', 'n/a', 'high', '', 'Normal'],
        'diabetes': ['No', 'yes', 'N/A', 'No', 'Y', 'Yes', 'NO', None, 'NaN', 'n', 'Yes'],
        'dose': ['10kg', 'NaN', '15 mg once a day', '20mg', '20 Mg', '25g', 'NaN', None, '30 MG', '35', '40ml']
        })
    return df


def check_message(expected_message):
    logs = qp.log()
    logs['text_full'] = logs['level'] + ': ' + logs['text']
    log_texts = logs['text_full'].to_list()
    assert expected_message in logs['text_full'].values, f'did not find expected message: {expected_message}\nin logs:\n{log_texts}'



def test_check_df():
    df = pd.DataFrame(
        0,
        columns=['a', 'a', ' a', 'a ', '$%&/', '='],
        index=[1, 1, 2]
        )
    log(clear=True, verbosity=1)
    df.check()
    check_message('ERROR: index is not unique')
    check_message('ERROR: cols are not unique')
    check_message('WARNING: the following colnames contain "%" which is used by the query syntax, use a tick (´) to escape such characters:<br>&emsp;[\'$%&/\']')
    check_message('WARNING: the following colnames contain "&" which is used by the query syntax, use a tick (´) to escape such characters:<br>&emsp;[\'$%&/\']')
    check_message('WARNING: the following colnames contain "/" which is used by the query syntax, use a tick (´) to escape such characters:<br>&emsp;[\'$%&/\']')
    check_message('WARNING: the following colnames contain "$" which is used by the query syntax, use a tick (´) to escape such characters:<br>&emsp;[\'$%&/\']')
    check_message("WARNING: the following colnames contain leading whitespace which should be removed:<br>&emsp;[' a']")
    check_message("WARNING: the following colnames contain trailing whitespace which should be removed:<br>&emsp;['a ']")
    check_message("WARNING: the following colnames start with a character sequence that can be read as a query instruction symbol when the default instruction operator is inferred:<br>['=']<br>explicitely use a valid operator to avoid conflicts.")



def test_col_eval():
    result = get_df().q(
        r"""
        id $rows col~ df["name"]
        is any;
        """)
    expected = get_df()
    expected['ID'] = expected['name']
    assert result.equals(expected), 'failed test0: copy col contents\n' + qp.diff(result, expected, output='str')

    result = get_df().q(
        r"""
        id $rows col~ df["name"]
        is any; %%is any;
        """)
    expected = get_df()
    expected['ID'] = expected['name']
    assert result.equals(expected), 'failed test1: copy col contents and select all\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        id / age $rows col~ df["name"]
        is any;
        """)
    expected = get_df()
    expected['ID'] = expected['name']
    expected['age'] = expected['name']
    assert result.equals(expected), 'failed test2: copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        $ col~ df["name"]
        """)
    expected = get_df()
    expected['ID'] = expected['ID'].astype('object')
    for col in expected.columns:
        expected[col] = expected['name']
    assert result.equals(expected), 'failed test3: copy col contents to all cols\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        id / age %%is num; $ col~ df["name"]
        is any;
        """)
    expected = get_df()
    expected['ID'] = expected['name']
    expected['age'] = expected['name']
    assert result.equals(expected), 'failed test4: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        id / age %%all is num; $ col~ df["name"]
        is any;
        """)
    expected = get_df().loc[[0,1,2,3,4,8,10], :]
    expected['ID'] = expected['name']
    expected['age'] = expected['name']
    assert result.equals(expected), 'failed test5: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        id / age %%%is num; $ col~ df["name"]
        is any;
        """)
    expected = get_df()
    expected['ID'] = expected['name']
    expected.loc[[0,1,2,3,4,8,10], 'age'] = expected.loc[[0,1,2,3,4,8,10], 'name']
    assert (result.loc[[5,6,7,9], 'age'] != df.loc[[5,6,7,9], 'name']).all(), 'failed test6: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')
    assert result.equals(expected), 'failed test7: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


def test_comment():
    result = get_df().q('#id')
    expected = get_df()
    assert result.equals(expected), qp.diff(result, expected, output='str')

    result = get_df().q('id #name')
    expected = get_df().loc[:, ['ID']]
    assert result.equals(expected), qp.diff(result, expected, output='str')

    result = get_df().q(
        r"""
        id
        #name
        """
        )
    expected = get_df().loc[:, ['ID']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_diff_mix():
    result = get_df().q(r'$new=a   $diff=mix').data
    result = result.reindex(sorted(result.columns), axis=1)
    expected = get_df()
    expected.insert(0, 'meta', '')
    expected.insert(1, 'new1', ['a']*11)
    expected = expected.reindex(sorted(expected.columns), axis=1)
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_diff_new():
    result = get_df().q(r'$new=1   $diff=new').data
    expected = pd.DataFrame({
        'meta': ['']*11,
        'new1': ['1']*11,
        })
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_diff_new_plus():
    df = pd.DataFrame({
        'a': [1,2,3],
        })
    result = df.q(r'$vals=b   $diff=new+').data
    expected = pd.DataFrame({
            'meta': ['<br>vals changed: 1', '<br>vals changed: 1', '<br>vals changed: 1'],
            'a': ['b', 'b', 'b'],
            'old: a': [1, 2, 3],
            })
    expected['old: a'] = expected['old: a'].astype('object')
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_diff_old():
    result = get_df().q(r'$new=1   $diff=old').data
    expected = get_df()
    expected.insert(0, 'meta', '')
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_diff_retain_meta():
    df = get_df()
    df.insert(0, 'meta', 'test')
    result = df.q(r'id $diff=new').data
    expected = df.loc[:, ['meta', 'ID']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_eval():
    result = get_df().q(
        r"""
        name $vals ~ x.lower()
        is any; %%is any;
        """)
    expected = get_df()
    expected['name'] = expected['name'].str.lower()
    assert result.equals(expected), 'failed test0: change to lowercase\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        name  %%!~ x == x.lower()  $ ~ x.lower()
        is any; %%is any;
        """)
    expected = get_df()
    expected['name'] = expected['name'].str.lower()
    assert result.equals(expected), 'failed test1: conditionally change to lowercase\n' + qp.diff(result, expected, output='str')

    result = get_df().q(
        r"""
        gender  %%is any; $to str;  $~ x.lower()
        is any;
        """)
    expected = get_df()
    expected['gender'] = expected['gender'].astype(str).str.lower()
    assert result.equals(expected), 'failed test2: convert and change to lowercase\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        id  %%10001  $ ~ str(10001)
        """)
    expected = get_df().loc[[0], ['ID']]
    expected.loc[0, 'ID'] = '10001'
    assert result.equals(expected), 'failed test3: convert single entry to str\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        id / age  %%is num;  $ ~ str(0)
        """)
    expected = get_df().loc[:, ['ID', 'age']]
    expected['ID'] = str(0)
    expected['age'] = str(0)
    assert result.equals(expected), 'failed test4: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        id / age  %%all is num;  $ ~ 0
        """)
    rows = [0,1,2,3,4,8,10]
    expected = get_df().loc[rows, ['ID', 'age']]
    expected.loc[rows, 'ID'] = 0
    expected.loc[rows, 'age'] = 0
    assert result.equals(expected), 'failed test5: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        id / age  %%%is num;  $ ~ 10
        """)
    expected = get_df().loc[:, ['ID', 'age']]
    expected['ID'] = 10
    expected.loc[[0,1,2,3,4,8,10], 'age'] = 10
    assert (result['ID'] == 10).all(), 'failed test6: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')
    assert (result.loc[[5,6,7,9], 'age'] != 10).all(), 'failed test7: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')
    assert result.equals(expected), 'failed test8: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')


    result = get_df().q(r'name  %%is any;  $~str(1)')
    expected = get_df().loc[:, ['name']]
    expected['name'] = '1'
    assert result.equals(expected), 'failed test9: convert all entries to python expression\n' + qp.diff(result, expected, output='str')



params = [   
    (r'id  $cols id',                                                                 get_df().rename(columns={'ID': 'id'}).loc[:, ['id']]),
    (r'id  $cols id   %name  $cols n   %date of birth  $cols dob  %is any;',      get_df().rename(columns={'ID': 'id', 'name': 'n', 'date of birth': 'dob'})),
    (r'id  $cols += abc',                                                             get_df().rename(columns={'ID': 'IDabc'}).loc[:, ['IDabc']]),
    (r'id / name / date of birth   $cols += abc  %is any;',                           get_df().rename(columns={'ID': 'IDabc', 'name': 'nameabc', 'date of birth': 'date of birthabc'})),
    (r'id  $cols ~ x.lower() + str(len(x))   %is any;',                               get_df().rename(columns={'ID': 'id2'})),
    (r'id / weight / diabetes    $cols ~ x.lower() + str(len(x))   %is any;',         get_df().rename(columns={'ID': 'id2', 'weight': 'weight6', 'diabetes': 'diabetes8'})),
    ]
@pytest.mark.parametrize('code, expected', params)
def test_cols(code, expected):
    result = get_df().q(code)
    assert result.equals(expected), 'failed test5: rename multiple col cols with eval\n' + qp.diff(result, expected, output='str')



def test_invert():

    df = get_df()

    result = df.q(r'id  %invert;')
    expected = df.loc[:,['name', 'date of birth', 'age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']]
    assert result.equals(expected), qp.diff(result, expected, output='str')

    result = df.q(r'name  /gender  %invert;')
    expected = df.loc[:,['ID', 'date of birth', 'age', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    
    result = df.q(r'name  %%?j  %%invert;')
    expected = df.loc[3:8, ['name']]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    
    result = df.q(r'name  %%%?j  %%%invert;')
    expected = df.loc[:, ['name']]
    assert result.equals(expected), qp.diff(result, expected, output='str')



def test_logging():
    
    df = get_df()

    #wip: does not work in older python versions
    # docstring = df.q.__str__()
    # expected = "docstring of dataframe accessor pd_object.q():\n\nA query language for pandas data exploration/analysis/modification.\ndf.qi() without any args can be used to interactively build a query in Jupyter notebooks.\n\n\nexamples:\n\n#select col\ndf.q('id')\ndf.q('%id')  #equivalent\ndf.q('%=id') #equivalent\ndf.q('%==id') #equivalent\ndf.q('% == id') #equivalent\n\n#select multiple cols\ndf.q('id  /name')\n\n#select rows in a col which fullfill a condition\ndf.q('id  %%>20000')\n\n#select rows fullfilling multiple conditions in the same col\ndf.q('id  %%>20000   &&<30000')\n\n#select rows fullfilling both conditions in different cols\ndf.q('id  %%>20000    %name   &&?john')\n\n#select rows fullfilling either condition in different cols\ndf.q('id   %%>20000   %name   //?john')\n\n"
    # assert docstring == expected, f'docstring does not match expected:\n{docstring}\n\nexpected:\n{expected}'

    result = df.q(r'$diff=mix  $color=red')
    check_message('WARNING: diff and style formatting are not compatible. formatting will be ignored')

    log(clear=True, verbosity=1)
    result = df.q(r'diabetes  %%is na;')
    result1 = df.q(r'diabetes  %%is na;0')
    expected = df.copy().loc[[2,7,8], ['diabetes']]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    assert result1.equals(expected), qp.diff(result1, expected, output='str')
    check_message('WARNING: value 0 will be ignored for unary operator <"is na;": IS_NA>')

    log(clear=True, verbosity=1)
    result = df.q(r'test  %%is na;')
    expected = pd.DataFrame(index=df.index)
    assert result.equals(expected), qp.diff(result, expected, output='str')
    check_message('WARNING: no cols fulfill the condition in "%test  "')
    check_message('ERROR: row selection cannot be applied when no cols where selected')

    log(clear=True, verbosity=1)
    result = df.q(r'id %%=@test')
    expected = df.copy().loc[:, ['ID']]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    check_message('ERROR: col "test" not found in dataframe. cannot use "@test" for row selection')

    log(clear=True, verbosity=1)
    with pytest.raises(AttributeError):
        result = df.q(r'%%load1')
    check_message('ERROR: selection "1" is not in saved selections')

    log(clear=True, verbosity=1)
    result = df.q(r'$verbosity=6')
    check_message('WARNING: verbosity must be an integer between 0 and 5. "6" is not valid')

    log(clear=True, verbosity=1)
    qp.qlang.VERBOSITY = 0
    result = df.q(r'$verbosity=0')
    qp.qlang.VERBOSITY = 3
    assert len(log()) == 0, 'log should be empty when verbosity is set to 0'

    log(clear=True, verbosity=1)
    result = df.q(r'$diff=old+')
    check_message('WARNING: diff must be one of [None, mix, old, new, new+]. "old+" is not valid')

    result1 = df.q(r'$diff=None')
    result2 = df.q(r'$diff=0')
    result3 = df.q(r'$diff=false')
    assert isinstance(result1, pd.DataFrame), f'setting diff to None should return a dataframe, not {type(result1)}'
    assert result1.equals(result2), qp.diff(result1, result2, output='str')
    assert result1.equals(result3), qp.diff(result1, result3, output='str')
    
    log(clear=True, verbosity=1)
    result = df.q(r'test  $cols=abc')
    check_message('ERROR: colname modification cannot be applied when no cols where selected')
    
    log(clear=True, verbosity=1)
    result = df.q(r'$new=@test')
    check_message('ERROR: col "test" not found in dataframe. cannot add a new col thats a copy of it')




params = [
    (r'a    %%>0            $meta = >0  %is any;        %%is any;',                             ['', '', '>0']),
    (r'a    %%>0            $meta += >0  %is any;       %%is any;',                             ['', '', '>0']),
    (r'=a   %%>0            $meta +=>0  %is any;        %%is any;',                             [ '', '', '>0']),
    (r'=a   %%>0            $meta+= >0  %is any;        %%is any;',                             [ '', '', '>0']),
    (r'=a   %%>0            $meta ~ x + ">0"            %is any;            %%is any;',         [ '', '', '>0']),
    (r'=a   %%>0            $meta ~ ">" + str(0)        %is any;            %%is any;',         [ '', '', '>0']),
    (r'=a   %%>0            $meta col~ col + ">0"       %is any;            %%is any;',         [ '', '', '>0']),
    (r'=a   %%>0            $tag   %is any;             %%is any;',                             [ '', '', '\n@a: ']),
    (r'=a   %%>0    /b      $tag  %is any;              %%is any;',                             [ '', '', '\n@a@b: ']),
    (r'=a   %%>0    /b      $tag=value >0               %is any;            %%is any;',         [ '', '', '\n@a@b: value >0']),
    ]
@pytest.mark.parametrize('code, metadata', params)
def test_metadata(code, metadata):
    result = get_df_simple_tagged().q(code)
    expected = get_df_simple_tagged()
    expected['meta'] = metadata
    assert result.equals(expected), 'failed test: metadata tagging\n' + qp.diff(result, expected, output='str')


def test_metadata_init():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        })
    result = df.q(r'$meta=a   %is any;')
    expected = df.copy()
    expected['meta'] = ['a', 'a', 'a']
    assert result.equals(expected), 'failed test: metadata init\n' + qp.diff(result, expected, output='str')


def test_metadata_continous():

    expected = get_df_simple_tagged()
    result = get_df_simple_tagged()

    result = result.q(r'=a  %%>0   $meta +=>0   %is any;  %%is any;')
    expected['meta'] = [ '', '', '>0']
    assert result.equals(expected), 'failed test0: continous metadata tagging\n' + qp.diff(result, expected, output='str')

    result = result.q(r'=a   %%>0  $meta+=>0   %is any;  %%is any;')
    expected['meta'] = [ '', '', '>0>0']
    assert result.equals(expected), 'failed test1: continous metadata tagging\n' + qp.diff(result, expected, output='str')

    result = result.q(r'=a   %%==0    $meta += 0   %is any;  %%is any;')
    expected['meta'] = [ '', '0', '>0>0']
    assert result.equals(expected), 'failed test2: continous metadata tagging\n' + qp.diff(result, expected, output='str')

    result = result.q(r'a   %%==0    $meta ~ x.replace("0", "")   %is any;  %%is any;')
    expected['meta'] = [ '', '', '>0>0']
    assert result.equals(expected), 'failed test3: continous metadata tagging\n' + qp.diff(result, expected, output='str')

    result = result.q(r'=a   %%>0    $meta~x.replace("0", "")   %is any;  %%is any;')
    expected['meta'] = [ '', '', '>>']
    assert result.equals(expected), 'failed test4: continous metadata tagging\n' + qp.diff(result, expected, output='str')

    result = result.q(r'=a     $meta=   %is any;  %%is any;')
    expected['meta'] = [ '', '', '']
    assert result.equals(expected), 'failed test5: continous metadata tagging\n' + qp.diff(result, expected, output='str')



def test_modify_rows_vals():
    
    result = get_df().q(
        r"""
        name  $vals a
        is any;  %%is any;
        """)
    expected = get_df()
    expected['name'] = 'a'
    assert result.equals(expected), 'failed test0: set values\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        name %%is any; $ =a
        is any; %%is any;
        """)
    expected = get_df()
    expected['name'] = 'a'
    assert result.equals(expected), 'failed test1: set values\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        name  %%%is any; $ a
        gender $rows b
        is any; %%is any;
        """)
    expected = get_df()
    expected['name'] = 'a'
    expected['gender'] = 'b'
    assert result.equals(expected), 'failed test2: set values in multiple cols\n' + qp.diff(result, expected, output='str')


    result = df.q(
        r"""
        name  $rows=a   %%%is any;  $+=a
        is any; %%is any;
        """)
    expected = get_df()
    expected['name'] = 'aa'
    assert result.equals(expected), 'failed test3: appending to values\n' + qp.diff(result, expected, output='str')


    result = df.q(r'name  %%%?j  $=@ID  %is any;')
    expected = get_df()
    expected['name'] = [
        10001,
        10002,
        10003,
        'Bob Brown',
        'eva white',
        'Frank miller',
        'Grace TAYLOR',
        'Harry Clark',
        'IVY GREEN',
        30004,
        30005,
        ]
    assert result.equals(expected), 'failed test: replace values with column content\n' + qp.diff(result, expected, output='str')


    result = df.q(r'name  %%%?j  $+=@ID  %is any;')
    expected = get_df()
    expected['name'] = [
        'John Doe10001',
        'Jane Smith10002',
        'Alice Johnson10003',
        'Bob Brown',
        'eva white',
        'Frank miller',
        'Grace TAYLOR',
        'Harry Clark',
        'IVY GREEN',
        'JAck Williams30004',
        'john Doe30005',
        ]
    assert result.equals(expected), 'failed test: add column content to selected values\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        $new=___ERROR  $cols=error code
            %%%is any;  $vals+=@ID

        %age  /gender
            %%idx>5  &&idx<=8
                %%%is na;
                    $vals+=@error code
                    $bg=orange

        is any;  %%is any;
        """
        ).data
    expected = get_df()
    expected['error code'] = [
        '___ERROR10001',
        '___ERROR10002',
        '___ERROR10003',
        '___ERROR20001',
        '___ERROR20002',
        '___ERROR20003',
        '___ERROR30001',
        '___ERROR30002',
        '___ERROR30003',
        '___ERROR30004',
        '___ERROR30005',
        ]
    expected.loc[6, 'age'] = 'nan___ERROR30001'
    expected.loc[8, 'age'] = '___ERROR30003'
    expected.loc[7, 'gender'] = 'NaN___ERROR30002'
    expected.loc[8, 'gender'] = 'None___ERROR30003'
    assert result.equals(expected), qp.diff(result, expected, output='str')





params = [
    (r'$new',               '',                     ['new1']),
    (r'$new a',             'a',                    ['new1']),
    (r'$newa',              'a',                    ['new1']),
    (r'$new =a',            'a',                    ['new1']),
    (r'$new= a',            'a',                    ['new1']),
    (r'$new = a',           'a',                    ['new1']),
    (r'$new ~ "a"',         'a',                    ['new1']),
    (r'$new ~ df["ID"]',    get_df()['ID'],         ['new1']),
    (r'$new @ID',           get_df()['ID'],         ['new1']),
    (r'$new  %id',          '',                     ['ID']),
    ]
@pytest.mark.parametrize('code, content, cols', params)
def test_new_col(code, content, cols):
    result = get_df().q(code)
    expected = get_df()
    expected['new1'] = content
    expected = expected.loc[:, cols]
    assert result.equals(expected), 'failed test0: creating new col\n' + qp.diff(result, expected, output='str')


def test_new_col1():
    result = get_df().q('$new a  $cols = new col')
    expected = get_df()
    expected['new col'] = 'a'
    expected = expected.loc[:,['new col']]
    assert result.equals(expected), 'failed test1: creating new col with colname\n' + qp.diff(result, expected, output='str')


    result = get_df().q('$new a   &newb   /=new1 /=new2')
    expected = get_df()
    expected['new1'] = 'a'
    expected = expected.loc[:,['new1']]
    assert result.equals(expected), 'failed test2: creating new col\n' + qp.diff(result, expected, output='str')
    check_message('WARNING: no cols fulfill the condition in "/=new2"')


    result = get_df().q('$new a /new b   /=new1 /=new2')
    expected = get_df()
    expected['new1'] = 'a'
    expected = expected.loc[:,['new1']]
    assert result.equals(expected), 'failed test3: creating new col\n' + qp.diff(result, expected, output='str')
    check_message('WARNING: no cols fulfill the condition in "/=new2"')


    result = get_df().q('$new a  $new b   /=new1 /=new2')
    expected = get_df()
    expected['new1'] = 'a'
    expected['new2'] = 'b'
    expected = expected.loc[:,['new1', 'new2']]
    assert result.equals(expected), 'failed test4: creating multiple new cols\n' + qp.diff(result, expected, output='str')


    result = get_df().q(r'%%idx = 0  $new a')
    expected = get_df()
    expected['new1'] = 'a'
    expected = expected.loc[[0],['new1']]
    assert result.equals(expected), 'failed test5: creating new col with index\n' + qp.diff(result, expected, output='str')


    result = get_df().q(r'$new a  %%idx = 0  ')
    expected = get_df()
    expected['new1'] = 'a'
    expected = expected.loc[[0],['new1']]
    assert result.equals(expected), 'failed test6: creating new col and select with index\n' + qp.diff(result, expected, output='str')


    result = get_df().q(r'%%idx = 0   $save1  %%is any;   $new a  %%load1')
    expected = get_df()
    expected['new1'] = 'a'
    expected = expected.loc[[0],['new1']]
    assert result.equals(expected), 'failed test7: check if selection saving and loading works correctly with new col creation\n' + qp.diff(result, expected, output='str')



def test_previous_bugs():
    try:
        df.q(
            r"""
            %age ///!is num;
            is any;
            $bg=orange
            """
            )
    except Exception as e:
        assert False, f'failed test0: check if previous bug is fixed: {e}'
    
    #from v0.7.5
    result = get_df().q(r'%%%>0    &&&<50  %trim;  $bg=orange').data
    expected = get_df().loc[:, ['age', 'height', 'bp systole', 'dose']]
    assert result.equals(expected), qp.diff(result, expected, output='str')

    
    


df = get_df()
params = [
    (r'id %%?1      $save1      %%is any;   %%load1',                           df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%?1      $save=1     %%is any;   %%load=1',                          df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%any?1   $save1      %%is any;   %%load1',                           df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%all?1   $save1      %%is any;   %%load1',                           df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%?1      $save1      %is any;    %%is any;   %id         %%load1',   df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%?2      $save1      %%?1        $save1      %%is any;   %%load1',   df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%%?1     $save1      %%is any;   %%load1',                           df.loc[:,['ID']], None),
    
    (
        r"""
        id   %%?1   $save1
        id   %%?2   $save2
        id   %%load1   &&load2
        """,
        df.loc[[1,3],['ID']],
        None
    ),

    (
        r"""
        id   %%?1   $save1
        id   %%?2   $save2
        id   %%load1   //load2
        """,
        df.loc[[0,1,2,3,4,5,6,7],['ID']],
        None
    ),
    
    (
        r"""
        id   %%!?1   $save1
        id   %%?2   $save2
        id   %%load1   &&load2
        """,
        df.loc[[4,5,7],['ID']],
        None
    ),
    
    (
        r"""
        id   %%!?1      $save1
        id   %%!?2      $save2
        id   %%load1    &&load2
        """,
        df.loc[[8,9,10],['ID']],
        None
    ),
    ]
@pytest.mark.parametrize('code, expected, message', params)
def test_save_load(code, expected, message):
    result = get_df().q(code)
    assert result.equals(expected), qp.diff(result, expected, output='str')
    if message: #pragma: no cover
        check_message(message)




df = get_df()
params = [

    #using operator: set/equals
    ('', df.columns, None),
    ('ID', ['ID'], None),
    (' ID', ['ID'], None),
    ('ID ', ['ID'], None),
    (' ID ', ['ID'], None),

    ('=ID', ['ID'], None),
    (' =ID', ['ID'], None),
    ('= ID', ['ID'], None),
    ('= ID ', ['ID'], None),
    (' = ID ', ['ID'], None),

    ('==ID', ['ID'], None),
    (' ==ID', ['ID'], None),
    ('== ID', ['ID'], None),
    ('== ID ', ['ID'], None),
    (' == ID ', ['ID'], None),

    ('date of birth', ['date of birth'], None),
    ('date of birth / age', ['date of birth', 'age'], None),
    ('=date of birth / age', ['date of birth', 'age'], None),
    ('=date of birth / =age', ['date of birth', 'age'], None),
    ('date of birth / =age', ['date of birth', 'age'], None),
    ('!=date of birth', ['ID', 'name', 'age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose'], None),
    ("""ID""", ['ID'], None),
    (
        """ID
        """,
        ['ID']
    , None),
    (
        """
        ID""",
        ['ID']
    , None),
    (
        """
        ID
        """,
        ['ID']
    , None),
    (r"""ID""", ['ID'], None),
    (
        r"""ID
        """,
        ['ID']
    , None),
    (
        r"""
        ID""",
        ['ID']
    , None),
    (
        r"""
        ID
        """,
        ['ID']
    , None),


    #negation flag
    ('!=date of birth', ['ID', 'name', 'age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose'], None),
    ('!==date of birth', ['ID', 'name', 'age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose'], None),
   
    #strictness flag
    ('strict=', [], 'WARNING: no cols fulfill the condition in "%strict="'),
    ('strict=ID', ['ID'], None),
    ('strict =ID', ['ID'], None),
    ('strict= ID', ['ID'], None),
    ('strict = ID', ['ID'], None),
    ('strict=id', [], 'WARNING: no cols fulfill the condition in "%strict=id"'),
    ('strict=date of birth / age', ['date of birth', 'age'], None),
    ('strict=date of birth / =age', ['date of birth', 'age'], None),

    #using operator: contains
    ('?bp', ['bp systole', 'bp diastole'], None),
    ('?I', ['ID', 'date of birth', 'height', 'weight', 'bp diastole', 'diabetes'], None),
    ('!?I', ['name', 'age', 'gender', 'bp systole', 'cholesterol', 'dose'], None),

    #using operator: strict contains
    ('strict?I', ['ID'], None),


    #using operator: regex equality
    ('regex=ID´$', ['ID'], None),
    ('regex=ID', ['ID'], None),
    ('regex=.', [], r'WARNING: no cols fulfill the condition in "%regex=."'),
    ('regex=..', ['ID'], None),

    #using operator: regex strict equality
    ('regex? ID', ['ID'], None),
    ('regex? e.', ['date of birth', 'gender', 'height', 'weight', 'cholesterol', 'diabetes'], None),


    #using multiple conditions
    ('?bp / =diabetes', ['bp systole', 'bp diastole', 'diabetes'], None),
    ('?bp / =diabetes /= cholesterol', ['bp systole', 'bp diastole', 'cholesterol', 'diabetes'], None),
    ('?bp /=cholesterol/= diabetes', ['bp systole', 'bp diastole', 'cholesterol', 'diabetes'], None),
    ('?bp & ?systole', ['bp systole'], None),
    ('?bp & !?systole', ['bp diastole'], None),
    ('?bp & !?systole & ?diastole', ['bp diastole'], None),
    ('?bp & !?systole / ?ID', ['ID', 'bp diastole'], None),

    ]
@pytest.mark.parametrize('code, expected_cols, message', params)
def test_select_cols(code, expected_cols, message):
    result = df.q(code)
    expected = df.loc[:, expected_cols]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    if message:
        check_message(message)




df = get_df()
params = [

    #numeric comparison
    (r'age  %%=30',             df.loc[[1], ['age']], None),
    (r'age  %%==30.0',          df.loc[[1], ['age']], None),
    (r'age  %%strict=30.0',     df.loc[[1], ['age']], None),
    (r'age  %%strict==30.0',    df.loc[[1], ['age']], None),
    (r'age  %%>30',             df.loc[[4,10], ['age']], None),
    (r'age  %%>=30',            df.loc[[1,4,10], ['age']], None),
    (r'age  %%<30',             df.loc[[0], ['age']], None),
    (r'age  %%<=30',            df.loc[[0,1], ['age']], None),
    (r'age  %%!=30',            df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%!==30.0',         df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%strict!=30',      df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%strict!==30.0',   df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%!strict=30',      df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%!strict==30.0',   df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%!>30',            df.loc[[0,1,2,3,5,6,7,8,9], ['age']], None),
    (r'age  %%!>=30',           df.loc[[0,2,3,5,6,7,8,9], ['age']], None),
    (r'age  %%!<30',            df.loc[[1,2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%!<=30',           df.loc[[2,3,4,5,6,7,8,9,10], ['age']], None),
    (r'age  %%=40',             df.loc[[4], ['age']], None),
    (r'age  %%==40',            df.loc[[4], ['age']], None),
    (r'age  %%=40.0',           df.loc[[4], ['age']], None),
    (r'age  %%==40.0',          df.loc[[4], ['age']], None),
    (r'age  %%strict=40',       df.loc[[4], ['age']], None),
    (r'age  %%strict==40',      df.loc[[4], ['age']], None),
    (r'age  %%strict=40.0',     df.loc[[4], ['age']], None),
    (r'age  %%strict==40.0',    df.loc[[4], ['age']], None),
    
    #date comparison
    (r'date of birth  %%=1995-01-02',                                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==1995-01-02',                                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995.01.02',                                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995_01_02',                                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995´/01´/02',                                             df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==1995 01 02',                                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995-Jan-02',                                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02-01-1995',                                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02-Jan-1995',                                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==Jan-02-1995',                                             df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02-01.1995',                                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02 Jan-1995',                                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==Jan´/02_1995',                                            df.loc[[0], ['date of birth']], None),
    (r'date of birth  $rows to datetime;        %%=05-11-2007',                     df.loc[[4], ['date of birth']], None),
    (r'date of birth  $vals to datetime;        %%>1990-01-01',                     df.loc[[0,1,4], ['date of birth']], None),
    (r'date of birth  %%is any;  $to datetime;  %%>1990-01-01  &&<2000-01-01',      df.loc[[0,1], ['date of birth']], None),


    #using type operators
    (r'name  %%is str;',                 df.loc[:, ['name']], None),
    (r'name  %%!is str;',                df.loc[[], ['name']], None),
    (r'name  %%is num;',                 df.loc[[], ['name']], None),
    (r'name  %%!is num;',                df.loc[:, ['name']], None),
    (r'name  %%is na;',                  df.loc[[], ['name']], None),
    (r'name  %%!is na;',                 df.loc[:, ['name']], None),

    (r'age   %%is int;',                 df.loc[[0,1,4,10], ['age']], None),
    (r'age   %%strict is int;',          df.loc[[0,10], ['age']], None),
    (r'age   %%is float;',               df.loc[[0,1,2,4,6,10], ['age']], None),
    (r'age   %%strict is float;',        df.loc[[2], ['age']], None),
    (r'age   %%is na;',                  df.loc[[2,3,6,8], ['age']], None),

    (r'weight  %%is int;',               df.loc[[1,9,10], ['weight']], None),
    (r'weight  %%strict is int;',        df.loc[[10], ['weight']], None),
    (r'weight  %%is float;',             df.loc[[0,1,7,9,10], ['weight']], None),
    (r'weight  %%strict is float;',      df.loc[[0], ['weight']], None),
    (r'weight  %%is num;',               df.loc[[0,1,4,6,7,9,10], ['weight']], None),
    (r'weight  %%strict is num;',        df.loc[[0,10], ['weight']], None),
    (r'weight  %%is num;  &&!is na;',    df.loc[[0,1,7,9,10], ['weight']], None),

    (r'height       %%is bool;',         df.loc[[6], ['height']], None),
    (r'bp diastole  %%is bool;',         df.loc[[9], ['bp diastole']], None),
    (r'diabetes     %%is bool;',         df.loc[[0,1,3,4,5,6,9,10], ['diabetes']], None),
    (r'diabetes     %%strict is bool;',  df.loc[[], ['diabetes']], None),

    (r'date of birth  %%is date;',       df.loc[:, ['date of birth']], None),
    (r'date of birth  %%is datetime;',   df.loc[:, ['date of birth']], None),

    (r'diabetes  %%is yn;',              df.loc[[0,1,3,4,5,6,9,10], ['diabetes']], None),
    (r'diabetes  %%is na;  //is yn;',    df.loc[:, ['diabetes']], None),
    (r'diabetes  %%is yes;',             df.loc[[1,4,5,10], ['diabetes']], None),
    (r'diabetes  %%is no;',              df.loc[[0,3,6,9], ['diabetes']], None),

    (r'cholesterol  %%is na;',           df.loc[[2,4,7,9], ['cholesterol']], None),
    (r'age          %%is na;',           df.loc[[2,3,6,8], ['age']], None),
    (r'age          %%strict is na;',    df.loc[[2,3], ['age']], None),

    (r'age  %%is nk;',              df.loc[[7,9], ['age']], None),


    #using regex equality
    (r'ID %%regex=1....',                                df.loc[[0,1,2], ['ID']], None),
    (r'ID %%regex!=3....',                               df.loc[[0,1,2,3,4,5], ['ID']], None),
    (r'ID %%!regex=3....',                               df.loc[[0,1,2,3,4,5], ['ID']], None),
    (r'name %%regex=\b[A-Z][a-z]*\s[A-Z][a-z]*\b',    df.loc[[0,1,2,3,7], ['name']], None), #two words with first letter capitalized and separated by a space
    (r'name %%regex=^[^A-Z]*´$',                          df.loc[[4], ['name']], None), #all lowercase
    (r'dose %%regex=^(?=.*[a-zA-Z])(?=.*[0-9]).*´$',      df.loc[[0,2,3,4,5,8,10], ['dose']], None), #containing letters and numbers


    #using regex search
    (r'bp systole %%regex?m', df.loc[[4], ['bp systole']], None),
    (r'bp systole %%regex?\D', df.loc[[2,4,6], ['bp systole']], None),
    (r'bp systole %%regex?\d', df.loc[[0,1,3,4,5,7,9,10], ['bp systole']], None),


     #using index
    (r'%%idx = 3',                                   df.iloc[[3], :], None),
    (r'%%idx > 5',                                   df.iloc[6:, :], None),
    (r'%%idx < 5',                                   df.iloc[:5, :], None),
    (r'%%idx >= 5',                                  df.iloc[5:, :], None),
    (r'%%idx <= 5',                                  df.iloc[:6, :], None),
    (r'%%idx != 5',                                  df.iloc[[0,1,2,3,4,6,7,8,9,10], :], None),
    (r'%%idx == 5',                                  df.iloc[[5], :], None),
    (r'%%idx >5  &&idx <8',                          df.iloc[6:8, :], None),
    (r'%%idx >5  &&idx <8  &&idx != 6',              df.iloc[[7], :], None),
    (r'%%idx >5  &&idx <8  &&idx != 6  &&idx != 7',  df.iloc[[], :], None),
    (r'%%idx ~ len(str(x)) > 1',                     df.iloc[[10], :], None),
    (r'%%idx ?1',                                    df.iloc[[1, 10], :], None),


    #comparison between cols
    (r'id  %%=@ID',                                         df.loc[:, ['ID']], None),
    (r'age / height  $rows to num;   %height  %%>@age',     df.loc[[0, 10], ['height']], None),
    (r'age / height  $vals to num;   %height  %%<@age',     df.loc[[], ['height']], None),
    (r'cholesterol   %%=@bp systole',                       df.loc[[2], ['cholesterol']], None),


    #apply row filter condition on multiple cols
    (r'id / name  %%?j',                df.loc[[0,1,2,9,10], ['ID', 'name']], None),
    (r'id / name  %%?j  //?n',          df.loc[[0,1,2,3,5,8,9,10], ['ID', 'name']], None),
    (r'id / name  %%?j  &&?n',          df.loc[[0,1,2,10], ['ID', 'name']], None),
    (r'height / weight   %%is num;',     df.loc[:, ['height', 'weight']], None),
    (r'height / weight   %%any is num;', df.loc[:, ['height', 'weight']], None),
    (r'height / weight   %%anyis num;',  df.loc[:, ['height', 'weight']], None),
    (r'height / weight   %%all is num;', df.loc[[0,6,9,10], ['height', 'weight']], None),


    #using uniqueness operators
    (
        r"""
        diabetes  %%is unique;
        is any;
        """,
        df.loc[[1,2,4,6,7,8,9], :],
        None
    ),
    (
        r"""
        diabetes  %%is first;
        is any;
        """,
        df.loc[[0,1,2,4,5,6,7,8,9], :],
        None
    ),
    (
        r"""
        diabetes  %%is last;
        is any;
        """,
        df.loc[[1,2,3,4,6,7,8,9,10], :],
        None
    ),


    #by evaluating python expressions
    (r'age  %%~ isinstance(x, int)', df.loc[[0, 10], ['age']], None),
    (
        r"""
        age / height  %%is any;  $to num;
        age  %%col~ col < df["height"]
        """,
        df.loc[[0, 10], ['age']],
        None
    ),
    (
        r"""
        age / height  %%%is any;  $to num;
        age  %%col~ col == df["age"].max()
        age  $valsto str;
        """,
        df.loc[[4], ['age']],
        None
    ),


    #combining multiple instructions and conditions
    (
        r"""
        ID  %%r=1....
        diabetes  %%is yes;
        """,
        df.loc[[1, 4, 5, 10], ['diabetes']],
        None
    ),
    (
        r"""
        ID  %%regex=1....
        diabetes  &&is yes;
        ID / diabetes
        """,
        df.loc[[1], ['ID', 'diabetes']],
        None
    ),
    (
        r"""
        diabetes  %%is yes;
        ID  &&regex=1....
        / diabetes
        """,
        df.loc[[1], ['ID', 'diabetes']],
        None
    ),
    (
        r"""
        diabetes %%is yes;
        / ID &&regex=1....
        """,
        df.loc[[1], ['ID', 'diabetes']],
        None
    ),
    (
        r"""
        ID  %%regex=1....  //regex=2....  $save1
        gender  %%=m  //=male  &&load1
        ID / gender
        """,
        df.loc[[0,3,5], ['ID', 'gender']],
        None
    ),
    (
        r"""
        ID  %%regex=1.... // regex=2....  $save1
        gender %%=m // =male  &&load1
        """,
        df.loc[[0,3,5], ['gender']],
        None
    ), 
    (
        r"""
        ID  %%regex=1.... // regex=2....  $save1
        gender %%=m // =m // =male  // load 1
        """,
        df.loc[[0,1,2,3,4,5], ['gender']],
        None
    ),
    (
        r"""
        ID  %%regex=1.... // regex=2....   $save1
        gender %%=f // =f // =female  // load 1
        ID
        """,
        df.loc[[0,1,2,3,4,5,10], ['ID']],
        None
    ),
    (
        r"""
        gender  %%=f // =female
        age  &&>30
        """,
        df.loc[[10], ['age']],
        None
    ),
    (
        r"""
        gender  %%=f  //=female  $save a
        age  %%>30  && load a 
        """,
        df.loc[[10], ['age']],
        None
    ),
    (
        r"""
        age  %%>30
        age  // <18
        """,
        df.loc[[0,4,10], ['age']],
        None
    ),
    (
        r"""
        age  %%>30  $save=a
        age  %%<18  //load=a
        """,
        df.loc[[0,4,10], ['age']],
        None
    ),
    (
        r"""
        age  %%>30   //<18
        """,
        df.loc[[0,4,10], ['age']],
        None
    ),
    (
        r"""
        weight  %%<70  &&>40  $save=between 40 and 70
        diabetes %%is yes;  &&load=between 40 and 70   
        """,
        df.loc[[1], ['diabetes']],
        None
    ),
    (
        r"""
        weight  %%<70  $save1
        &weight  %%>40  $save2
        diabetes  %%is no;  &&load1
        weight  / diabetes
        """,
        df.loc[[], ['weight', 'diabetes']],
        None
    ),

    ]
@pytest.mark.parametrize('code, expected, message', params)
def test_select_rows(code, expected, message):
    temp = get_df().q(code)
    result = get_df().loc[temp.index, temp.columns]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    if message: #pragma: no cover
        check_message(message)


def test_select_rows_scopes():
    result = get_df().q(r'%%is na; $')
    expected = get_df().loc[[1,2,3,4,6,7,8,9,10], :]
    expected.loc[:, :] = ''
    assert result.equals(expected), 'failed test0: check if default selection scope is "any"\n' + qp.diff(result, expected, output='str')
    

    result = get_df().q(r'%%any is na; $')
    expected = get_df().loc[[1,2,3,4,6,7,8,9,10], :]
    expected.loc[:, :] = ''
    assert result.equals(expected), 'failed test1: check selection scope "any"\n' + qp.diff(result, expected, output='str')


    result = get_df().q(r'%%all is na; $')
    expected = get_df().loc[[], :]
    assert result.equals(expected), 'failed test2: check selection scope "all"\n' + qp.diff(result, expected, output='str')
    check_message('WARNING: modification cannot be applied when no values where selected')



def test_select_vals():
    result = get_df().q(r'%%%is na; $')
    expected = get_df()
    expected.loc[[2,3,6], 'age'] = ''
    expected.loc[[7,8], 'gender'] = ''
    expected.loc[[2,4], 'height'] = ''
    expected.loc[[3,6], 'weight'] = ''
    expected.loc[[2,6], 'bp systole'] = ''
    expected.loc[[2,4,6,7,10], 'bp diastole'] = ''
    expected.loc[[2,4,7], 'cholesterol'] = ''
    expected.loc[[2,7,8], 'diabetes'] = ''
    expected.loc[[1,6,7], 'dose'] = ''
    assert result.equals(expected), 'failed test0: check val selection\n' + qp.diff(result, expected, output='str')


    result = get_df().q(
        r"""
        %%%is na;
        %age  /height  /weight  /?bp
            ///!is num;
            ///<0

        gender
            %%male  //m  //female  //f  //other
            %%invert;  ///is any;

        cholesterol
            %%normal  //high  //low  //good  //bad
            %%invert;  ///is any;

        is any;  %%is any;
        $vals=FOUND
        """
        )
    expected = get_df()
    expected.loc[[0,2,3,5,6,7,8,9], 'age'] = 'FOUND'
    expected.loc[[6,7,8,9], 'gender'] = 'FOUND'
    expected.loc[[1,2,4,7,8,9], 'height'] = 'FOUND'
    expected.loc[[2,3,4,5,6,8,10], 'weight'] = 'FOUND'
    expected.loc[[2,4,6,8], 'bp systole'] = 'FOUND'
    expected.loc[[2,3,4,6,7,10], 'bp diastole'] = 'FOUND'
    expected.loc[[1,2,4,7,9], 'cholesterol'] = 'FOUND'
    expected.loc[[2,7,8], 'diabetes'] = 'FOUND'
    expected.loc[[1,6,7], 'dose'] = 'FOUND'
    assert result.equals(expected), qp.diff(result, expected, output='str')



params = [
    (
        r"""
        id  $sort;   %is any;
        """,
        'ID'
    ),
    (
        r"""
        name  $sort;   %is any;
        """,
        'name'
    ),
    (
        r"""
        id /name  $sort;   %is any;
        """,
        ['ID', 'name']
    ),
    (
        r"""
        name /id  $sort;   %is any;
        """,
        ['ID', 'name']
    ),

    (
        r"""
        id  $!sort;   %is any;
        """,
        'ID'
    ),
    (
        r"""
        name  $!sort;   %is any;
        """,
        'name'
    ),
    (
        r"""
        id /name  $!sort;   %is any;
        """,
        ['ID', 'name']
    ),
    (
        r"""
        name /id  $!sort;   %is any;
        """,
        ['ID', 'name']
    ),
    ]
@pytest.mark.parametrize('code, expected_cols', params)
def test_sort(code, expected_cols):
    result = get_df().q(code)
    if '$!sort;' in code:
        expected = get_df().sort_values(by=expected_cols, ascending=False)
    else:
        expected = get_df().sort_values(by=expected_cols)
    assert result.equals(expected), 'failed test: sort values\n' + qp.diff(result, expected, output='str')


def test_style():
    result = get_df().q(r'$color=red')
    assert isinstance(result, pd.io.formats.style.Styler)

    #updating the style
    result = df.q(
        r"""
        $color=red
        $new=a
        $color=blue
        """
        )
    isinstance(result, pd.io.formats.style.Styler)
    expected = pd.DataFrame('a', index=df.index, columns=['new1'])
    assert result.data.equals(expected), 'failed test: updating style\n' + qp.diff(result.data, expected, output='str')


def test_symbols():
    sym1 = qp.qlang.OPERATORS['=']
    sym1a = qp.qlang.OPERATORS.SET
    sym1b = qp.qlang.OPERATORS['SET']
    sym2 = qp.qlang.OPERATORS.TRIM
    assert sym1 == sym1a, f'symbol {sym1} should be equal to {sym1a}'
    assert sym1 == sym1b, f'symbol {sym1} should be equal to {sym1b}'
    assert sym1 != sym2, f'symbol {sym1} should not be equal to {sym2}'
    assert sym1 < sym2, f'symbol {sym1} should be less than {sym2}'

    details = 'symbol:\n\tname: SET\n\tsymbol: =\n\tdescription: set values\n\ttraits:\n\t\tselect\n\t\tselect_vals\n\t\tselect_rows\n\t\tselect_cols\n\t\tmodify\n\t\tsettings\n\t\tmetadata\n\t'
    assert sym1.details() == details, f'symbol {sym1} should have details:\n{details}'

    symbols_modify1 = qp.qlang.OPERATORS.modify
    symbols_modify2 = qp.qlang.OPERATORS['modify']
    assert symbols_modify1 == symbols_modify2, f'symbols {symbols_modify1} should be equal to {symbols_modify2}'
    assert sym1 in symbols_modify1, f'symbol {sym1} should be in modification symbols'

    assert qp.qlang.OPERATORS['x'] is None, f'should return None for unknown symbol'
    check_message('ERROR: symbol "x" not found in "OPERATORS"')

    connectors = qp.qlang.CONNECTORS.__str__()
    expected = 'CONNECTORS:\n\t<"%%%": NEW_SELECT_VALS>\n\t<"&&&": AND_SELECT_VALS>\n\t<"///": OR_SELECT_VALS>\n\t<"%%": NEW_SELECT_ROWS>\n\t<"&&": AND_SELECT_ROWS>\n\t<"//": OR_SELECT_ROWS>\n\t<"%": NEW_SELECT_COLS>\n\t<"&": AND_SELECT_COLS>\n\t<"/": OR_SELECT_COLS>\n\t<"$": MODIFY>'
    assert connectors == expected, f'CONNECTORS should be:\n{expected}'


def test_tagging():

    result = get_df_simple_tagged()
    expected = get_df_simple_tagged()
    expected['meta'] = [ '', '', '\n@a: 1']
    result = result.q(r'=a  %%>0   $tag1   %is any;  %%is any;')
    assert result.equals(expected), qp.diff(result, expected, output='str')

    expected['meta'] = [ '', '', '\n@a: 1\n@a: 1']
    result = result.q(r'=a   %%>0  $tag1   %is any;  %%is any;')
    assert result.equals(expected), qp.diff(result, expected, output='str')
    
    expected['meta'] = [ '', '', '\n@a: 1\n@a: 1\n@a: 1']
    result = result.q(r'=a   %%>0  $tag+=1   %is any;  %%is any;')
    assert result.equals(expected), qp.diff(result, expected, output='str')
    
    expected['meta'] = [ '', '', '\n@a: 1']
    result = result.q(r'=a   %%>0  $tag=1   %is any;  %%is any;')
    assert result.equals(expected), qp.diff(result, expected, output='str')

    #no inplace modification should take place
    expected['meta'] = [ '', '', '\n@a: 1']
    result = get_df_simple_tagged()
    result = result.q(r'=a  %%>0   $tag1   %is any;  %%is any;')
    assert result.equals(expected), qp.diff(result, expected, output='str')

    expected['meta'] = [ '', '', '\n@a@b: 1']
    result = get_df_simple_tagged()
    result = result.q(r'a /b  %%all>0  $tag1  %is any;  %%is any;')
    assert result.equals(expected), qp.diff(result, expected, output='str')



def test_to_int():
    result = get_df().q('age $to int;')
    expected = get_df()
    expected['age'] = [
        -25,
        30,
        None,
        None,
        40,
        None,
        None,
        None,
        None,
        None,
        35,
        ]
    expected['age'] = expected['age'].astype('Int64')
    expected = expected.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_float():
    result = get_df().q('=age  $ to float;')
    expected = get_df()
    expected['age'] = [
        -25.0,
        30.0,
        None,
        None,
        40.0,
        None,
        None,
        None,
        None,
        None,
        35.0,
        ]
    expected['age'] = expected['age'].astype('Float64')
    expected = expected.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_num():
    result = get_df().q('=age   $ to num;')
    expected = get_df()
    expected['age'] = [
        -25,
        30,
        np.nan,
        np.nan,
        40,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        35,
        ]
    expected['age'] = expected['age'].astype('object')
    expected = expected.loc[:,['age']].astype('object')
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_str():
    result = get_df().q('=age   $ to str;')
    expected = get_df()
    expected['age'] = [
        '-25',
        '30',
        'nan',
        'None',
        '40.0',
        'forty-five',
        'nan',
        'unk',
        '',
        'unknown',
        '35',
        ]
    expected['age'] = expected['age'].astype(str)
    expected = expected.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_date():
    result = get_df().q('date of birth   $to date;')
    expected = get_df()
    expected['date of birth'] = [
        pd.to_datetime('1995-01-02', dayfirst=False).date(),
        pd.to_datetime('1990/09/14', dayfirst=False).date(),
        pd.to_datetime('1985.08.23', dayfirst=False).date(),
        pd.to_datetime('19800406', dayfirst=False).date(),
        pd.to_datetime('05-11-2007', dayfirst=True).date(),
        pd.to_datetime('06-30-1983', dayfirst=False).date(),
        pd.to_datetime('28-05-1975', dayfirst=True).date(),
        pd.to_datetime('1960 Mar 08', dayfirst=False).date(),
        pd.to_datetime('1955-Jan-09', dayfirst=False).date(),
        pd.to_datetime('1950 Sep 10', dayfirst=False).date(),
        pd.to_datetime('1945 October 11', dayfirst=False).date(),
        ]
    expected['date of birth'] = pd.to_datetime(expected['date of birth']).astype('datetime64[ns]')
    expected = expected.loc[:,['date of birth']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_na():
    result = get_df().q('=age   $to na;')
    expected = get_df()
    expected['age'] = [
        -25,
        '30',
        None,
        None,
        '40.0',
        'forty-five',
        None,
        'unk',
        None,
        'unknown',
        35,
        ]
    expected['age'] = expected['age'].astype('object')
    expected = expected.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_nk():
    result = get_df().q('=age   $to nk;')
    expected = get_df()
    expected['age'] = [
        -25,
        '30',
        np.nan,
        None,
        '40.0',
        'forty-five',
        'nan',
        'unknown',
        '',
        'unknown',
        35,
        ]
    expected['age'] = expected['age'].astype('object')
    expected = expected.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_yn():
    result = get_df().q('=diabetes   $ to yn;')
    expected = get_df()
    expected['diabetes'] = [
        'no',
        'yes',
        None,
        'no',
        'yes',
        'yes',
        'no',
        None,
        None,
        'no',
        'yes',
        ]
    expected['age'] = expected['age'].astype('object')
    expected = expected.loc[:,['diabetes']]
    assert result.equals(expected), qp.diff(result, expected, output='str')



params = [
    (r'%%%is na;  %trim;',                  df.loc[:, ['age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']]),
    (r'%%%is na;  %%trim;',                 df.loc[[1,2,3,4,6,7,8,9,10], :]),
    (r'%%%is na;  %trim;  %%trim;',         df.loc[[1,2,3,4,6,7,8,9,10], ['age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']]),
    (r'%%%is na;  %!trim;',                 df.loc[:, ['ID', 'name', 'date of birth']]),
    (r'%%%is na;  %is any;  %trim;',        df.loc[:, ['age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']]),
    (r'%%%is na;  %is any;  %!trim;',       df.loc[:, ['ID', 'name', 'date of birth']]),
    ]
@pytest.mark.parametrize('code, expected', params)
def test_trim(code, expected):
    temp = get_df().q(code)
    result = get_df().loc[temp.index, temp.columns]
    assert result.equals(expected), qp.diff(result, expected, output='str')



def test_type_inference():
    df1 = pd.DataFrame({1:[1,2,3], 'a':[4,5,6]})
    df2 = pd.DataFrame({'1':[1,2,3], 'a':[4,5,6]})
    result1 = df1.q('1')
    result2 = df2.q('1')
    expected1 = df1.loc[:, [1]]
    expected2 = df2.loc[:, ['1']]
    assert result1.equals(expected1), qp.diff(result1, expected1, output='str')
    assert result2.equals(expected2), qp.diff(result2, expected2, output='str')





