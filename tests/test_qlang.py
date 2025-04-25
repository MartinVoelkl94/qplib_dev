
import pytest
import qplib as qp
import pandas as pd
import numpy as np




def get_df_simple():
    df = pd.DataFrame({
        'a': [-1, 0, 1],
        'b': [1, 2, 3]
        })
    return df

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


def get_df_tagged():
    df1 = get_df()
    df2 = pd.DataFrame('', index=df1.index, columns=['meta', *df1.columns])
    df2.iloc[:, 1:] = df1.loc[:, :]
    return df2


def check_message(expected_message):
    logs = qp.log()
    last = len(logs) - 1
    message = logs.loc[last, 'level'] + ': ' + logs.loc[last, 'text']

    assert message == expected_message, f'\nExpected message: {expected_message}\nGot message: {message}'


df = get_df()
@pytest.mark.parametrize("code, expected_cols, message", [

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
    ('strict=', [], 'WARNING: no columns fulfill the condition in "%strict="'),
    ('strict=ID', ['ID'], None),
    ('strict =ID', ['ID'], None),
    ('strict= ID', ['ID'], None),
    ('strict = ID', ['ID'], None),
    ('strict=id', [], 'WARNING: no columns fulfill the condition in "%strict=id"'),
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
    ('regex=.', [], r'WARNING: no columns fulfill the condition in "%regex=."'),
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

    ])

def test_col_selection(code, expected_cols, message):
    result = df.q(code)
    expected = df.loc[:, expected_cols]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    if message:
        check_message(message)





df = get_df()
@pytest.mark.parametrize('code, expected, message', [

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
    (r'date of birth  %%=1995-01-02',                                df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==1995-01-02',                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995.01.02',                                df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995_01_02',                                df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995´/01´/02',                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==1995 01 02',                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=1995-Jan-02',                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02-01-1995',                                df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02-Jan-1995',                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==Jan-02-1995',                              df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02-01.1995',                                df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%=02 Jan-1995',                               df.loc[[0], ['date of birth']], None),
    (r'date of birth  %%==Jan´/02_1995',                             df.loc[[0], ['date of birth']], None),
    (r'date of birth  $to datetime;  %%=05-11-2007',                 df.loc[[4], ['date of birth']], None),
    (r'date of birth  $to datetime;  %%>1990-01-01',                 df.loc[[0,1,4], ['date of birth']], None),
    (r'date of birth  $to datetime;  %%>1990-01-01  &&<2000-01-01',  df.loc[[0,1], ['date of birth']], None),


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

    (r'diabetes  %%is yn;',              df.loc[[0,1,3,4,5,6,9,10], ['diabetes']], None),
    (r'diabetes  %%is na;  //is yn;',    df.loc[:, ['diabetes']], None),
    (r'diabetes  %%is yes;',             df.loc[[1,4,5,10], ['diabetes']], None),
    (r'diabetes  %%is no;',              df.loc[[0,3,6,9], ['diabetes']], None),

    (r'cholesterol  %%is na;',           df.loc[[2,4,7,9], ['cholesterol']], None),
    (r'age          %%is na;',           df.loc[[2,3,6,8], ['age']], None),
    (r'age          %%strict is na;',           df.loc[[2,3], ['age']], None),


    #using regex equality
    (r'ID %%regex=1....',                                df.loc[[0,1,2], ['ID']], None),
    (r'ID %%regex!=3....',                               df.loc[[0,1,2,3,4,5], ['ID']], None),
    (r'ID %%!regex=3....',                               df.loc[[0,1,2,3,4,5], ['ID']], None),
    (r'name %%regex=\b[A-Z][a-z]*\s[A-Z][a-z]*\b',    df.loc[[0,1,2,3,7], ['name']], None), #two words with first letter capitalized and separated by a space
    (r'name %%regex=^[^A-Z]*´$',                          df.loc[[4], ['name']], None), #all lowercase
    (r'dose %%regex=^(?=.*[a-zA-Z])(?=.*[0-9]).*´$',      df.loc[[0,2,3,4,5,8,10], ['dose']], None), #containing letters and numbers


    #using regex search
    ( 'bp systole %%regex?m', df.loc[[4], ['bp systole']], None),
    (r'bp systole %%regex?\D', df.loc[[2,4,6], ['bp systole']], None),
    ( 'bp systole %%regex?\d', df.loc[[0,1,3,4,5,7,9,10], ['bp systole']], None),


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


    #comparison between columns
    (r'id  %%=@ID',                                  df.loc[:, ['ID']], None),
    (r'age / height  $to num;   %height  %%>@age',   df.loc[[0, 10], ['height']], None),
    (r'age / height  $to num;   %height  %%<@age',   df.loc[[], ['height']], None),
    (r'cholesterol   %%=@bp systole',                df.loc[[2], ['cholesterol']], None),


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
        age / height  $to num;
        age  %%col~ col < df["height"]
        """,
        df.loc[[0, 10], ['age']],
        None
    ),
    (
        r"""
        age / height  $to num;
        age  %%col~ col == df["age"].max()
        age  $to str;
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
        ID  %%regex=1....  //regex=2....  %%save1
        gender  %%=m  //=male  &&load1
        ID / gender
        """,
        df.loc[[0,3,5], ['ID', 'gender']],
        None
    ),
    (
        r"""
        ID  %%regex=1.... // regex=2....  // save 1
        gender %%=m // =male  &&load1
        """,
        df.loc[[0,3,5], ['gender']],
        None
    ), 
    (
        r"""
        ID  %%regex=1.... // regex=2....  &&save1
        gender %%=m // =m // =male  // load 1
        """,
        df.loc[[0,1,2,3,4,5], ['gender']],
        None
    ),
    (
        r"""
        ID  %%regex=1.... // regex=2....   %%save1
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
        gender  %%=f  //=female  %%save a
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
        age  %%>30  %%save=a
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
        weight  %%<70  &&>40  %%save=between 40 and 70
        diabetes %%is yes;  &&load=between 40 and 70   
        """,
        df.loc[[1], ['diabetes']],
        None
    ),
    (
        r"""
        weight  %%<70  &&save1
        &weight  %%>40  &&save2
        diabetes  %%is no;  &&load1
        weight  / diabetes
        """,
        df.loc[[], ['weight', 'diabetes']],
        None
    ),

    ])
def test_row_selection(code, expected, message):
    df = get_df()
    temp = df.q(code)
    result = df.loc[temp.index, temp.columns]
    assert result.equals(expected), qp.diff(result, expected, output='str')
    if message:
        check_message(message)




def test_col_eval():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id $ col~ df["name"]
        """)
    df1['ID'] = df1['name']
    expected = df1.loc[:, ['ID']]
    assert result.equals(expected), 'failed test0: copy col contents\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id $ col~ df["name"]
        is any; %%is any;
        """)
    df1['ID'] = df1['name']
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test1: copy col contents and select all\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age $ col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1['age'] = df1['name']
    expected = df1.loc[:, ['ID', 'age']]
    assert result.equals(expected), 'failed test2: copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        $ col~ df["name"]
        """)
    for col in df1.columns:
        df1[col] = df1['name']
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test3: copy col contents to all cols\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age %%is num; $ col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1['age'] = df1['name']
    expected = df1.loc[:, ['ID', 'age']]
    assert result.equals(expected), 'failed test4: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age %%all is num; $ col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1['age'] = df1['name']
    expected = df1.loc[[0,1,2,3,4,8,10], ['ID', 'age']]
    assert result.equals(expected), 'failed test5: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age %%each is num; $ col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1.loc[[0,1,2,3,4,8,10], 'age'] = df1.loc[[0,1,2,3,4,8,10], 'name']
    expected = df1.loc[:, ['ID', 'age']]
    assert (result.loc[[5,6,7,9], 'age'] != df.loc[[5,6,7,9], 'name']).all(), 'failed test6: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')
    assert result.equals(expected), 'failed test7: conditionally copy col contents to multiple cols\n' + qp.diff(result, expected, output='str')



def test_eval():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name $ ~ x.lower()
        is any; %%is any;
        """)
    df1['name'] = df1['name'].str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test0: change to lowercase\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name  %%!~ x == x.lower()  $ ~ x.lower()
        is any; %%is any;
        """)
    df1['name'] = df1['name'].str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test1: conditionally change to lowercase\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        gender  $to str;  $~ x.lower()
        is any;
        """)
    df1['gender'] = df1['gender'].astype(str).str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test2: convert and change to lowercase\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id %%10001  $ ~ str(10001)
        """)
    df1.loc[0, 'ID'] = '10001'
    expected = df1.loc[[0], ['ID']]
    assert result.equals(expected), 'failed test3: convert single entry to str\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age %%is num; $ ~ str(0)
        """)
    df1['ID'] = str(0)
    df1['age'] = str(0)
    expected = df1.loc[:, ['ID', 'age']]
    assert result.equals(expected), 'failed test4: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age %%all is num; $ ~ 0
        """)
    df1.loc[[0,1,2,3,4,8,10], 'ID'] = 0
    df1.loc[[0,1,2,3,4,8,10], 'age'] = 0
    expected = df1.loc[[0,1,2,3,4,8,10], ['ID', 'age']]
    assert result.equals(expected), 'failed test5: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age %%each is num; $ ~ 10
        """)
    df1['ID'] = 10
    df1.loc[[0,1,2,3,4,8,10], 'age'] = 10
    expected = df1.loc[:, ['ID', 'age']]
    assert (result['ID'] == 10).all(), 'failed test6: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')
    assert (result.loc[[5,6,7,9], 'age'] != 10).all(), 'failed test7: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')
    assert result.equals(expected), 'failed test8: conditionally convert multiple entries to str\n' + qp.diff(result, expected, output='str')




def test_header():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id  $header id')
    expected = df2.rename(columns={'ID': 'id'}).loc[:, ['id']]
    assert result.equals(expected), 'failed test0: rename col header\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    df2 = get_df()
    result = df1.q(r'id  $header id   %name  $header n   %date of birth  $header dob  %is any;')
    expected = df2.rename(columns={'ID': 'id', 'name': 'n', 'date of birth': 'dob'})
    assert result.equals(expected), 'failed test1: rename multiple col headers\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    df2 = get_df()
    result = df1.q('id  $header += abc')
    expected = df2.rename(columns={'ID': 'IDabc'}).loc[:, ['IDabc']]
    assert result.equals(expected), 'failed test2: append to col header\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    df2 = get_df()
    result = df1.q('id / name / date of birth   $header += abc  %is any;')
    expected = df2.rename(columns={'ID': 'IDabc', 'name': 'nameabc', 'date of birth': 'date of birthabc'})
    assert result.equals(expected), 'failed test3: append to multiple col headers\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    df2 = get_df()
    result = df1.q('id  $header ~ x.lower() + str(len(x))   %is any;')
    expected = df2.rename(columns={'ID': 'id2'})
    assert result.equals(expected), 'failed test4: rename col header with eval\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    df2 = get_df()
    result = df1.q('id / weight / diabetes    $header ~ x.lower() + str(len(x))   %is any;')
    expected = df2.rename(columns={'ID': 'id2', 'weight': 'weight6', 'diabetes': 'diabetes8'})
    assert result.equals(expected), 'failed test5: rename multiple col headers with eval\n' + qp.diff(result, expected, output='str')



df = get_df()
@pytest.mark.parametrize("code, metadata", [
    (r'a %%>0  $meta = >0  %is any;  %%is any;', ['', '', '>0']),
    (r'a %%>0  $meta += >0  %is any;  %%is any;', ['', '', '>0']),
    (r'=a   %%>0   $meta +=>0  %is any;  %%is any;', [ '', '', '>0']),
    (r'=a%%>0     $meta+= >0  %is any;  %%is any;', [ '', '', '>0']),
    (r'=a   %%>0   $meta ~ x + ">0"  %is any;  %%is any;', [ '', '', '>0']),
    (r'=a   %%>0   $meta col~ col + ">0"  %is any;  %%is any;', [ '', '', '>0']),
    (r'=a%%>0     $tag   %is any;  %%is any;', [ '', '', '\n@a: ']),
    (r'=a%%>0  /b     $tag  %is any;  %%is any;', [ '', '', '\n@a@b: ']),
    (r'=a%%>0  /b     $tag=value >0   %is any;  %%is any;', [ '', '', '\n@a@b: value >0']),
    ])
def test_metadata(code, metadata):
    df = get_df_simple_tagged()
    df = df.q(code)
    df1 = get_df_simple_tagged()
    df1['meta'] = metadata
    assert df.equals(df1), 'failed test: metadata tagging\n' + qp.diff(df, df1, output='str')


def test_metadata_continous():

    df = get_df_simple_tagged()
    df1 = get_df_simple_tagged()
    df1['meta'] = [ '', '', '>0']
    df = df.q(r'=a  %%>0   $meta +=>0   %is any;  %%is any;')
    assert df.equals(df1), 'failed test0: continous metadata tagging\n' + qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '', '>0>0']
    df = df.q(r'=a   %%>0  $meta+=>0   %is any;  %%is any;')
    assert df.equals(df1), 'failed test1: continous metadata tagging\n' + qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '0', '>0>0']
    df = df.q(r'=a   %%==0    $meta += 0   %is any;  %%is any;')
    assert df.equals(df1), 'failed test2: continous metadata tagging\n' + qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '', '>0>0']
    df = df.q(r'a   %%==0    $meta ~ x.replace("0", "")   %is any;  %%is any;')
    assert df.equals(df1), 'failed test3: continous metadata tagging\n' + qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '', '>>']
    df = df.q(r'=a   %%>0    $meta~x.replace("0", "")   %is any;  %%is any;')
    assert df.equals(df1), 'failed test4: continous metadata tagging\n' + qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '', '']
    df = df.q(r'=a     $meta=   %is any;  %%is any;')
    assert df.equals(df1), 'failed test5: continous metadata tagging\n' + qp.diff(df, df1, output='str')



@pytest.mark.parametrize("code, content, cols", [
    (r'$new', '', ['new1']),
    (r'$new a', 'a', ['new1']),
    (r'$newa', 'a', ['new1']),
    (r'$new =a', 'a', ['new1']),
    (r'$new= a', 'a', ['new1']),
    (r'$new = a', 'a', ['new1']),
    (r'$new ~ "a"', 'a', ['new1']),
    (r'$new ~ df["ID"]', qp.get_df()['ID'], ['new1']),
    (r'$new @ID', qp.get_df()['ID'].astype(str), ['new1']),
    (r'$new  %id', '', ['ID']),
    ])
def test_new_col(code, content, cols):
    df1 = get_df()
    result = df1.q(code)
    df2 = get_df()
    df2['new1'] = content
    expected = df2.loc[:, cols]
    assert result.equals(expected), 'failed test0: creating new col\n' + qp.diff(result, expected, output='str')


def test_new_col1():
    df1 = get_df()
    result = df1.q('$new a  $header = new col')
    df2 = get_df()
    df2['new col'] = 'a'
    expected = df2.loc[:,['new col']]
    assert result.equals(expected), 'failed test1: creating new col with header\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    result = df1.q('$new a   &newb   /=new1 /=new2')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[:,['new1']]
    assert result.equals(expected), 'failed test2: creating new col\n' + qp.diff(result, expected, output='str')
    check_message('WARNING: no columns fulfill the condition in "/=new2"')


    df1 = get_df()
    result = df1.q('$new a /new b   /=new1 /=new2')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[:,['new1']]
    assert result.equals(expected), 'failed test3: creating new col\n' + qp.diff(result, expected, output='str')
    check_message('WARNING: no columns fulfill the condition in "/=new2"')


    df1 = get_df()
    result = df1.q('$new a  $new b   /=new1 /=new2')
    df2 = get_df()
    df2['new1'] = 'a'
    df2['new2'] = 'b'
    expected = df2.loc[:,['new1', 'new2']]
    assert result.equals(expected), 'failed test4: creating multiple new cols\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    result = df1.q(r'%%idx = 0  $new a')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[[0],['new1']]
    assert result.equals(expected), 'failed test5: creating new col with index\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    result = df1.q(r'$new a  %%idx = 0  ')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[[0],['new1']]
    assert result.equals(expected), 'failed test6: creating new col and select with index\n' + qp.diff(result, expected, output='str')


    df1 = get_df()
    result = df1.q(r'%%idx = 0   %%save1  %%is any;   $new a  %%load1')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[[0],['new1']]
    assert result.equals(expected), 'failed test7: check if selection saving and loading works correctly with new col creation\n' + qp.diff(result, expected, output='str')



@pytest.mark.parametrize('code, expected, message', [
    (r'id %%?1  %%save1   %%is any;  %%load1',                   df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%?1  %%save=1   %%is any;  %%load=1',                 df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%any?1  %%save1   %%is any;  %%load1',                df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%all?1  %%save1   %%is any;  %%load1',                df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%each?1  %%save1   %%is any;  %%load1',               df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%?1  %%save1   %is any;  %%is any;   %id  %%load1',   df.loc[[0,1,2,3,6],['ID']], None),
    (r'id %%?2  %%save1  %%?1  %%save1  %%is any;  %%load1',     df.loc[[0,1,2,3,6],['ID']], None),
    
    (
        r"""
        id   %%?1   %%save1
        id   %%?2   %%save2
        id   %%load1   &&load2
        """,
        df.loc[[1,3],['ID']],
        None
    ),

    (
        r"""
        id   %%?1   %%save1
        id   %%?2   %%save2
        id   %%load1   //load2
        """,
        df.loc[[0,1,2,3,4,5,6,7],['ID']],
        None
    ),
    
    (
        r"""
        id   %%!?1   %%save1
        id   %%?2   %%save2
        id   %%load1   &&load2
        """,
        df.loc[[4,5,7],['ID']],
        None
    ),
    
    (
        r"""
        id   %%!?1   %%save1
        id   %%!?2   %%save2
        id   %%load1   &&load2
        """,
        df.loc[[8,9,10],['ID']],
        None
    ),
    ])
def test_save_load(code, expected, message):
    result = get_df().q(code)
    assert result.equals(expected), qp.diff(result, expected, output='str')
    if message:
        check_message('WARNING: a selection was already saved as "1". overwriting it')



def test_selection_scopes():
    df1 = qp.get_df()
    result = df1.q(r'%%is na; $')
    df2 = qp.get_df()
    df2.loc[[1,2,3,4,6,7,8,9,10], :] = ''
    expected = df2.loc[[1,2,3,4,6,7,8,9,10], :]
    assert result.equals(expected), 'failed test0: check if default selection scope is "any"\n' + qp.diff(result, expected, output='str')
    

    df1 = qp.get_df()
    result = df1.q(r'%%any is na; $')
    df2 = qp.get_df()
    df2.loc[[1,2,3,4,6,7,8,9,10], :] = ''
    expected = df2.loc[[1,2,3,4,6,7,8,9,10], :]
    assert result.equals(expected), 'failed test1: check selection scope "any"\n' + qp.diff(result, expected, output='str')


    df1 = qp.get_df()
    result = df1.q(r'%%all is na; $')
    df2 = qp.get_df()
    expected = df2.loc[[], :]
    assert result.equals(expected), 'failed test2: check selection scope "all"\n' + qp.diff(result, expected, output='str')
    check_message('WARNING: value modification cannot be applied when no values where selected')


    df1 = qp.get_df()
    result = df1.q(r'%%each is na; $')
    df2 = qp.get_df()
    df2.loc[[2,3,6], 'age'] = ''
    df2.loc[[7,8], 'gender'] = ''
    df2.loc[[2,4], 'height'] = ''
    df2.loc[[3,6], 'weight'] = ''
    df2.loc[[2,6], 'bp systole'] = ''
    df2.loc[[2,4,6,7,10], 'bp diastole'] = ''
    df2.loc[[2,4,7], 'cholesterol'] = ''
    df2.loc[[2,7,8], 'diabetes'] = ''
    df2.loc[[1,6,7], 'dose'] = ''
    expected = df2.loc[[1,2,3,4,6,7,8,9,10], :]
    assert result.equals(expected), 'failed test3: check selection scope "each"\n' + qp.diff(result, expected, output='str')



def test_set_val():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name $ a
        is any; %%is any;
        """)
    df1['name'] = 'a'
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test0: set values\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name $ =a
        is any; %%is any;
        """)
    df1['name'] = 'a'
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test1: set values\n' + qp.diff(result, expected, output='str')


    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name $ a
        gender $ b
        is any; %%is any;
        """)
    df1['name'] = 'a'
    df1['gender'] = 'b'
    expected = df1.loc[:, :]
    assert result.equals(expected), 'failed test2: set values in multiple cols\n' + qp.diff(result, expected, output='str')




@pytest.mark.parametrize("code, expected_cols", [
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
    ])
def test_sort(code, expected_cols):
    df = qp.get_df()
    result = df.q(code)
    expected = df.sort_values(by=expected_cols)
    assert result.equals(expected), 'failed test: sort values\n' + qp.diff(result, expected, output='str')



def test_tagging():

    df = get_df_simple_tagged()
    df1 = get_df_simple_tagged()
    df1['meta'] = [ '', '', '\n@a: 1']
    df = df.q(r'=a  %%>0   $tag1   %is any;  %%is any;')
    assert df.equals(df1), qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '', '\n@a: 1\n@a: 1']
    df = df.q(r'=a   %%>0  $tag1   %is any;  %%is any;')
    assert df.equals(df1), qp.diff(df, df1, output='str')
    
    df1['meta'] = [ '', '', '\n@a: 1\n@a: 1\n@a: 1']
    df = df.q(r'=a   %%>0  $tag+=1   %is any;  %%is any;')
    assert df.equals(df1), qp.diff(df, df1, output='str')
    
    df1['meta'] = [ '', '', '\n@a: 1']
    df = df.q(r'=a   %%>0  $tag=1   %is any;  %%is any;')
    assert df.equals(df1), qp.diff(df, df1, output='str')

    #no inplace modification should take place
    df1['meta'] = [ '', '', '\n@a: 1']
    df = get_df_simple_tagged()
    df = df.q(r'=a  %%>0   $tag1   %is any;  %%is any;')
    df2 = df.q(r'=a  %%>0   $tag2   %is any;  %%is any;')
    assert df.equals(df1), qp.diff(df, df1, output='str')

    df1['meta'] = [ '', '', '\n@a@b: 1']
    df = get_df_simple_tagged()
    df = df.q(r'a /b  %%all>0  $tag1  %is any;  %%is any;')
    assert df.equals(df1), qp.diff(df, df1, output='str')



def test_to_int():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('age $to int;')
    df2['age'] = [-25, 30, None, None, 40, None, None, None, None, None, 35]
    df2['age'] = df2['age'].astype('Int64')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_float():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age  $ to float;')
    df2['age'] = [-25.0, 30.0, None, None, 40.0, None, None, None, None, None, 35.0]
    df2['age'] = df2['age'].astype('Float64')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_num():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   $ to num;')
    df2['age'] = [-25, 30, np.nan, np.nan, 40, np.nan, np.nan, np.nan, np.nan, np.nan, 35]
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['age']].astype('object')
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_str():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   $ to str;')
    df2['age'] = ['-25', '30', 'nan', 'None', '40.0', 'forty-five', 'nan', 'unk', '', 'unknown', '35']
    df2['age'] = df2['age'].astype(str)
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_date():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('date of birth   $to date;')
    df2['date of birth'] = [
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
    df2['date of birth'] = pd.to_datetime(df2['date of birth']).astype('datetime64[ns]')
    expected = df2.loc[:,['date of birth']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_na():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   $to na;')
    df2['age'] = [-25, '30', None, None, '40.0', 'forty-five', None, 'unk', None, 'unknown', 35]
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_nk():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   $to nk;')
    df2['age'] = [-25, '30', np.nan, None, '40.0', 'forty-five', 'nan', 'unknown', '', 'unknown', 35]
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, output='str')


def test_to_yn():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=diabetes   $ to yn;')
    df2['diabetes'] = ['no', 'yes', None, 'no', 'yes', 'yes', 'no', None, None, 'no', 'yes']
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['diabetes']]
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



