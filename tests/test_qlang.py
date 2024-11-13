
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




df = get_df()
@pytest.mark.parametrize("instructions, expected_cols", [

    #using operator: equals
    ('', df.columns),
    ('ID', ['ID']),
    (' ID', ['ID']),
    ('ID ', ['ID']),
    (' ID ', ['ID']),
    ('=ID', ['ID']),
    (' =ID', ['ID']),
    ('= ID', ['ID']),
    ('= ID ', ['ID']),
    (' = ID ', ['ID']),
    ('date of birth', ['date of birth']),
    ('date of birth / age', ['date of birth', 'age']),
    ('=date of birth / age', ['date of birth', 'age']),
    ('=date of birth / =age', ['date of birth', 'age']),
    ('date of birth / =age', ['date of birth', 'age']),
    ('!=date of birth', ['ID', 'name', 'age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']),
    ("""ID""", ['ID']),
    (
        """ID
        """,
        ['ID']
    ),
    (
        """
        ID""",
        ['ID']
    ),
    (
        """
        ID
        """,
        ['ID']
    ),
    (r"""ID""", ['ID']),
    (
        r"""ID
        """,
        ['ID']
    ),
    (
        r"""
        ID""",
        ['ID']
    ),
    (
        r"""
        ID
        """,
        ['ID']
    ),


    #using operator: strict equals
    ('==', []),
    ('==ID', ['ID']),
    ('= =ID', []),
    ('==id', []),
    ('==date of birth / age', ['date of birth', 'age']),
    ('==date of birth / =age', ['date of birth', 'age']),
    ('!==date of birth', ['ID', 'name', 'age', 'gender', 'height', 'weight', 'bp systole', 'bp diastole', 'cholesterol', 'diabetes', 'dose']),
   

    #using operator: contains
    ('?bp', ['bp systole', 'bp diastole']),
    ('?I', ['ID', 'date of birth', 'height', 'weight', 'bp diastole', 'diabetes']),
    ('!?I', ['name', 'age', 'gender', 'bp systole', 'cholesterol', 'dose']),

    #using operator: strict contains
    ('??I', ['ID']),


    #using operator: regex equality
    ('r=ID$', ['ID']),
    ('r=ID', ['ID']),
    ('r=.', []),
    ('r=..', ['ID']),

    #using operator: regex strict equality
    ('r? ID', ['ID']),
    ('r? e.', ['date of birth', 'gender', 'height', 'weight', 'cholesterol', 'diabetes']),


    #using multiple conditions
    ('?bp / =diabetes', ['bp systole', 'bp diastole', 'diabetes']),
    ('?bp / =diabetes /= cholesterol', ['bp systole', 'bp diastole', 'cholesterol', 'diabetes']),
    ('?bp /=cholesterol/= diabetes', ['bp systole', 'bp diastole', 'cholesterol', 'diabetes']),
    ('?bp & ?systole', ['bp systole']),
    ('?bp & !?systole', ['bp diastole']),
    ('?bp & !?systole & ?diastole', ['bp diastole']),
    ('?bp & !?systole / ?ID', ['ID', 'bp diastole']),

    ])

def test_col_selection(instructions, expected_cols):
    result = df.q(instructions)
    expected = df.loc[:, expected_cols]
    assert result.equals(expected), qp.diff(result, expected, returns='str')





df = get_df()
@pytest.mark.parametrize("instructions, expected_df", [

    #numeric comparison
    ('age ´r =30', df.loc[[1], ['age']]),
    ('age ´r ==30.0', df.loc[[1], ['age']]),
    ('age ´r >30', df.loc[[4,10], ['age']]),
    ('age ´r >=30', df.loc[[1,4,10], ['age']]),
    ('age ´r <30', df.loc[[0], ['age']]),
    ('age ´r <=30', df.loc[[0,1], ['age']]),
    ('age ´r !=30', df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']]),
    ('age ´r !==30.0', df.loc[[0,2,3,4,5,6,7,8,9,10], ['age']]),
    ('age ´r !>30', df.loc[[0,1,2,3,5,6,7,8,9], ['age']]),
    ('age ´r !>=30', df.loc[[0,2,3,5,6,7,8,9], ['age']]),
    ('age ´r !<30', df.loc[[1,2,3,4,5,6,7,8,9,10], ['age']]),
    ('age ´r !<=30', df.loc[[2,3,4,5,6,7,8,9,10], ['age']]),
    ('age ´r =40', df.loc[[4], ['age']]),
    ('age ´r ==40', df.loc[[4], ['age']]),
    ('age ´r =40.0', df.loc[[4], ['age']]),
    ('age ´r ==40.0', df.loc[[4], ['age']]),
    
    #date comparison
    ('date of birth ´r =1995-01-02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r ==1995-01-02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =1995.01.02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =1995_01_02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =1995`/01`/02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r ==1995 01 02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =1995-Jan-02', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =02-01-1995', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =02-Jan-1995', df.loc[[0], ['date of birth']]),
    ('date of birth ´r ==Jan-02-1995', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =02-01.1995', df.loc[[0], ['date of birth']]),
    ('date of birth ´r =02 Jan-1995', df.loc[[0], ['date of birth']]),
    ('date of birth ´r ==Jan`/02_1995', df.loc[[0], ['date of birth']]),
    ('date of birth ´v to datetime ´r =05-11-2007', df.loc[[4], ['date of birth']]),
    ('date of birth ´v to datetime ´r >1990-01-01', df.loc[[0,1,4], ['date of birth']]),
    ('date of birth ´v to datetime ´r >1990-01-01 & <2000-01-01', df.loc[[0,1], ['date of birth']]),

    #using type operators
    ('name ´r is str', df.loc[:, ['name']]),
    ('name ´r !is str', df.loc[[], ['name']]),
    ('name ´r is num', df.loc[[], ['name']]),
    ('name ´r !is num', df.loc[:, ['name']]),
    ('name ´r is na', df.loc[[], ['name']]),
    ('name ´r !is na', df.loc[:, ['name']]),
    ('age ´r is na', df.loc[[2,3,6,8], ['age']]),
    ('cholesterol ´r is na', df.loc[[2,4,7,9], ['cholesterol']]),
    ('weight ´r is num', df.loc[[0,1,4,6,7,9,10], ['weight']]),
    ('weight ´r is num & !is na', df.loc[[0,1,7,9,10], ['weight']]),
    ('diabetes ´r is yn', df.loc[[0,1,3,4,5,6,9,10], ['diabetes']]),
    ('diabetes ´r is na / is yn', df.loc[:, ['diabetes']]),
    ('diabetes ´r is yes', df.loc[[1,4,5,10], ['diabetes']]),
    ('diabetes ´r is no', df.loc[[0,3,6,9], ['diabetes']]),


    #using regex equality
    ('ID ´r r=1....', df.loc[[0,1,2], ['ID']]),
    ('ID ´r !r=3....', df.loc[[0,1,2,3,4,5], ['ID']]),
    ('name ´r r=\\b[A-Z][a-z]*\\s[A-Z][a-z]*\\b', df.loc[[0,1,2,3,7], ['name']]), #two words with first letter capitalized and separated by a space
    ('name ´r r=^[^A-Z]*$', df.loc[[4], ['name']]), #all lowercase
    ('dose ´r r=^(?=.*[a-zA-Z])(?=.*[0-9]).*$', df.loc[[0,2,3,4,5,8,10], ['dose']]), #containing letters and numbers


    #using regex search
    ('bp systole ´r r?m', df.loc[[4], ['bp systole']]),
    (r'bp systole ´r r?\D', df.loc[[2,4,6], ['bp systole']]),
    ('bp systole ´r r?\d', df.loc[[0,1,3,4,5,7,9,10], ['bp systole']]),


     #using index
    ('´r idx = 3', df.iloc[[3], :]),
    ('´r idx > 5', df.iloc[6:, :]),
    ('´r idx < 5',df.iloc[:5, :]),
    ('´r idx >= 5', df.iloc[5:, :]),
    ('´r idx <= 5', df.iloc[:6, :]),
    ('´r idx != 5', df.iloc[[0,1,2,3,4,6,7,8,9,10], :]),
    ('´r idx == 5', df.iloc[[5], :]),
    ('´r idx >5 & idx <8', df.iloc[6:8, :]),
    ('´r idx >5 & idx <8 & idx != 6', df.iloc[[7], :]),
    ('´r idx >5 & idx <8 & idx != 6 & idx != 7', df.iloc[[], :]),
    ('´r idx ~ len(str(x)) > 1', df.iloc[[10], :]),
    ('´r idx ?1', df.iloc[[1, 10], :]),


    #comparison between columns
    ('age / height ´v to num  ´c height ´r > @age', df.loc[[0, 10], ['height']]),
    ('age / height ´v to num  ´c height ´r < @age', df.loc[[], ['height']]),
    ('id ´r = @ID', df.loc[:, ['ID']]),
    ('cholesterol ´r = @bp systole', df.loc[[2], ['cholesterol']]),


    #apply row filter condition on multiple cols
    ('id / name ´r ?j', df.loc[[0,1,2,9,10], ['ID', 'name']]),
    ('id / name ´r ?j / ?n', df.loc[[0,1,2,3,5,8,9,10], ['ID', 'name']]),
    ('id / name ´r ?j & ?n', df.loc[[0,1,2,10], ['ID', 'name']]),
    ('height / weight ´r is num', df.loc[:, ['height', 'weight']]),
    ('height / weight ´r any is num', df.loc[:, ['height', 'weight']]),
    ('height / weight ´r anyis num', df.loc[:, ['height', 'weight']]),
    ('height / weight ´r all is num', df.loc[[0,6,9,10], ['height', 'weight']]),


    #using uniqueness operators
    (
        r"""
        diabetes ´r is unique
        is any
        """,
        df.loc[[1,2,4,6,7,8,9], :]
    ),
    (
        r"""
        diabetes ´r is first
        is any
        """,
        df.loc[[0,1,2,4,5,6,7,8,9], :]
    ),
    (
        r"""
        diabetes ´r is last
        is any
        """,
        df.loc[[1,2,3,4,6,7,8,9,10], :]
    ),


    #by evaluating python expressions
    ('age ´r ~ isinstance(x, int)', df.loc[[0, 10], ['age']]),
    (
        r"""
        age / height ´v to num
        age ´r col~ col < df["height"]
        """,
        df.loc[[0, 10], ['age']]
    ),
    (
        r"""
        age / height ´v to num
        age ´r col~ col == df["age"].max()
        age ´v to str
        """,
        df.loc[[4], ['age']]
    ),


    #combining multiple instructions and conditions
    (
        r"""
        ID ´r r=1....
        diabetes ´r is yes
        """,
        df.loc[[1, 4, 5, 10], ['diabetes']]
    ),
    (
        r"""
        ID ´r r=1....
        diabetes ´r & is yes
        ID / diabetes
        """,
        df.loc[[1], ['ID', 'diabetes']]
    ),
    (
        r"""
        diabetes ´r is yes
        ID ´r & r=1....
        / diabetes
        """,
        df.loc[[1], ['ID', 'diabetes']]
    ),
    (
        r"""
        diabetes ´r is yes
        / ID ´r & r=1....
        """,
        df.loc[[1], ['ID', 'diabetes']]
    ),
    (
        r"""
        ID  ´r r=1.... / r=2....  ´m save1
        gender ´r =m / =male & load1
        ID / gender
        """,
        df.loc[[0,3,5], ['ID', 'gender']]
    ),
    (
        r"""
        ID  ´r r=1.... / r=2....  ´m save 1
        gender ´r =m / =male &load1
        """,
        df.loc[[0,3,5], ['gender']]
    ), 
    (
        r"""
        ID  ´r r=1.... / r=2....  ´m save1
        gender ´r =m / =m / =male  / load 1
        """,
        df.loc[[0,1,2,3,4,5], ['gender']]
    ),
    (
        r"""
        ID  ´r r=1.... / r=2....  ´m save1
        gender ´r =f / =f / =female  / load 1
        ID
        """,
        df.loc[[0,1,2,3,4,5,10], ['ID']]
    ),
    (
        r"""
        gender ´r =f / =female
        age ´r & >30
        """,
        df.loc[[10], ['age']]
    ),
    (
        r"""
        gender ´r =f / =female  ´m save a
        age ´r >30  & load a 
        """,
        df.loc[[10], ['age']]
    ),
    (
        r"""
        age ´r >30
        age ´r / <18
        """,
        df.loc[[0,4,10], ['age']]
    ),
    (
        r"""
        age ´r >30  ´m save a
        age ´r <18 /loada
        """,
        df.loc[[0,4,10], ['age']]
    ),
    (
        r"""
        age ´r >30 / <18
        """,
        df.loc[[0,4,10], ['age']]
    ),
    (
        r"""
        weight ´r <70 & >40  ´m save between 40 and 70
        diabetes ´r is yes & loadbetween 40 and 70   
        """,
        df.loc[[1], ['diabetes']]
    ),
    (
        r"""
        weight ´r <70  ´m save <70
        &weight ´r >40  ´msave>40
        diabetes ´r is no & load<70
        weight / diabetes
        """,
        df.loc[[], ['weight', 'diabetes']]
    ),

    ])
def test_row_selection(instructions, expected_df):
    df = get_df()
    temp = df.q(instructions)
    result = df.loc[temp.index, temp.columns]
    assert result.equals(expected_df), qp.diff(result, expected_df, returns='str')



df = get_df()
@pytest.mark.parametrize("code, metadata", [
    ('a ´r >0  ´m = >0  ´c is any  ´r is any', ['', '', '>0']),
    ('a ´r >0  ´m += >0  ´c is any  ´r is any', ['', '', '>0']),
    ('=a   ´r >0   ´m +=>0  ´c is any  ´r is any', [ '', '', '>0']),
    ('=a´r >0     ´m+= >0  ´c is any  ´r is any', [ '', '', '>0']),
    ('=a   ´r >0   ´m ~ x + ">0"  ´c is any  ´r is any', [ '', '', '>0']),
    ('=a   ´r >0   ´m col~ col + ">0"  ´c is any  ´r is any', [ '', '', '>0']),
    ('=a´r >0     ´mtag   ´c is any  ´r is any', [ '', '', '\n@a: ']),
    ('=a´r >0  ´c /b     ´m tag  ´c is any  ´r is any', [ '', '', '\n@a@b: ']),
    ('=a´r >0  ´c /b     ´m tagvalue >0   ´c is any  ´r is any', [ '', '', '\n@a@b: value >0']),
    ])

def test_metadata(code, metadata):
    df = get_df_simple_tagged()
    df = df.q(code)
    df1 = get_df_simple_tagged()
    df1['meta'] = metadata
    assert df.equals(df1), qp.diff(df, df1, returns='str')


def test_scope1():
    df1 = qp.get_df()
    result = df1.q('´r is na ´v')
    df2 = qp.get_df()
    df2.loc[[1,2,3,4,6,7,8,9,10], :] = ''
    expected = df2.loc[[1,2,3,4,6,7,8,9,10], :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')
    
def test_scope2():
    df1 = qp.get_df()
    result = df1.q('´r any is na ´v')
    df2 = qp.get_df()
    df2.loc[[1,2,3,4,6,7,8,9,10], :] = ''
    expected = df2.loc[[1,2,3,4,6,7,8,9,10], :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')

def test_scope3():
    df1 = qp.get_df()
    result = df1.q('´r all is na ´v')
    df2 = qp.get_df()
    expected = df2.loc[[], :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')

def test_scope4():
    df1 = qp.get_df()
    result = df1.q('´r val is na ´v')
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
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_metadata_continous():

    df = get_df_simple_tagged()
    df1 = get_df_simple_tagged()
    df1['meta'] = [ '', '', '>0']
    df = df.q('=a  ´r >0   ´m +=>0  ´c is any  ´r is any')
    assert df.equals(df1), qp.diff(df, df1, returns='str')

    df1['meta'] = [ '', '', '>0>0']
    df = df.q('=a   ´r >0  ´m+=>0  ´c is any  ´r is any')
    assert df.equals(df1), qp.diff(df, df1, returns='str')

    df1['meta'] = [ '', '0', '>0>0']
    df = df.q('=a   ´r ==0    ´m += 0  ´c is any  ´r is any')
    assert df.equals(df1), qp.diff(df, df1, returns='str')

    df1['meta'] = [ '', '', '>0>0']
    df = df.q('a   ´r ==0    ´m ~ x.replace("0", "")  ´c is any  ´r is any')
    assert df.equals(df1), qp.diff(df, df1, returns='str')

    df1['meta'] = [ '', '', '>>']
    df = df.q('=a   ´r >0    ´m~x.replace("0", "")  ´c is any  ´r is any')
    assert df.equals(df1), qp.diff(df, df1, returns='str')

    df1['meta'] = [ '', '', '']
    df = df.q('=a     ´m=  ´c is any  ´r is any')
    assert df.equals(df1), qp.diff(df, df1, returns='str')



def test_set_val():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name ´v a
        is any ´r is any
        """)
    df1['name'] = 'a'
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_set_val1():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name ´v =a
        is any ´r is any
        """)
    df1['name'] = 'a'
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_set_val2():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name ´v a
        gender ´v b
        is any ´r is any
        """)
    df1['name'] = 'a'
    df1['gender'] = 'b'
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')



def test_eval():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name ´v ~ x.lower()
        is any ´r is any
        """)
    df1['name'] = df1['name'].str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval1():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        name  ´r !x? x == x.lower()  ´v ~ x.lower()
        is any ´r is any
        """)
    df1['name'] = df1['name'].str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval2():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        gender ´v to str & ~ x.lower()
        is any
        """)
    df1['gender'] = df1['gender'].astype(str).str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval3():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        gender ´v to str / ~ x.lower()
        is any
        """)
    df1['gender'] = df1['gender'].astype(str).str.lower()
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval4():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id ´r 10001  ´v ~ str(10001)
        """)
    df1.loc[0, 'ID'] = '10001'
    expected = df1.loc[[0], ['ID']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval5():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´r is num ´v ~ str(0)
        """)
    df1['ID'] = str(0)
    df1['age'] = str(0)
    expected = df1.loc[:, ['ID', 'age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval6():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´r all is num ´v ~ 0
        """)
    df1.loc[[0,1,2,3,4,8,10], 'ID'] = 0
    df1.loc[[0,1,2,3,4,8,10], 'age'] = 0
    expected = df1.loc[[0,1,2,3,4,8,10], ['ID', 'age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_eval7():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´r val is num ´v ~ 10
        """)
    df1['ID'] = 10
    df1.loc[[0,1,2,3,4,8,10], 'age'] = 10
    expected = df1.loc[:, ['ID', 'age']]
    assert (result['ID'] == 10).all()
    assert (result.loc[[5,6,7,9], 'age'] != 10).all()
    assert result.equals(expected), qp.diff(result, expected, returns='str')



def test_col_eval():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id ´v col~ df["name"]
        """)
    df1['ID'] = df1['name']
    expected = df1.loc[:, ['ID']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')

def test_col_eval1():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id ´v col~ df["name"]
        is any ´r is any
        """)
    df1['ID'] = df1['name']
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')

def test_col_eval2():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´v col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1['age'] = df1['name']
    expected = df1.loc[:, ['ID', 'age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_col_eval3():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        ´v col~ df["name"]
        """)
    for col in df1.columns:
        df1[col] = df1['name']
    expected = df1.loc[:, :]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_col_eval4():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´r is num ´v col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1['age'] = df1['name']
    expected = df1.loc[:, ['ID', 'age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_col_eval5():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´r all is num ´v col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1['age'] = df1['name']
    expected = df1.loc[[0,1,2,3,4,8,10], ['ID', 'age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_col_eval6():
    df = qp.get_df()
    df1 = qp.get_df()
    result = df.q(
        r"""
        id / age ´r val is num ´v col~ df["name"]
        """)
    df1['ID'] = df1['name']
    df1.loc[[0,1,2,3,4,8,10], 'age'] = df1.loc[[0,1,2,3,4,8,10], 'name']
    expected = df1.loc[:, ['ID', 'age']]
    assert (result.loc[[5,6,7,9], 'age'] != df.loc[[5,6,7,9], 'name']).all()
    assert result.equals(expected), qp.diff(result, expected, returns='str')




@pytest.mark.parametrize("instructions, expected", [
    (
        r"""
        id ´v sort  ´c is any
        """,
        'ID'
    ),
    (
        r"""
        name ´v sort  ´c is any
        """,
        'name'
    ),
    (
        r"""
        id / name ´v sort  ´c is any
        """,
        ['ID', 'name']
    ),
    (
        r"""
        name / id ´v sort  ´c is any
        """,
        ['ID', 'name']
    ),
    ])
def test_sort(instructions, expected):
    df = qp.get_df()
    result = df.q(instructions)
    expected_df = df.sort_values(by=expected)
    assert result.equals(expected_df), qp.diff(result, expected_df, returns='str')


def test_to_int():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('age ´v to int')
    df2['age'] = [-25, 30, None, None, 40, None, None, None, None, None, 35]
    df2['age'] = df2['age'].astype('Int64')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_float():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age  ´v to float')
    df2['age'] = [-25.0, 30.0, None, None, 40.0, None, None, None, None, None, 35.0]
    df2['age'] = df2['age'].astype('Float64')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_num():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   ´v to num')
    df2['age'] = [-25, 30, np.nan, np.nan, 40, np.nan, np.nan, np.nan, np.nan, np.nan, 35]
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['age']].astype('object')
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_str():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   ´v to str')
    df2['age'] = ['-25', '30', 'nan', 'None', '40.0', 'forty-five', 'nan', 'unk', '', 'unknown', '35']
    df2['age'] = df2['age'].astype(str)
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_date():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('date of birth   ´v to date')
    df2['date of birth'] = [
        pd.to_datetime('1995-01-02', dayfirst=False).date(),
        pd.to_datetime('1990/09/14', dayfirst=False).date(),
        pd.to_datetime('1985.08.23', dayfirst=False).date(),
        pd.to_datetime('19800406', dayfirst=False).date(),
        pd.to_datetime('05-11-2007', dayfirst=True).date(),
        pd.to_datetime('06-30-1983', dayfirst=False).date(),
        pd.to_datetime('28-05-1975', dayfirst=True).date(),
        pd.NaT,
        pd.to_datetime('1955-Jan-09', dayfirst=False).date(),
        pd.to_datetime('1950 Sep 10', dayfirst=False).date(),
        pd.to_datetime('1945 October 11', dayfirst=False).date(),
        ]
    df2['date of birth'] = pd.to_datetime(df2['date of birth']).astype('datetime64[ns]')
    expected = df2.loc[:,['date of birth']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_na():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   ´v to na')
    df2['age'] = [-25, '30', None, None, '40.0', 'forty-five', None, 'unk', None, 'unknown', 35]
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_nk():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=age   ´v to nk')
    df2['age'] = [-25, '30', np.nan, None, '40.0', 'forty-five', 'nan', 'unknown', '', 'unknown', 35]
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['age']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_to_yn():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('=diabetes   ´v to yn')
    df2['diabetes'] = ['no', 'yes', None, 'no', 'yes', 'yes', 'no', None, None, 'no', 'yes']
    df2['age'] = df2['age'].astype('object')
    expected = df2.loc[:,['diabetes']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')



# @pytest.mark.parametrize("code, expected_cols", [
#     ('ID', ['ID']),
#     ])

# def test(code, expected_cols):
#     result = df.q(code)
#     expected = df.loc[:, expected_cols]
#     assert result.equals(expected), qp.diff(result, expected, returns='str')



@pytest.mark.parametrize("code, content, cols", [
    ('´n', '', ['new1']),
    ('´n a', 'a', ['new1']),
    ('´na', 'a', ['new1']),
    ('´n =a', 'a', ['new1']),
    ('´n= a', 'a', ['new1']),
    ('´n = a', 'a', ['new1']),
    ('´n ~ "a"', 'a', ['new1']),
    ('´n ~ df["ID"]', qp.get_df()['ID'], ['new1']),
    ('´n @ID', qp.get_df()['ID'].astype(str), ['new1']),
    ('´n ´c id', '', ['ID']),
    ])
def test_new_col(code, content, cols):
    df1 = get_df()
    result = df1.q(code)
    df2 = get_df()
    df2['new1'] = content
    expected = df2.loc[:, cols]
    assert result.equals(expected), qp.diff(result, expected, returns='str')

def test_new_col1():
    df1 = get_df()
    result = df1.q('´n a  ´h new col')
    df2 = get_df()
    df2['new col'] = 'a'
    expected = df2.loc[:,['new col']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_new_col2():
    df1 = get_df()
    result = df1.q('´n a & b ´c new1 / new2')
    df2 = get_df()
    df2['new1'] = 'a'
    df2['new2'] = 'b'
    expected = df2.loc[:,['new1', 'new2']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_new_col3():
    df1 = get_df()
    result = df1.q('´n a / b ´c new1 / new2')
    df2 = get_df()
    df2['new1'] = 'a'
    df2['new2'] = 'b'
    expected = df2.loc[:,['new1', 'new2']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_new_col4():
    df1 = get_df()
    result = df1.q('´r idx = 0  ´n a')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[[0],['new1']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_new_col5():
    df1 = get_df()
    result = df1.q('´n a  ´r idx = 0  ')

    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[[0],['new1']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_new_col6():
    df1 = get_df()
    result = df1.q('´r idx = 0  ´m save 1 ´r is any ´n a  ´r load 1')
    df2 = get_df()
    df2['new1'] = 'a'
    expected = df2.loc[[0],['new1']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')



def test_header_replace():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id ´h id')
    expected = df2.rename(columns={'ID': 'id'}).loc[:, ['id']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_header_replace1():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id ´h id  ´c name ´h n  ´c date of birth ´h dob ´c is any')
    expected = df2.rename(columns={'ID': 'id', 'name': 'n', 'date of birth': 'dob'})
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_header_append():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id ´h += abc')
    expected = df2.rename(columns={'ID': 'IDabc'}).loc[:, ['IDabc']]
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_header_append1():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id / name / date of birth  ´h += abc ´c is any')
    expected = df2.rename(columns={'ID': 'IDabc', 'name': 'nameabc', 'date of birth': 'date of birthabc'})
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_header_eval():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id ´h ~ x.lower() + str(len(x))  ´c is any')
    expected = df2.rename(columns={'ID': 'id2'})
    assert result.equals(expected), qp.diff(result, expected, returns='str')


def test_header_eval1():
    df1 = get_df()
    df2 = get_df()
    result = df1.q('id / weight / diabetes   ´h ~ x.lower() + str(len(x))  ´c is any')
    expected = df2.rename(columns={'ID': 'id2', 'weight': 'weight6', 'diabetes': 'diabetes8'})
    assert result.equals(expected), qp.diff(result, expected, returns='str')




