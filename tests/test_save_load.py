import datetime
import os
import shutil
import pandas as pd
import qplib as qp


#remove test data
def clean():
    if os.path.isdir('archive'):
        shutil.rmtree('archive')
    if os.path.isdir('test'):
        shutil.rmtree('test')
    os.mkdir('archive')
    os.mkdir('test')
clean()




#create test data

df1 = pd.DataFrame({'a':[1]})
df2 = pd.DataFrame({'a':[2]})
df3 = pd.DataFrame({'a':[3]})
df4 = pd.DataFrame({'a':[4]})

today1 = datetime.datetime.now().strftime('%Y_%m_%d')
today2 = datetime.datetime.now().strftime('%d_%m_%Y')
date0 = qp.date('2000_01_01').strftime('%Y_%m_%d')
date1 = (datetime.datetime.now() - datetime.timedelta(days=400)).date().strftime('%Y_%m_%d')
date2 = (datetime.datetime.now() - datetime.timedelta(days=40)).date().strftime('%Y_%m_%d')
date3 = (datetime.datetime.now() - datetime.timedelta(days=8)).date().strftime('%Y_%m_%d')
date4 = (datetime.datetime.now() - datetime.timedelta(days=1)).date().strftime('%Y_%m_%d')

def setup(): 
    pd.DataFrame({'a':[today1]}).to_excel(f'archive/df_{today1}.xlsx', index=False)
    pd.DataFrame({'a':[today2]}).to_excel(f'archive/df_{today2}.xlsx', index=False)
    pd.DataFrame({'a':[date0]}).to_excel(f'archive/df_{date0}.xlsx', index=False)
    pd.DataFrame({'a':[date1]}).to_excel(f'archive/df_{date1}.xlsx', index=False)
    pd.DataFrame({'a':[date2]}).to_excel(f'archive/df_{date2}.xlsx', index=False)
    pd.DataFrame({'a':[date3]}).to_excel(f'archive/df_{date3}.xlsx', index=False)
    pd.DataFrame({'a':[date4]}).to_excel(f'archive/df_{date4}.xlsx', index=False)



def test_default():
    clean()
    df1.save()
    assert qp.isfile('df.xlsx'), 'failed test for default saving behaviour'
    df1a = qp.load('df.xlsx')
    assert df1a.equals(df1), 'failed test for loading from default file'


def test_specific():
    clean()
    df1.save('df1.xlsx')
    df1b = qp.load('df1.xlsx')
    assert qp.isfile('df1.xlsx'), 'failed test for saving to a specific file'
    assert df1b.equals(df1), 'failed test for loading from a specific file'


def test_folder():
    clean()
    df1.save('test/df1.xlsx')
    df1c = qp.load('test/df1.xlsx')
    assert qp.isfile('test/df1.xlsx'), 'failed test for saving to a specific file in folder'
    assert df1c.equals(df1), 'failed test for loading from a specific file in folder'


def test_sheet():
    clean()
    df1.save('df1.xlsx', sheet='data2')
    df1d = qp.load('df1.xlsx', sheet='data2')
    assert df1d.equals(df1), 'failed test for saving and loading to and from a specific sheet'


def test_overwriting_sheets():
    clean()
    df1.save('df1.xlsx')
    df1old = qp.load('df1.xlsx')
    df2.save('df1.xlsx')
    df1new = qp.load('df1.xlsx')
    assert df1old.loc[0, 'a'] == 1 and df1new.loc[0, 'a'] == 2, 'failed test for overwriting sheets'


def test_archiving():
    clean()
    today = datetime.datetime.now().strftime('%Y_%m_%d')
    df1.save('df1')
    assert qp.isfile(f'archive/df1_{today}.xlsx'), 'failed test for archiving file'


def test_nested_folder():
    clean()
    os.mkdir('test/archive')
    today = datetime.datetime.now().strftime('%Y_%m_%d')
    df1.save('test/df1')
    assert qp.isfile(f'test/archive/df1_{today}.xlsx'), 'failed test for archiving in nested folder'


def test_datefmt():
    clean()
    today = datetime.datetime.now().strftime('%d_%m_%Y')
    df1.save('test/df1.xlsx', sheet='data1', datefmt='%d_%m_%Y')
    assert qp.isfile(f'test/archive/df1_{today}.xlsx'), 'failed test for archiving with different date format'


def test_most_recent():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', index=False).loc[0, 'a'] == today1, 'failed test for loading most recent file'
    assert qp.load('archive/df', sheet='Sheet1', before='now', index=False).loc[0, 'a'] == today1, 'failed test for loading most recent file explicitly'



def test_most_recent_datefmt():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', index=False).loc[0, 'a'] == today2, 'failed test for loading most recent file with different date format'
    assert qp.load('archive/df', sheet='Sheet1', before='now', index=False).loc[0, 'a'] == today2, 'failed test for loading most recent file with different date format explicitly'


def test_before_date():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='2000_01_02', index=False).loc[0, 'a'] == date0, 'failed test for loading most recent file from before specific date'


def test_before_this_year():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this year', index=False).loc[0, 'a'] == date1, 'failed test for loading most recent file from before this year'


def test_before_this_month():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this month', index=False).loc[0, 'a'] == date2, 'failed test for loading most recent file from before this month'


def test_before_this_week():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this week', index=False).loc[0, 'a'] == date3, 'failed test for loading most recent file from before this week'


def test_before_this_day():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this day', index=False).loc[0, 'a'] == date4, 'failed test for loading most recent file from before this day'


def test_before_today():
    clean()
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='today', index=False).loc[0, 'a'] == date4, 'failed test for loading most recent file from before this day'





