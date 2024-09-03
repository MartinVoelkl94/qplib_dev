import datetime
import os
import shutil
import pandas as pd
import qplib as qp

#create test data

today = datetime.datetime.now()
today1 = today.strftime('%Y_%m_%d')
today2 = today.strftime('%d_%m_%Y')
yesterday = (today - datetime.timedelta(days=1)).strftime('%Y_%m_%d')
last_week = (today - datetime.timedelta(days=today.weekday()+1)).strftime('%Y_%m_%d')
last_month = (today - datetime.timedelta(days=today.day)).strftime('%Y_%m_%d')
last_year = (today - datetime.timedelta(days=365)).strftime('%Y_%m_%d')
old_date = qp.date('2000_01_01').strftime('%Y_%m_%d')

df_today1 = pd.DataFrame({'a':[today1]})
df_today2 = pd.DataFrame({'a':[today2]})
df_yesterday = pd.DataFrame({'a':[yesterday]})
df_last_week = pd.DataFrame({'a':[last_week]})
df_last_month = pd.DataFrame({'a':[last_month]})
df_last_year = pd.DataFrame({'a':[last_year]})
df_old_date = pd.DataFrame({'a':[old_date]})



#prepare testing environment

def setup():

    current_folder = os.path.basename(os.getcwd())
    
    if current_folder != 'tests_temp_pn75Nv9H9p81Xul':
        if os.path.exists('tests_temp_pn75Nv9H9p81Xul'):
            shutil.rmtree('tests_temp_pn75Nv9H9p81Xul')
        
        os.mkdir('tests_temp_pn75Nv9H9p81Xul')
        os.chdir('tests_temp_pn75Nv9H9p81Xul')

    if os.path.exists('archive'):
        shutil.rmtree('archive')
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    if os.path.exists('df.xlsx'):
        os.remove('df.xlsx')
    
    os.mkdir('archive')
    os.mkdir('temp')
    os.mkdir('temp/archive')

    pd.DataFrame(df_today1).to_excel(f'archive/df_{today1}.xlsx', index=False)
    pd.DataFrame(df_today2).to_excel(f'archive/df_{today2}.xlsx', index=False)
    pd.DataFrame(df_yesterday).to_excel(f'archive/df_{yesterday}.xlsx', index=False)
    pd.DataFrame(df_last_week).to_excel(f'archive/df_{last_week}.xlsx', index=False)
    pd.DataFrame(df_last_month).to_excel(f'archive/df_{last_month}.xlsx', index=False)
    pd.DataFrame(df_last_year).to_excel(f'archive/df_{last_year}.xlsx', index=False)
    pd.DataFrame(df_old_date).to_excel(f'archive/df_{old_date}.xlsx', index=False)

setup()







def test_default():
    setup()
    df_today1.save()
    assert qp.isfile('df.xlsx'), 'failed test for default saving behaviour'
    df_today1a = qp.load('df.xlsx')
    assert df_today1a.equals(df_today1), 'failed test for loading from default file'


def test_specific():
    setup()
    df_today1.save('df_today1.xlsx')
    df_today1b = qp.load('df_today1.xlsx')
    assert qp.isfile('df_today1.xlsx'), 'failed test for saving to a specific file'
    assert df_today1b.equals(df_today1), 'failed test for loading from a specific file'


def test_folder():
    setup()
    df_today1.save('temp/df_today1.xlsx')
    df_today1c = qp.load('temp/df_today1.xlsx')
    assert qp.isfile('temp/df_today1.xlsx'), 'failed test for saving to a specific file in folder'
    assert df_today1c.equals(df_today1), 'failed test for loading from a specific file in folder'


def test_sheet():
    setup()
    df_today1.save('df_today1.xlsx', sheet='data2')
    df_today1d = qp.load('df_today1.xlsx', sheet='data2')
    assert df_today1d.equals(df_today1), 'failed test for saving and loading to and from a specific sheet'


def test_overwriting_sheets():
    setup()
    df_today1.save('df_today1.xlsx')
    df_today1old = qp.load('df_today1.xlsx')
    df_today2.save('df_today1.xlsx')
    df_today1new = qp.load('df_today1.xlsx')
    assert df_today1old.loc[0, 'a'] == today1 and df_today1new.loc[0, 'a'] == today2, 'failed test for overwriting sheets'


def test_archiving():
    setup()
    df_today1.save('df_today1')
    assert qp.isfile(f'archive/df_today1_{today1}.xlsx'), 'failed test for archiving file'


def test_nested_folder():
    setup()
    df_today1.save('temp/df_today1')
    assert qp.isfile(f'temp/archive/df_today1_{today1}.xlsx'), 'failed test for archiving in nested folder'


def test_datefmt():
    setup()
    df_today1.save('temp/df_today1.xlsx', sheet='data1', datefmt='%d_%m_%Y')
    assert qp.isfile(f'temp/archive/df_today1_{today2}.xlsx'), 'failed test for archiving with different date format'


def test_most_recent():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', index=False).loc[0, 'a'] == today1, 'failed test for loading most recent file'
    assert qp.load('archive/df', sheet='Sheet1', before='now', index=False).loc[0, 'a'] == today1, 'failed test for loading most recent file explicitly'



def test_most_recent_datefmt():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', index=False).loc[0, 'a'] == today1, 'failed test for loading most recent file with different date format'
    assert qp.load('archive/df', sheet='Sheet1', before='now', index=False).loc[0, 'a'] == today1, 'failed test for loading most recent file with different date format explicitly'


def test_before_date():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='2000_01_02', index=False).loc[0, 'a'] == old_date, 'failed test for loading most recent file from before specific date'


def test_before_this_year():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this year', index=False).loc[0, 'a'] == last_year, 'failed test for loading most recent file from before this year'


def test_before_this_month():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this month', index=False).loc[0, 'a'] == last_month, 'failed test for loading most recent file from before this month'


def test_before_this_week():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this week', index=False).loc[0, 'a'] == last_week, 'failed test for loading most recent file from before this week'


def test_before_this_day():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='this day', index=False).loc[0, 'a'] == yesterday, 'failed test for loading most recent file from before this day (==today)'


def test_before_today():
    setup()
    assert qp.load('archive/df', sheet='Sheet1', before='today', index=False).loc[0, 'a'] == yesterday, 'failed test for loading most recent file from before today (==this day)'




