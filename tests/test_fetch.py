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

dates = [today1, today2, yesterday, last_week, last_month, last_year, old_date]


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

    for date in dates:
        with open(f'archive/date_{date}.txt', 'w') as f:
            f.write(date)

setup()



def test_most_recent():
    setup()
    file = qp.fetch(f'archive/date_')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today1)
    assert result == expected, f'failed test for loading most recent file.\nResult: {result}\nexpected: {expected}'




def test_most_recent_datefmt():
    setup()
    file = qp.fetch(f'archive/date_')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today2)
    assert result == expected, f'failed test for loading most recent file with different date format.\nResult: {result}\nexpected: {expected}'


def test_before_date():
    setup()
    file = qp.fetch(f'archive/date_', before='2000_01_02')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(old_date)
    assert result == expected, f'failed test for loading most recent file from before 2000_01_02.\nResult: {result}\nexpected: {expected}'


def test_before_this_year():
    setup()

    if qp.date(yesterday).year < qp.date(today).year and qp.date(yesterday) > qp.date(last_year):
        date_correct = yesterday
    elif qp.date(last_week).year < qp.date(today).year and qp.date(last_week) > qp.date(last_year):
        date_correct = last_week
    elif qp.date(last_month).year < qp.date(today).year and qp.date(last_month) > qp.date(last_year):
        date_correct = last_month
    else:
        date_correct = last_year
       
    file = qp.fetch(f'archive/date_', before='this year')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(date_correct)
    assert result == expected, f'failed test for loading most recent file from before this year.\nResult: {result}\nexpected: {expected}'


def test_before_this_month():
    setup()
    
    if qp.date(yesterday).year < qp.date(today).year and qp.date(yesterday) > qp.date(last_year):
        date_correct = yesterday
    elif qp.date(last_week).year < qp.date(today).year and qp.date(last_week) > qp.date(last_year):
        date_correct = last_week
    else:
        date_correct = last_month
       
    file = qp.fetch(f'archive/date_', before='this month')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(date_correct)
    assert result == expected, f'failed test for loading most recent file from before this month.\nResult: {result}\nexpected: {expected}'


def test_before_this_week():
    setup()
    
    if qp.date(yesterday).year < qp.date(today).year and qp.date(yesterday) > qp.date(last_year):
        date_correct = yesterday
    else:
        date_correct = last_week
       
    file = qp.fetch(f'archive/date_', before='this week')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(date_correct)
    assert result == expected, f'failed test for loading most recent file from before this week.\nResult: {result}\nexpected: {expected}'

 

def test_before_this_day():
    setup()    

    file = qp.fetch(f'archive/date_', before='this day')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(yesterday)
    assert result == expected, f'failed test for loading most recent file from before this day.\nResult: {result}\nexpected: {expected}'


def test_before_today():
    setup()     

    file = qp.fetch(f'archive/date_', before='today')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(yesterday)
    assert result == expected, f'failed test for loading most recent file from before today.\nResult: {result}\nexpected: {expected}'




