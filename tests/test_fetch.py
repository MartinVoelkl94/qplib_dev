import datetime
import os
import shutil
import pytest
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

def setup(tmpdir):
 
    os.chdir(tmpdir)  
    os.mkdir('archive')
    os.mkdir('temp')
    os.mkdir('temp/archive')

    for date in dates:
        with open(f'archive/date_{date}.txt', 'w') as f:
            f.write(date)

    with open(f'archive/date_123.txt', 'w') as f:
        f.write('incorrect date')



def test_no_timestamps(tmpdir):
    # setup(tmpdir)
    with pytest.raises(FileNotFoundError):
        file = qp.fetch(f'date_')


def test_fetch(tmpdir):
    setup(tmpdir)
    file = qp.fetch(f'archive/date_{today1}.txt')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today1)
    assert result == expected, f'failed test for loading most recent file.\nResult: {result}\nexpected: {expected}'


def test_fetch_in_root(tmpdir):
    setup(tmpdir)
    qp.cd('archive')
    file = qp.fetch(f'date_{today1}.txt')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today1)
    assert result == expected, f'failed test for loading most recent file.\nResult: {result}\nexpected: {expected}'


def test_most_recent(tmpdir):
    setup(tmpdir)
    file = qp.fetch(f'archive/date_')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today1)
    assert result == expected, f'failed test for loading most recent file.\nResult: {result}\nexpected: {expected}'


def test_most_recent_in_root(tmpdir):
    setup(tmpdir)
    qp.cd('archive')
    file = qp.fetch(f'date_')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today1)
    assert result == expected, f'failed test for loading most recent file.\nResult: {result}\nexpected: {expected}'



def test_most_recent_datefmt(tmpdir):
    setup(tmpdir)
    file = qp.fetch(f'archive/date_')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(today2)
    assert result == expected, f'failed test for loading most recent file with different date format.\nResult: {result}\nexpected: {expected}'


def test_before_date(tmpdir):
    setup(tmpdir)
    file = qp.fetch(f'archive/date_', before='2000_01_02')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(old_date)
    assert result == expected, f'failed test for loading most recent file from before 2000_01_02.\nResult: {result}\nexpected: {expected}'


def test_before_this_year(tmpdir):
    setup(tmpdir)

    if qp.date(yesterday).year < qp.date(today).year and qp.date(yesterday) > qp.date(last_year):
        date_correct = yesterday #pragma: no cover
    elif qp.date(last_week).year < qp.date(today).year and qp.date(last_week) > qp.date(last_year):
        date_correct = last_week #pragma: no cover
    elif qp.date(last_month).year < qp.date(today).year and qp.date(last_month) > qp.date(last_year):
        date_correct = last_month #pragma: no cover
    else:
        date_correct = last_year #pragma: no cover
       
    file = qp.fetch(f'archive/date_', before='this year')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(date_correct)
    assert result == expected, f'failed test for loading most recent file from before this year.\nResult: {result}\nexpected: {expected}'


def test_before_this_month(tmpdir):
    setup(tmpdir)
    
    if qp.date(yesterday).year < qp.date(today).year and qp.date(yesterday) > qp.date(last_year):
        date_correct = yesterday #pragma: no cover
    elif qp.date(last_week).year < qp.date(today).year and qp.date(last_week) > qp.date(last_year):
        date_correct = last_week #pragma: no cover
    else:
        date_correct = last_month #pragma: no cover
       
    file = qp.fetch(f'archive/date_', before='this month')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(date_correct)
    assert result == expected, f'failed test for loading most recent file from before this month.\nResult: {result}\nexpected: {expected}'


def test_before_this_week(tmpdir):
    setup(tmpdir)
    
    if qp.date(yesterday).year < qp.date(today).year and qp.date(yesterday) > qp.date(last_year):
        date_correct = yesterday #pragma: no cover
    else:
        date_correct = last_week
       
    file = qp.fetch(f'archive/date_', before='this week')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(date_correct)
    assert result == expected, f'failed test for loading most recent file from before this week.\nResult: {result}\nexpected: {expected}'

 

def test_before_this_day(tmpdir):
    setup(tmpdir)    

    file = qp.fetch(f'archive/date_', before='this day')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(yesterday)
    assert result == expected, f'failed test for loading most recent file from before this day.\nResult: {result}\nexpected: {expected}'


def test_before_today(tmpdir):
    setup(tmpdir)     

    file = qp.fetch(f'archive/date_', before='today')
    with open(file, 'r') as f:
        result = qp.date(f.read())
        expected = qp.date(yesterday)
    assert result == expected, f'failed test for loading most recent file from before today.\nResult: {result}\nexpected: {expected}'




