import pandas as pd
import numpy as np
import copy
import os
import sys
import shutil
import datetime
import qplib as qp
from qplib import log
from IPython.display import display


def compare(result, expected_df, text=None):
    if not result.equals(expected_df):
        print('result:')
        display(result)
        print('expected:')
        display(expected_df)
        if text is None:
            raise Exception(f'failed test for arg: "{result.qp.code}"')
        else:
            raise Exception(f'failed test for: {text}')
    else:
        if text is None:
            print(f'passed test for arg: "{result.qp.code}"')
        else:
            print(f'passed test for: {text}')


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



