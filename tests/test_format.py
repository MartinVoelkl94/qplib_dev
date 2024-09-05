import pandas as pd
import qplib as qp



def test1():
    df = pd.DataFrame(columns=['a', 'b', 'c'])
    result = df.format(verbosity=0)
    expected = pd.DataFrame('', columns=['meta', 'a', 'b', 'c'], index=[])
    assert result.equals(expected), f'\nRESULT:\n{result}\nEXPECTED:\n{expected}'


def test2():
    df = pd.DataFrame(columns=['meta', 'a', 'b', 'c'], index=[])
    result = df.format(verbosity=0)
    expected = pd.DataFrame('', columns=['meta', 'a', 'b', 'c'], index=[])
    assert result.equals(expected), f'\nRESULT:\n{result}\nEXPECTED:\n{expected}'


def test3():
    df = pd.DataFrame(columns=[' a', 'b ', ' c ', 'a b c '])
    result = df.format(verbosity=0)
    expected = pd.DataFrame('', columns=['meta', 'a', 'b', 'c', 'a b c'], index=[])
    assert result.equals(expected), f'\nRESULT:\n{result}\nEXPECTED:\n{expected}'

