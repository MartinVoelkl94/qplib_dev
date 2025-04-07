import qplib as qp
import pytest



@pytest.mark.parametrize('patterns, value, expected', [
    (1, 1, True),
    (1, 2, False),
    (2, 1, False),

    ('a', 1, False),
    (None, 1, False),

    ([1,2,3], 1, True),
    ([1,2,3], 4, False),

    ('.', 'a', True),
    ('...', 'abc', True),
    ('..', 'abc', False),
    ('..', 'a', False),

    (['a', '..'], 'a', True),
    (['a', '..'], 'ab', True),
    (['a', '..'], 'abc', False),
    ])
def test(patterns, value, expected):
    assert qp.match(patterns, value) == expected
