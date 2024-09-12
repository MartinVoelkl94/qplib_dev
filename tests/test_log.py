
import pytest
import datetime
import qplib as qp
from qplib import log



def test_log():
    log(clear=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log('test')
    logs = qp.util.logs
    last = len(logs) - 1
    result_text = logs.loc[last, 'text']
    result_context = logs.loc[last, 'context']
    result_level = logs.loc[last, 'level']
    result_time = logs.loc[last, 'time'].strftime('%Y-%m-%d %H:%M:%S')
    assert result_text == 'test', f'expected: "test", got: "{result_text}"'
    assert result_context == '', f'expected: "", got: "{result_context}"'
    assert result_level == 'info', f'expected: "info", got: "{result_level}"'
    assert result_time == now, f'expected: "{now}", got: "{result_time}"'



@pytest.mark.parametrize('input, expected', [
    ('test', 'info'),
    ('error: test', 'ERROR'),
    ('warning: test', 'Warning'),
    ('info: test', 'info'),
    ('debug: test', 'debug'),
    ('trace: test', 'trace'),
    ])
def test_log_levels(input, expected):
    log(clear=True)
    log(input)
    logs = qp.util.logs
    result = logs.loc[len(logs) - 1, 'level']
    assert result == expected, f'expected: {expected}, got: "{result}"'

