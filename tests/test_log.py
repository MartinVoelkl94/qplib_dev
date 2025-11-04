import pytest
import datetime
from qplib import log
from freezegun import freeze_time



@freeze_time("2025-11-04 09:33:40.390551")
def test_log():
    log(clear=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log('test')
    logs = log()
    last = len(logs) - 1
    result_text = logs.loc[last, 'text']
    result_context = logs.loc[last, 'context']
    result_level = logs.loc[last, 'level']
    result_time = logs.loc[last, 'time'].strftime('%Y-%m-%d %H:%M:%S')
    assert result_text == 'test', f'expected: "test", got: "{result_text}"'
    assert result_context == '', f'expected: "", got: "{result_context}"'
    assert result_level == 'INFO', f'expected: "INFO", got: "{result_level}"'
    assert result_time == now, f'expected: "{now}", got: "{result_time}"'



@pytest.mark.parametrize('input, expected', [
    ('test', 'INFO'),
    ('error: test', 'ERROR'),
    ('warning: test', 'WARNING'),
    ('info: test', 'INFO'),
    ('debug: test', 'DEBUG'),
    ('trace: test', 'TRACE'),
    ])
def test_log_levels(input, expected):
    log(clear=True)
    log(input)
    logs = log()
    result = logs.loc[len(logs) - 1, 'level']
    assert result == expected, f'expected: {expected}, got: "{result}"'
