import os
import qplib as qp



def test_isdir(tmpdir):
    os.chdir(tmpdir)
    text = 'failed test checking existence of non existing directory'
    assert qp.isdir('dir1') is False, text


def test_mkdir_isdir(tmpdir):
    os.chdir(tmpdir)
    qp.mkdir('dir1')
    text = 'failed test checking existence of existing directory'
    assert qp.isdir('dir1') is True, text


def test_cd_pwd(tmpdir):
    os.chdir(tmpdir)
    qp.mkdir('dir1')
    qp.cd('dir1')
    path = qp.pwd()
    if '\\' in path:
        path = path.split('\\')[-1]
    elif '/' in path:  #pragma: no cover
        path = path.split('/')[-1]
    text = 'failed test for changing directory and finding path to current directory'
    assert path == 'dir1', text


def test_cd_return(tmpdir):
    os.chdir(tmpdir)
    qp.mkdir('dir1')
    qp.mkdir('dir1/dir2')
    qp.cd('dir1/dir2')
    result1 = qp.pwd()
    if '\\' in result1:
        result1 = result1.split('\\')[-1]
    elif '/' in result1:  #pragma: no cover
        result1 = result1.split('/')[-1]

    qp.cd('..')
    result2 = qp.pwd()
    if '\\' in result2:
        result2 = result2.split('\\')[-1]
    elif '/' in result2:  #pragma: no cover
        result2 = result2.split('/')[-1]

    text = 'failed test for going back and forth in directory structure'
    assert result1 == 'dir2' and result2 == 'dir1', text
