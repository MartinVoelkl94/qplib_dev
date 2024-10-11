import os
import shutil
import qplib as qp



#prepare testing environment

def setup():

    current_folder = os.path.basename(os.getcwd())
    
    if current_folder != 'tests_temp_pn75Nv9H9p81Xul':
        if os.path.exists('tests_temp_pn75Nv9H9p81Xul'):
            shutil.rmtree('tests_temp_pn75Nv9H9p81Xul')
        
        os.mkdir('tests_temp_pn75Nv9H9p81Xul')
        os.chdir('tests_temp_pn75Nv9H9p81Xul')

setup()




def test_isdir():
    setup()
    assert qp.isdir('dir1') == False, 'failed test checking existence of non existing directory'


def test_mkdir_isdir():
    setup()
    qp.mkdir('dir1')
    assert qp.isdir('dir1') == True, 'failed test checking existence of existing directory'


def test_cd_pwd():
    setup()
    qp.mkdir('dir1')
    qp.cd('dir1')
    path = qp.pwd()
    if '\\' in path:
        path = path.split('\\')[-1]
    elif '/' in path:
        path = path.split('/')[-1]
    assert path == 'dir1', 'failed test for changing directory and finding path to current directory'


def test_cd_return():
    setup()
    qp.mkdir('dir1')
    qp.mkdir('dir1/dir2')
    qp.cd('dir1/dir2')
    result1 = qp.pwd()
    if '\\' in result1:
        result1 = result1.split('\\')[-1]
    elif '/' in result1:
        result1 = result1.split('/')[-1]


    qp.cd('..')
    result2 = qp.pwd()
    if '\\' in result2:
        result2 = result2.split('\\')[-1]
    elif '/' in result2:
        result2 = result2.split('/')[-1]

    assert result1 == 'dir2' and result2 == 'dir1', 'failed test for going back and forth in directory structure'




