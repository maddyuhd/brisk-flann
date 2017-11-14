from features.info import inlocal, success, failed
from log.log import logInfo
from db.pick import openFile, saveFile
from db.in_memory import Manage_del_dis


remove = "0001"
disable = "0002"

obj = Manage_del_dis()

def test_disable(benchmark):
    responds = benchmark(obj.disable, disable)
    assert responds == 1 or responds == 0

def test_remove(benchmark):
    responds = benchmark(obj.remove, remove)
    assert responds == 1 or responds == 0
