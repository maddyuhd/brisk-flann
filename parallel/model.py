import threading
from tree.searcher import searchll
from tree.numpy_help import chunkify
from tensor_flow.pre_cluster import tfInit
from tree.temp import analyse

class myThread(threading.Thread):
    def __init__(self, threadID, name, data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data = data

    def run(self):
        # print "Starting " + self.name
        searchll(self.name, self.data, resultObj, tfObj)
        # print "Exiting " + self.name


def loadTf(n_clusters):
    global tfObj
    tfObj = tfInit(n_clusters, inSearchMode=True)


def llProcess(data, n_threads, n_clusters):
    global resultObj
    resultObj = analyse()

    loadTf(n_clusters)

    chunks = chunkify(data, n_threads)

    bucket = []
    for idx, des in enumerate(chunks):
        thread = myThread(idx, "Thread-" + str(idx), des)
        bucket.append(thread)

    for thread in bucket:
        thread.start()
    for thread in bucket:
        thread.join()

    result = resultObj.output(data)
    return result
