import threading
from features.searcher import searchll
from image.numpy_help import chunkify
from tensor_flow.pre_cluster import tfInit


class myThread (threading.Thread):
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
    tfObj = tfInit(n_clusters)
    tfObj.finalVariable()


def llProcess(data, n_threads, n_clusters, result):
    global resultObj
    resultObj = result

    loadTf(n_clusters)

    chunks = chunkify(data, n_threads)

    bucket = []
    for idx, data in enumerate(chunks):
        thread = myThread(idx, "Thread-" + str(idx), data)
        bucket.append(thread)

    for thread in bucket:
        thread.start()
    for thread in bucket:
        thread.join()
