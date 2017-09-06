import threading
from features.searcher import searchll

from tensor_flow.pre_cluster import tfInit
# from features.info import n_clusters


def loadTf(n_clusters):
    global tfObj
    tfObj = tfInit(n_clusters)
    tfObj.finalVariable()


class myThread (threading.Thread):
    def __init__(self, threadID, name, data, resultObj):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data = data
        self.resultObj = resultObj

    def run(self):
        # print "Starting " + self.name
        searchll(self.name, self.data, self.resultObj, tfObj)
        # print "Exiting " + self.name
