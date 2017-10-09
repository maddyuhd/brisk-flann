from heapq import heappush, heappop
from features.info import inlocal
from db.pick import openFile
from tree.numpy_help import hamming2
from tree.indexer import getVal
exitFlag = 0


def loaddb():
    global tree, imagesInLeaves, nodes, remove
    tree = openFile("tree", inlocal)
    imagesInLeaves = openFile("imagesInLeaves", inlocal)
    nodes = openFile("nodes", inlocal)
    remove = openFile("remove", inlocal)
    disable = openFile("disable", inlocal)
    remove = remove | disable


class searchMe(object):
    Lmax = 50

    def __init__(self, obj, T, Q):
        self._length = 0
        self._PQ, self.result = [], []
        self._process(obj, T, Q)

    def _process(self, obj, T, Q):

        self._traverseTree(obj, T, Q)  # ll here
        while len(self._PQ) > 0 and self._length < self.Lmax:
            try:
                N = heappop(self._PQ)[1]
            except Exception as e:
                raise e
            self._traverseTree(obj, N, Q)

    def _traverseTree(self, obj, N, Q):
        global imagesInLeaves, tree, nodes

        if N in imagesInLeaves:
            for i in imagesInLeaves[N]:
                val = getVal(i)
                cost = hamming2(val, Q)
                heappush(self.result, (cost, i))
                self._length += 1

        elif N in tree:
            C = tree[N]

            data = obj.search(C, nodes, Q, heappush)

            Cq = heappop(data)[1]

            for i in data:
                heappush(self._PQ, i)

            self._traverseTree(obj, Cq, Q)

        else:
            print "bla...bla..."


def searchll(threadName, des, resultObj, tfObj):
    for vec in des:
        if exitFlag:
            threadName.exit()

        searchObj = searchMe(tfObj, 0, vec)
        resultObj.update(searchObj.result)
