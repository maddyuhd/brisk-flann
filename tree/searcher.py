import tensorflow as tf
import collections
from heapq import heappush, heappop
from construct import vecVal
from features.info import inlocal
from db.pick import openFile
from image.numpy_help import hamming2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
exitFlag = 0


def loaddb():
    global tree, imagesInLeaves, nodes, remove
    tree = openFile("tree", inlocal)
    imagesInLeaves = openFile("imagesInLeaves", inlocal)
    nodes = openFile("nodes", inlocal)
    remove = openFile("remove", inlocal)
    disable = openFile("disable", inlocal)
    remove = remove | disable


# def hamming2(s1, s2):
#     r = (1 << np.arange(8))[:, None]
#     return np.count_nonzero((np.bitwise_xor(s1, s2) & r) != 0)


class searchMe(object):
    Lmax = 50

    def __init__(self, obj, T, Q):
        self.length = 0
        self.PQ, self.result = [], []
        self.process(obj, T, Q)

    def process(self, obj, T, Q):

        self.traverseTree(obj, T, Q)  # ll here
        while (len(self.PQ) > 0 and self.length < self.Lmax):
            try:
                N = heappop(self.PQ)[1]
            except Exception as e:
                raise e
            self.traverseTree(obj, N, Q)

    def traverseTree(self, obj, N, Q):
        global imagesInLeaves, tree, nodes

        if N in imagesInLeaves:
            for i in imagesInLeaves[N]:
                val = vecVal(i)
                cost = hamming2(val, Q)
                heappush(self.result, (cost, i))
                self.length += 1

        elif N in tree:
            C = tree[N]

            with tf.Session(graph=obj.graph) as sess:
                with tf.device("/gpu:0"):
                    data = []
                    for i in C:
                        vect = vecVal(nodes[i])
                        distances = sess.run(obj.euclid_dist, feed_dict={
                            obj.v1: vect, obj.v2: Q})
                        heappush(data, (distances, i))

            Cq = heappop(data)[1]

            for i in data:
                heappush(self.PQ, i)

            self.traverseTree(obj, Cq, Q)

        else:
            print "bla...bla..."


class analyse():
    def __init__(self):
        self.c = collections.Counter()
        # self.setVal = []

    def update(self, result, top_n=5):
        for _ in range(top_n):
            top_result = heappop(result)
            self.c.update([top_result[1][0]])
            # self.c.update([top_result[1][0].name])

    """
    Clean me
    """
    def output(self, qdes, top_n=5):
        self.c = self.c.most_common(top_n)

        h = temp()
        arr = []

        for idx, i in enumerate(self.c):
            i, _ = i
            count = h.match(qdes, i.des)
            arr.append((i.name, count))

        import heapq
        arr = heapq.nlargest(5, arr, key=lambda x: x[1])

        if arr[0][1] >= 5:
            # print arr
            arr = arr[0][0]
            return (1, arr)
        else:
            return (0, arr)

        # setVal = [x for x in self.c if x[1] >= 10 and not in remove] <

"""
Clean me
"""
class temp(object):
    def __init__(self):
        import cv2

        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2

        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(
            index_params, search_params)

    def match(self, qdes, sdes):
        count = 0
        matches = self.flann.knnMatch(qdes, sdes, k=2)
        # ratio test as per Lowe's paper
        for i, val in enumerate(matches):
            if len(val) == 2:
                m, n = val[0], val[1]
                if m.distance < 0.7 * n.distance:
                    count += 1
        return count


def searchll(threadName, des, resultObj, tfObj):
    for vec in des:
        if exitFlag:
            threadName.exit()

        tree = searchMe(tfObj, 0, vec)
        resultObj.update(tree.result)
