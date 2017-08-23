import tensorflow as tf
from heapq import heappush, heappop
from construct import vecVal
from main import inlocal
from db.pick import openFile
import collections
import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Lmax = 50

tree = openFile("tree", inlocal)
imagesInLeaves = openFile("imagesInLeaves", inlocal)
nodes = openFile("nodes", inlocal)


def hamming2(s1, s2):
    # assert len(s1) == len(s2)
    r = (1 << np.arange(8))[:, None]
    # return np.count_nonzero((s1 & r) != (s2 & r))
    return np.count_nonzero((np.bitwise_xor(s1, s2) & r) != 0)


class searchTree(object):
    result = []
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

        try:
            # result.append(imagesInLeaves[N]) ?
            for i in imagesInLeaves[N]:
                val = vecVal(i)
                cost = hamming2(val, Q)
                heappush(self.result, (cost, i))
                self.length += 1

        except KeyError:
            C = tree[N]
            # if len(C) > 0:
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

        except Exception as e:
            raise e


class similarImages():
    def __init__(self):
        self.total = []

    def add_results(self, result, top_n=5):
        for _ in range(top_n):
            top_result = heappop(result)
            self.total.append(top_result[1][0].name)

    def similar_result(self, top_n=5):
        sim_imgs = collections.Counter(self.total).most_common(top_n)
        return sim_imgs
