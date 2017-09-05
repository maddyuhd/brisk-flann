from random import sample
import os
import tensorflow as tf
from info import debug
from db.pick import saveFile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def randomeCentroid(K, N):
    return tuple(sample(xrange(0, K), N))


def vecVal(val):
    obj, idx = val
    return obj.des[idx]


class constructMe():

    def __init__(self, node, vectors, tfObj):

        self.tree = {}
        self.nodes = {}
        self.imagesInLeaves = {}
        self.nodeIndex = 0
        self.disable, self.remove = set(), set()

        if debug:
            from progress_bar.progress import progress
            self.bar = progress("Constructing", len(vectors))

        self.process(node, vectors, tfObj)

        if debug:
            self.bar.finish()

    def process(self, node, vectors, tfObj):
        self.tree[node] = []

        if (len(vectors) < tfObj.max_size_lev):
            self.imagesInLeaves[node] = []

            for idx, v in enumerate(vectors):
                self.imagesInLeaves[node].append(v)
                # v[0].updateLeaf((node, idx))

                if debug:
                    self.bar.update()

        else:
            pickedVal = randomeCentroid(len(vectors), tfObj.n_clusters)

            childIDs = [[] for i in range(tfObj.n_clusters)]
            centroidsVal = [vecVal(vectors[i])
                            for i in pickedVal]

            with tf.Session(graph=tfObj.graph) as sess:
                with tf.device("/gpu:0"):
                    # clustering...
                    for vector_n in range(len(vectors)):
                        vect = vecVal(vectors[vector_n])

                        distances = [sess.run(tfObj.euclid_dist, feed_dict={
                            tfObj.v1: vect, tfObj.v2: centroid})
                            for centroid in centroidsVal]

                        assignments_val = sess.run(
                            tfObj.cluster_assignment, feed_dict={
                                tfObj.centroidPh: distances})

                        childIDs[assignments_val].append(vectors[vector_n])

                    for i in range(tfObj.n_clusters):
                        self.nodeIndex += 1
                        self.nodes[self.nodeIndex] = vectors[pickedVal[i]]
                        self.tree[node].append(self.nodeIndex)
                        self.process(self.nodeIndex, childIDs[i], tfObj)

    def saveDb(self, inlocal):
        saveFile(self.tree, "tree", inlocal)
        saveFile(self.imagesInLeaves, "imagesInLeaves", inlocal)
        saveFile(self.nodes, "nodes", inlocal)
        saveFile(self.nodeIndex, "nodeIndex", inlocal)
        saveFile(self.disable, "disable", inlocal)
        saveFile(self.remove, "remove", inlocal)
