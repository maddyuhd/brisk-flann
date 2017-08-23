from random import sample
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def randomeCentroid(K, N):
    return tuple(sample(xrange(0, K), N))


def vecVal(val):
    obj, idx = val
    return obj.des[idx]


class constructTree():

    def __init__(self, node, vectors, tfObj, debug=True):

        self.tree = {}  # tree structure eg tree[0]=[1,2,3]
        self.nodes = {}  # centroid values 32d [] of the tree
        self.imagesInLeaves = {}  # leaf node
        self.nodeIndex = 0
        self.debug = debug
        if self.debug:
            from progress_bar.progress import progress
            self.bar = progress("Constructing", len(vectors))

        self.process(node, vectors, tfObj)
        if self.debug:
            self.bar.finish()

    def process(self, node, vectors, tfObj):
        self.tree[node] = []

        if (len(vectors) < tfObj.max_size_lev):
            self.imagesInLeaves[node] = []

            for idx, v in enumerate(vectors):

                self.imagesInLeaves[node].append(v)
                v[0].updateLeaf((node, idx))

                if self.debug:
                    self.bar.update()  # fansy progress bar

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
                        self.nodeIndex = self.nodeIndex + 1
                        self.nodes[self.nodeIndex] = vectors[pickedVal[i]]
                        self.tree[node].append(self.nodeIndex)
                        self.process(self.nodeIndex, childIDs[i], tfObj)
