import tensorflow as tf
from features.construct import vecVal

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class tfInit:

    def __init__(self, n_clusters, max_size_lev=None, dim=32):
        self.n_clusters = n_clusters
        self.max_size_lev = max_size_lev
        self.graph = tf.Graph()

        with tf.Session(graph=self.graph):
            with tf.device("/cpu:0"):
                self.v1 = tf.placeholder("float", [dim])
                self.v2 = tf.placeholder("float", [dim])
                self.euclid_dist = tf.sqrt(tf.reduce_sum(
                    tf.pow(tf.subtract(self.v1, self.v2), 2)))
                self.init_op = tf.global_variables_initializer()

    def clusterVar(self):
        with tf.Session(graph=self.graph):
            with tf.device("/cpu:0"):
                self.centroidPh = tf.placeholder("float", [self.n_clusters])
                self.cluster_assignment = tf.argmin(self.centroidPh, 0)

    def finalVariable(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init_op)

    def cluster(self, totalVal, sampleVal, pickedIdx):
        with tf.Session(graph=self.graph) as sess:
            childIDs = [[] for i in range(self.n_clusters)]
            centroidsVal = [vecVal(sampleVal[i])
                            for i in pickedIdx]
            with tf.device("/gpu:0"):

                for val in (totalVal):
                    vect = vecVal(val)

                    distances = [sess.run(self.euclid_dist,
                                 feed_dict={self.v1: vect,
                                            self.v2: centroid})
                                 for centroid in centroidsVal]

                    assignments_val = sess.run(
                        self.cluster_assignment,
                        feed_dict={self.centroidPh: distances})

                    childIDs[assignments_val].append(val)

        return childIDs
