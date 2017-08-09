import tensorflow as tf
from random import sample
# import numpy as np
import progressbar
import pickle
from dump import brief_brisk as feat
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tree = {}  # tree structure 'int' eg tree[0]=[1,2,3]
nodes = {}  # centroid values [32] of the tree
imagesInLeaves = {}  # leaf node
nodeIndex = 0

# inlocal = True
inlocal = False


class images:

    def __init__(self, path, resVal=320,
                 detector=cv2.BRISK_create(50, 1, 1.0)):
        self.name = os.path.split(path)[-1][:-4]
        self.kp, self.des = feat(path, resVal, detector)
        self.leafPos = []  # (index of leaf node, index position)

    def updateLeaf(self, val):
        self.leafPos.append(val)


def randomeCentroid(K, N):
    return tuple(sample(xrange(0, K), N))


def vecVal(a):
    x, y = a
    return x.des[y]


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


class progress:
    def __init__(self, msg, max):
        self.count = 0
        self.widgets = [str(msg) + ":", progressbar.Percentage(),
                        " ", progressbar.Bar(), " ", progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(maxval=max, widgets=self.widgets)
        self.pbar.start()

    def update(self):
        self.count += 1
        self.pbar.update(self.count)

    def finish(self):
        self.pbar.finish()


def saveFile(object, name, debug=False):
    if debug:
        f = open("module/" + name + '.pckl', 'wb')
    else:
        f = open("/home/ubuntu/brisk-flann/module/" + name + '.pckl', 'wb')

    pickle.dump(object, f)
    f.close()

    # if debug:
    #     print "[INFO] saved " + name


def openFile(object, debug=False):
    if debug:
        f = open("module/" + object + '.pckl', 'rb')
    else:
        f = open("/home/ubuntu/brisk-flann/module/" + object + '.pckl', 'rb')
    return pickle.load(f)
    # if debug:
    #     print "[INFO] opening " + object


def constructTree(node, vectors, obj, bar):
    global nodeIndex, nodes, tree, imagesInLeaves
    tree[node] = []

    if (len(vectors) < obj.max_size_lev):
        imagesInLeaves[node] = []

        for idx, v in enumerate(vectors):

            imagesInLeaves[node].append(v)
            v[0].updateLeaf((node, idx))

            bar.update()  # fansy progress bar

    else:

        pickedVal = randomeCentroid(len(vectors), obj.n_clusters)

        childIDs = [[] for i in range(obj.n_clusters)]
        centroidsVal = [vecVal(vectors[i])
                        for i in pickedVal]

        with tf.Session(graph=obj.graph) as sess:

            with tf.device("/gpu:0"):
                # clustering...
                for vector_n in range(len(vectors)):

                    vect = vecVal(vectors[vector_n])

                    distances = [sess.run(obj.euclid_dist, feed_dict={
                        obj.v1: vect, obj.v2: centroid})
                        for centroid in centroidsVal]

                    assignments_val = sess.run(
                        obj.cluster_assignment, feed_dict={
                            obj.centroidPh: distances})

                    childIDs[assignments_val].append(vectors[vector_n])

                for i in range(obj.n_clusters):
                    nodeIndex = nodeIndex + 1
                    nodes[nodeIndex] = vectors[pickedVal[i]]
                    tree[node].append(nodeIndex)
                    constructTree(nodeIndex, childIDs[i], obj, bar)
