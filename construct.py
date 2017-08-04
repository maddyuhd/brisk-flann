import tensorflow as tf
from random import shuffle, sample
# import numpy as np
import progressbar
import pickle
from dump import brief_brisk as feat
import os
import sys
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tree, imagesInLeaves, nodes = {}, {}, {}
nodeIndex = 0

n_clusters = 8
max_size_lev = 500

# inlocal = True
inlocal = False

# detector = cv2.BRISK_create(50, 1, 1.0)


class images:

    def __init__(self, path, resVal=320,
                 detector=cv2.BRISK_create(50, 1, 1.0)):
        self.name = os.path.split(path)[-1][:-4]
        # self.s3link = link
        self.kp, self.des = feat(path, resVal, detector)
        self.leafPos = []  # (index of leaf node, index position)

    def updateLeaf(self, val):
        self.leafPos.append(val)


def randomeCentroid(K, N):
    return sample(xrange(0, K), N)


def vecVal(a):
    x, y = a
    return x.des[y]


class tfInit:

    def __init__(self, dim, n_clusters):
        self.n_clusters = n_clusters
        self.graph = tf.Graph()

        with tf.Session(graph=self.graph):
            with tf.device("/cpu:0"):
                self.v1 = tf.placeholder("float", [dim])
                self.v2 = tf.placeholder("float", [dim])
                self.euclid_dist = tf.sqrt(tf.reduce_sum(
                    tf.pow(tf.subtract(self.v1, self.v2), 2)))
                self.init_op = tf.global_variables_initializer()

    def setRandomCentroid(self, vectors):
        vector_indices = randomeCentroid(len(vectors), self.n_clusters)

        with tf.Session(graph=self.graph):
            self.centroidPh = tf.placeholder("uint8", [self.n_clusters])
            self.cluster_assignment = tf.argmin(self.centroidPh, 0)
            self.centroidsVal = [tf.Variable((vecVal(
                vectors[vector_indices[i]]))) for i in range(self.n_clusters)]

    def updateRandomCentroid(self, vectors):
        self.vector_indices = randomeCentroid(len(vectors), self.n_clusters)

        # with tf.Session(graph=self.graph) as sess:
        self.centroidsVal = [self.centroidsVal[i].assign((vecVal(
            vectors[self.vector_indices[i]])))
            for i in range(self.n_clusters)]

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
    # print "[INFO] saved " + name
    f.close()


def openFile(object, debug=False):
    if debug:
        f = open("module/" + object + '.pckl', 'rb')
    else:
        f = open("/home/ubuntu/brisk-flann/module/" + object + '.pckl', 'rb')
    return pickle.load(f)


def constructTree(node, vectors, bar):
    global nodeIndex, nodes, tree, imagesInLeaves
    tree[node] = []

    if (len(vectors) < max_size_lev):
        imagesInLeaves[node] = []

        for idx, v in enumerate(vectors):

            imagesInLeaves[node].append(v)
            v[0].updateLeaf((node, idx))

            # fansy output
            bar.update()
        # print "leaves_node ", node

    else:
        dim = len(vecVal(vectors[0]))
        vector_indices = list(range(len(vectors)))
        shuffle(vector_indices)

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
        # with graph.as_default():
            # sess = tf.Session()
            with tf.device("/cpu:0"):
                childIDs = [[] for i in range(n_clusters)]
                centroidsVal = [tf.Variable((
                    vecVal(vectors[vector_indices[i]])))
                    for i in range(n_clusters)]

                centroidPh = tf.placeholder("uint8", [dim])

                v1 = tf.placeholder("float", [dim])
                v2 = tf.placeholder("float", [dim])

                # Euclidean distances
                euclid_dist = tf.sqrt(tf.reduce_sum(
                    tf.pow(tf.subtract(v1, v2), 2)))
                centroidPh = tf.placeholder("float", [n_clusters])
                cluster_assignment = tf.argmin(centroidPh, 0)

                init_op = tf.global_variables_initializer()
                sess.run(init_op)

            with tf.device("/gpu:0"):
                # print "clustering..."
                for vector_n in range(len(vectors)):

                    vect = vecVal(vectors[vector_n])

                    distances = [sess.run(euclid_dist, feed_dict={
                        v1: vect, v2: sess.run(centroid)})
                        for centroid in centroidsVal]

                    assignments_val = sess.run(cluster_assignment, feed_dict={
                        centroidPh: distances})

                    childIDs[assignments_val].append(vectors[vector_n])

                for i in range(n_clusters):
                    nodeIndex = nodeIndex + 1
                    # need_changes !
                    #              v
                    nodes[nodeIndex] = vectors[vector_indices[i]]
                    tree[node].append(nodeIndex)
                    constructTree(nodeIndex, childIDs[i], bar)


if __name__ == "__main__":

    if (inlocal):
        n_clusters = int(sys.argv[1])
        max_size_lev = int(sys.argv[2])

    features = []
    rootDir = 'data/1'
    fileList = sorted(os.listdir(rootDir))

    for imgname in fileList:
        img_path = rootDir + '/' + str(imgname)
        img = images(img_path)
        # kp, des = feat(img_path)
        for i in range(len(img.des)):
            features.append((img, i))
        # del kp, des

    bar = progress("Constructing", len(features))

    # tfObj = tfInit(len(vecVal(features[0])), n_clusters)
    # tfObj.setRandomCentroid(features)
    # tfObj.finalVariable()

    constructTree(0, features, bar)

    bar.finish()
    inlocal = True
    saveFile(tree, "tree", inlocal)
    saveFile(imagesInLeaves, "imagesInLeaves", inlocal)
    saveFile(nodes, "nodes", inlocal)
    # print("[INFO] indexed {} images, {} vectors".format(
    #     len(fileList), len(features)))
