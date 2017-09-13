import tensorflow as tf
from construct import vecVal, randomeCentroid
from db.pick import openFile, saveFile
from info import inlocal

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


max_size_lev, n_clusters = 500, 8

tree = openFile("tree", inlocal)
imagesInLeaves = openFile("imagesInLeaves", inlocal)
nodes = openFile("nodes", inlocal)
nodeIndex = openFile("nodeIndex", inlocal)


def saveme():
    saveFile(tree, "tree", inlocal)
    saveFile(imagesInLeaves, "imagesInLeaves", inlocal)
    saveFile(nodes, "nodes", inlocal)
    saveFile(nodeIndex, "nodeIndex", inlocal)


def add2Db(node, vector, bar):
    global tree, imagesInLeaves, nodes, nodeIndex

    if node in imagesInLeaves:

        if (len(imagesInLeaves[node]) + len(vector) >= max_size_lev):

            for _ in range(len(vector)):
                bar.update()
            bar = None

            tree[node] = []
            combVal = imagesInLeaves[node] + vector
            del imagesInLeaves[node]

            pickedVal = randomeCentroid(len(combVal), n_clusters)
            dim = 32

            graph = tf.Graph()
            with tf.Session(graph=graph) as sess:
                childIDs = [[] for i in range(n_clusters)]
                centroidsVal = [tf.Variable(
                    vecVal(combVal[i])) for i in pickedVal]

                v1 = tf.placeholder("float", [dim])
                v2 = tf.placeholder("float", [dim])
                euclid_dist = tf.sqrt(tf.reduce_sum(
                    tf.pow(tf.subtract(v1, v2), 2)))

                centroidPh = tf.placeholder("float", [n_clusters])
                cluster_assignment = tf.argmin(centroidPh, 0)

                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                with tf.device("/gpu:0"):

                    for val in (combVal):
                        vect = vecVal(val)

                        distances = [sess.run(euclid_dist, feed_dict={
                            v1: vect, v2: sess.run(centroid)})
                            for centroid in centroidsVal]

                        assignments_val = sess.run(
                            cluster_assignment, feed_dict={
                                centroidPh: distances})

                        childIDs[assignments_val].append(val)

            for idx, val in enumerate(childIDs):
                nodeIndex += 1
                tree[node].append(nodeIndex)
                nodes[nodeIndex] = combVal[pickedVal[idx]]
                imagesInLeaves[nodeIndex] = []
                add2Db(nodeIndex, val, bar)

        else:

            for idx, v in enumerate(vector):
                imagesInLeaves[node].append(v)
                if bar is not None:
                    bar.update()

    else:

        C = tree[node]
        dim = 32

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            childIDs = [[] for i in range(n_clusters)]
            centroidsVal = [tf.Variable(vecVal(nodes[i])) for i in C]

            v1 = tf.placeholder("float", [dim])
            v2 = tf.placeholder("float", [dim])
            euclid_dist = tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(v1, v2), 2)))

            centroidPh = tf.placeholder("float", [n_clusters])
            cluster_assignment = tf.argmin(centroidPh, 0)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            with tf.device("/gpu:0"):

                for vector_n in range(len(vector)):

                    vect = vecVal(vector[vector_n])

                    distances = [sess.run(euclid_dist, feed_dict={
                        v1: vect, v2: sess.run(centroid)})
                        for centroid in centroidsVal]

                    assignments_val = sess.run(cluster_assignment, feed_dict={
                        centroidPh: distances})

                    childIDs[assignments_val].append(vector[vector_n])

        for idx, val in enumerate(childIDs):
            # print node,idx #clean up
            nodeIndex = tree[node][idx]
            add2Db(nodeIndex, val, bar)

# import logging

# loggeing.basicConfig(filename="log/log_add.log", level=logging.WARNING)
# debug, info, warning, error, critical

# try
# except Exception as e:
#     raise e
    # logging.critical("outside %s " % str(e))
