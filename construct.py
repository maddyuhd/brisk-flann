import tensorflow as tf
from random import choice, shuffle
# from numpy import array
from time import time
import numpy as np
import pickle
from dump import brief_brisk as feat
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tree, imagesInLeaves, nodes = {}, {}, {}
nodeIndex = 0
n_clusters = 16
max_size_lev = 150


class images:
    def __init__(self, path,resVal=320 , id = None, link = None):
        self.uuid = id
        self.name = os.path.split(path)[-1][:-4]
        self.s3link = link
        self.path = path
        self.kp, self.des = feat(path, resVal)
        self.leafPos = []  #(index of leaf node, index position)

    def updateLeaf(self, val):
        self.leafPos.append(val)

class timeCal:
    def __init__(self):
        self.start = time()
        self.end = ""

    def result(self,message = "taken"):
        self.end = "T (" + message + "): " + str(time()-self.start) + "sec"
        print self.end

def saveFile(object, name):
    print "saving " + name
    f = open(name + '.pckl', 'wb')
    pickle.dump(object, f)
    f.close()

def openFile(object):
    f = open(object+'.pckl', 'rb')
    return pickle.load(f)

def vecVal(a):
    x, y = a
    return x.des[y]


def constructTree(node, vectors):
    global nodeIndex, nodes, tree, imagesInLeaves
    tree[node] = []

    if (len(vectors) < max_size_lev):
        
        imagesInLeaves[node] = []
        
        for idx, v in enumerate(vectors):
            imagesInLeaves[node].append(v)
            v[0].updateLeaf((node, idx))
        
        print "leaves_node ", node

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
                centroidsVal = [tf.Variable((vecVal(vectors[vector_indices[i]])))
                                for i in range(n_clusters)]

                # centroidPh = tf.placeholder("uint8", [dim])

                v1 = tf.placeholder("float", [dim])
                v2 = tf.placeholder("float", [dim])

                # Euclidean distances
                euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

                centroidPh = tf.placeholder("float", [n_clusters])
                cluster_assignment = tf.argmin(centroidPh, 0)

                init_op = tf.global_variables_initializer()
                sess.run(init_op)

            with tf.device("/gpu:0"):
                print "clustering..."
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
                    nodes[nodeIndex] = vectors[vector_indices[i]]  ##need changes
                    tree[node].append(nodeIndex)
                    constructTree(nodeIndex, childIDs[i])



if __name__ == "__main__":
    start = timeCal()
    features = []
    rootDir = '/home/smacar/Desktop/dev/online/tree/binary_brief/data/1'
    fileList = sorted(os.listdir(rootDir))

    for imgname in fileList:
        img_path = rootDir + '/' + str(imgname)
        img = images(img_path)
        # kp, des = feat(img_path)
        for i in range(len(img.des)):
            features.append((img,i))
        # del kp, des

    print "Size of the Features : ", len(features)

    constructTree(0, features)

    start.result
    saveFile(tree,"tree")
    saveFile(imagesInLeaves,"imagesInLeaves")
    saveFile(nodes,"nodes")
