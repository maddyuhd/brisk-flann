import tensorflow as tf
from construct import vecVal, randomeCentroid
# from construct import randomeCentroid
from view.progress_bar import progress
from db.pick import openFile, saveFile
from info import inlocal

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# max_size_lev, n_clusters = 500, 8

tree = openFile("tree", inlocal)
imagesInLeaves = openFile("imagesInLeaves", inlocal)
nodes = openFile("nodes", inlocal)
nodeIndex = openFile("nodeIndex", inlocal)


# def saveme():
#     saveFile(tree, "tree", inlocal)
#     saveFile(imagesInLeaves, "imagesInLeaves", inlocal)
#     saveFile(nodes, "nodes", inlocal)
#     saveFile(nodeIndex, "nodeIndex", inlocal)


class add2Db():

    def __init__(self, node, vector, tfObj, debug=False):
        self.tree = tree
        self.imagesInLeaves = imagesInLeaves
        self.nodes = nodes
        self.nodeIndex = nodeIndex

        if debug:
            self.bar = progress("adding", len(vector))

        self.process(node, vector, tfObj, debug)

        if debug:
            self.bar.finish()

        self.saveme()

    def process(self, node, vector, tfObj, debug):

        if node in self.imagesInLeaves:
            if (len(self.imagesInLeaves[node]) + len(vector) >=
                    tfObj.max_size_lev):

                if debug:
                    for _ in range(len(vector)):
                        self.bar.update()
                    debug = False

                self.tree[node] = []
                combVal = self.imagesInLeaves[node] + vector
                del self.imagesInLeaves[node]

                pickedIdx = randomeCentroid(len(combVal),
                                            tfObj.n_clusters)
                childIDs = tfObj.cluster(combVal, combVal, pickedIdx)

                for idx, val in enumerate(childIDs):
                    self.nodeIndex += 1
                    self.tree[node].append(self.nodeIndex)
                    self.nodes[self.nodeIndex] = combVal[pickedIdx[idx]]
                    self.imagesInLeaves[self.nodeIndex] = []
                    self.process(self.nodeIndex, val, tfObj, debug)

            else:

                for idx, v in enumerate(vector):
                    self.imagesInLeaves[node].append(v)
                    if debug:
                        self.bar.update()

        elif node in self.tree:
            C = tree[node]

            childIDs = tfObj.cluster(vector, nodes, C)

            for idx, val in enumerate(childIDs):
                # print node,idx #clean up
                if val:
                    self.nodeIndex = self.tree[node][idx]
                    self.process(self.nodeIndex, val, tfObj, debug)

        else:
            print "bla...bla..."

    def saveme(self):
        saveFile(self.tree, "tree", inlocal)
        saveFile(self.imagesInLeaves, "imagesInLeaves", inlocal)
        saveFile(self.nodes, "nodes", inlocal)
        saveFile(self.nodeIndex, "nodeIndex", inlocal)


# import logging

# loggeing.basicConfig(filename="log/log_add.log", level=logging.WARNING)
# debug, info, warning, error, critical

# try
# except Exception as e:
#     raise e
    # logging.critical("outside %s " % str(e))
