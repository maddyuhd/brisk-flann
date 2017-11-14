from random import sample
from db.pick import saveFile
from features.info import inlocal

def randomeCentroid(K, N):
    return tuple(sample(xrange(0, K), N))


class ConstructMe(object):

    def __init__(self, vectors, tfObj, debug):

        self._tree = {}
        self._nodes = {}
        self._imagesInLeaves = {}
        self._nodeIndex = 0
        self._tf = tfObj
        self._debug = debug

        if self._debug:
            from view.progress_bar import progress
            self._bar = progress("Constructing", len(vectors))

        self._process(self._nodeIndex, vectors)

        self._saveDb()

        if self._debug:
            self._bar.finish()

    def _process(self, node, vectors):

        if len(vectors) < self._tf.max_size_lev:
            self._imagesInLeaves[node] = []

            for val in vectors:
                self._imagesInLeaves[node].append(val)

                if self._debug:
                    self._bar.update()

        else:
            self._tree[node] = []

            pickedVal = randomeCentroid(len(vectors), self._tf.n_clusters)

            childIDs = self._tf.cluster(vectors, vectors, pickedVal)

            for i in range(self._tf.n_clusters):
                self._nodeIndex += 1
                self._nodes[self._nodeIndex] = vectors[pickedVal[i]]
                self._tree[node].append(self._nodeIndex)
                self._process(self._nodeIndex, childIDs[i])

    def _saveDb(self):
        saveFile(self._tree, "tree", inlocal)
        saveFile(self._imagesInLeaves, "imagesInLeaves", inlocal)
        saveFile(self._nodes, "nodes", inlocal)
        saveFile(self._nodeIndex, "nodeIndex", inlocal)
