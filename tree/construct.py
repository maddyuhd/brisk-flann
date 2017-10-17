from random import sample
from db.pick import saveFile


def randomeCentroid(K, N):
    return tuple(sample(xrange(0, K), N))


class constructMe():

    def __init__(self, node, vectors, tfObj, debug):

        self._tree = {}
        self._nodes = {}
        self._imagesInLeaves = {}
        self._nodeIndex = 0
        self._disable, self._remove = set(), set()

        self._debug = debug
        if self._debug:
            from view.progress_bar import progress
            self._bar = progress("Constructing", len(vectors))

        self._process(node, vectors, tfObj)

        if self._debug:
            self._bar.finish()

    def _process(self, node, vectors, tfObj):

        if len(vectors) < tfObj.max_size_lev:
            self._imagesInLeaves[node] = []

            for idx, v in enumerate(vectors):
                self._imagesInLeaves[node].append(v)

                if self._debug:
                    self._bar.update()

        else:
            self._tree[node] = []

            pickedVal = randomeCentroid(len(vectors), tfObj.n_clusters)

            childIDs = tfObj.cluster(vectors, vectors, pickedVal)

            for i in range(tfObj.n_clusters):
                self._nodeIndex += 1
                self._nodes[self._nodeIndex] = vectors[pickedVal[i]]
                self._tree[node].append(self._nodeIndex)
                self._process(self._nodeIndex, childIDs[i], tfObj)

    def saveDb(self, inlocal):
        saveFile(self._tree, "tree", inlocal)
        saveFile(self._imagesInLeaves, "imagesInLeaves", inlocal)
        saveFile(self._nodes, "nodes", inlocal)
        saveFile(self._nodeIndex, "nodeIndex", inlocal)
        saveFile(self._disable, "disable", inlocal)
        saveFile(self._remove, "remove", inlocal)
