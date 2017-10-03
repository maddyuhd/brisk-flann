from random import sample
from db.pick import saveFile


def randomeCentroid(K, N):
    return tuple(sample(xrange(0, K), N))


def vecVal(val):
    obj, idx = val
    return obj.des[idx]


class constructMe():

    def __init__(self, node, vectors, tfObj, debug):

        self.tree = {}
        self.nodes = {}
        self.imagesInLeaves = {}
        self.nodeIndex = 0
        self.disable, self.remove = set(), set()

        self._debug = debug
        if self._debug:
            from view.progress_bar import progress
            self._bar = progress("Constructing", len(vectors))

        self._process(node, vectors, tfObj)

        if self._debug:
            self._bar.finish()

    def _process(self, node, vectors, tfObj):
        self.tree[node] = []

        if (len(vectors) < tfObj.max_size_lev):
            self.imagesInLeaves[node] = []

            for idx, v in enumerate(vectors):
                self.imagesInLeaves[node].append(v)

                if self._debug:
                    self._bar.update()

        else:
            pickedVal = randomeCentroid(len(vectors), tfObj.n_clusters)

            childIDs = tfObj.cluster(vectors, vectors, pickedVal)

            for i in range(tfObj.n_clusters):
                self.nodeIndex += 1
                self.nodes[self.nodeIndex] = vectors[pickedVal[i]]
                self.tree[node].append(self.nodeIndex)
                self._process(self.nodeIndex, childIDs[i], tfObj)

    def saveDb(self, inlocal):
        saveFile(self.tree, "tree", inlocal)
        saveFile(self.imagesInLeaves, "imagesInLeaves", inlocal)
        saveFile(self.nodes, "nodes", inlocal)
        saveFile(self.nodeIndex, "nodeIndex", inlocal)
        saveFile(self.disable, "disable", inlocal)
        saveFile(self.remove, "remove", inlocal)
