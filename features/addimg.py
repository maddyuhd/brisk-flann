from tree.construct import randomeCentroid
from view.progress_bar import progress
from db.pick import openFile, saveFile
from info import inlocal
from db.in_memory import Memory

# redis_obj = Memory(pipeline=False)

# redis_obj.get_dict_value("tree")
# redis_obj.get_dict_value("imagesInLeaves")
# nodes = redis_obj.get_dict_value("nodes")
# tree, imagesInLeaves, nodes = redis_obj.pipeline_close(send=True)

tree = openFile("tree", inlocal)
imagesInLeaves = openFile("imagesInLeaves", inlocal)
nodes = openFile("nodes", inlocal)
nodeIndex = openFile("nodeIndex", inlocal)


class add2Db():

    def __init__(self, node, vector, tfObj, debug=False):
        self.tree = tree
        self.imagesInLeaves = imagesInLeaves
        self.nodes = nodes
        self.nodeIndex = nodeIndex
        # self.nodeIndex = None

        if debug:
            self._bar = progress("adding", len(vector))

        self._process(node, vector, tfObj, debug)

        if debug:
            self._bar.finish()

        self.saveme()

    def _process(self, node, vector, tfObj, debug):

        if node in self.imagesInLeaves:

            if (len(self.imagesInLeaves[node]) + len(vector) >=
            # leaf_val = redis_obj.get_dict_value(key="imagesInLeaves",
            #                                     idx=node, entire=False)
            # if (len(leaf_val) + len(vector) >=
                    tfObj.max_size_lev):

                if debug:
                    for _ in range(len(vector)):
                        self._bar.update()
                    debug = False

                self.tree[node] = []
                # redis_obj.set_dict_value(key="tree", val=[],
                #                          idx=node, entire=False)
                combVal = self.imagesInLeaves[node] + vector
                # combVal = leaf_val + vector

                del self.imagesInLeaves[node]
                # redis_obj.delete("imagesInLeaves", node)  # move up

                pickedIdx = randomeCentroid(len(combVal),
                                            tfObj.n_clusters)
                childIDs = tfObj.cluster(combVal, combVal, pickedIdx)

                for idx, val in enumerate(childIDs):
                    self.nodeIndex += 1
                    # self.nodeIndex = redis_obj.get_node_idx()

                    self.tree[node].append(self.nodeIndex)
                    # redis_obj.update_dict_value(key="tree",
                                                # idx=node,
                                                # val=self.nodeIndex)

                    self.nodes[self.nodeIndex] = combVal[pickedIdx[idx]]
                    # redis_obj.set_dict_value(key="nodes",
                                            #  val=combVal[pickedIdx[idx]],
                                            #  idx=self.nodeIndex,
                                            #  entire=False)

                    self.imagesInLeaves[self.nodeIndex] = []
                    # redis_obj.set_dict_value(key="imagesInLeaves",
                                            #  val=[], idx=self.nodeIndex,
                                            #  entire=False)
                    self._process(self.nodeIndex, val, tfObj, debug)

            else:

                for idx, val in enumerate(vector):
                    self.imagesInLeaves[node].append(val)
                    # redis_obj.update_dict_value(key="imagesInLeaves",
                                                # idx=node,
                                                # val=val)
                    if debug:
                        self._bar.update()

        elif node in self.tree:
        # elif redis_obj.if_dict_exists(key="tree", val=node):
            C = tree[node]
            # C = redis_obj.get_dict_value(key="tree",
                                        #  idx=node,
                                        #  entire=False)

            childIDs = tfObj.cluster(vector, nodes, C)

            for idx, val in enumerate(childIDs):
                # print node,idx #clean up
                if val:
                    self.nodeIndex = self.tree[node][idx]
                    # self.nodeIndex = redis_obj.get_dict_value(key="tree",
                                                            #   idx=node,
                                                            #   entire=False)[idx]

                    self._process(self.nodeIndex, val, tfObj, debug)

    def saveme(self):
        saveFile(self.tree, "tree", inlocal)
        saveFile(self.imagesInLeaves, "imagesInLeaves", inlocal)
        saveFile(self.nodes, "nodes", inlocal)
        saveFile(self.nodeIndex, "nodeIndex", inlocal)

        # redis_obj.set_dict_value("tree", self.tree)
        # redis_obj.set_dict_value("imagesInLeaves", self.imagesInLeaves)
        # redis_obj.set_dict_value("nodes", self.nodes)
        # redis_obj.pipeline_close(send=False)
