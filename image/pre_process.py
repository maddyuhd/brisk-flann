from image.cv_help import briefBrisk as feat


class images:

    def __init__(self, path, resVal=320, src=True):
        # self.name = os.path.split(path)[-1][:-4]
        self.name = path[path.rfind("/") + 1:-4]
        self.kp, self.des = feat(path, resVal, src)
        self.leafPos = []  # (index of leaf node, index position)

    def updateLeaf(self, val):
        self.leafPos.append(val)
