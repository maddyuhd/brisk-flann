from image.cv_help import features


class images:

    def __init__(self, path, resVal=320, src=True):
        self.name = path[path.rfind("/") + 1:-4]
        self.kp, self.des = features(path, resVal, src)
