from image.cv_help import features


class images:
    """
    images properties (feature data and uuid)
    """

    def __init__(self, path, resize=320, src=True):
        self.name = path[path.rfind("/") + 1:-4]
        self.kp, self.des = features(path, resize, src)
