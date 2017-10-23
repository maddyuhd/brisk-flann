"""
Indexer for handling Data structure
"""

from image.pre_process import images

def index(imgList):
    features = []

    for img_path in imgList:
        img = images(img_path)

        for i in range(len(img.des)):
            features.append((img, i))

    return features


def getVal(val):
    obj, idx = val
    return obj.des[idx]
