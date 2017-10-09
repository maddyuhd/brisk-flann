"""
Indexer for handling Data structure
"""

from image.pre_process import images

def index(imagList):
    features = []

    for img_path in imagList:
        img = images(img_path)

        for i in range(len(img.des)):
            features.append((img, i))

    return features


def getVal(val):
    obj, idx = val
    return obj.des[idx]
